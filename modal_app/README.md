# Surya-2 on Modal — GPU host for the pipeline's structural pass

This folder deploys **Surya-2** (`datalab-to/surya-ocr-2`) to **Modal** as an
OpenAI-compatible **vLLM** endpoint. It's an alternative GPU host to HuggingFace
Inference Endpoints — picked because HF's serverless pool couldn't reliably
allocate a GPU for us. The MIVAA backend talks the **identical**
`/v1/chat/completions` contract to either host, so moving Surya between them is a
config switch (`SURYA_PROVIDER`), not a code change.

| | HuggingFace | Modal |
|---|---|---|
| Switch | `SURYA_PROVIDER=huggingface` (default) | `SURYA_PROVIDER=modal` |
| Lifecycle | SDK resume / poll-for-GPU / scale-to-zero | Modal owns autoscaling |
| Warmup (MIVAA side) | SDK resume + `/health` probe | `/health` probe (Modal auto-wakes) |
| Scale-to-zero | explicit SDK call at job end | `scaledown_window` idle clock |
| Cost when idle | $0 (scaled to zero) | $0 (container drained) |

The MIVAA lifecycle code (`app/services/pdf/endpoint_providers.py`) implements
both as `EndpointProvider`s; `SuryaEndpointManager` just delegates to whichever
the config selects.

---

## One-time deploy

### 0. Prerequisites
```bash
pip install modal
modal token new        # authenticate the CLI to your Modal workspace
```

### 1. Create the API-key secret (the bearer the endpoint requires)
Pick a strong random string — this becomes `SURYA_MODAL_API_KEY` on the MIVAA side.
```bash
modal secret create surya-vllm-api-key SURYA_VLLM_API_KEY=$(openssl rand -hex 32)
```
> If `datalab-to/surya-ocr-2` is gated on HuggingFace, also create an HF token
> secret and add it to the `secrets=[...]` list in `surya_vllm.py`:
> ```bash
> modal secret create huggingface-token HF_TOKEN=hf_xxx
> ```

### 2. Deploy
```bash
modal deploy modal_app/surya_vllm.py
```
Modal prints a public URL like:
```
https://<your-workspace>--surya-vllm-serve.modal.run
```
That URL is **`SURYA_MODAL_URL`**. The `SURYA_VLLM_API_KEY` value you set in step 1
is **`SURYA_MODAL_API_KEY`**.

### 3. Point MIVAA at Modal
Set these on the MIVAA server (GitHub Actions secrets → `deploy.yml` `Environment=`
lines, or the platform `Settings → Keys` DB fallback):
```
SURYA_PROVIDER=modal
SURYA_MODAL_URL=https://<your-workspace>--surya-vllm-serve.modal.run
SURYA_MODAL_API_KEY=<the value from step 1>
```
Redeploy MIVAA (or let autodeploy run). The next PDF job warms + drives Surya on
Modal. To roll back to HuggingFace, set `SURYA_PROVIDER=huggingface`.

---

## Verify the endpoint directly
```bash
# health is open (no key) — this is what MIVAA's warmup probe hits
curl https://<your-workspace>--surya-vllm-serve.modal.run/health

# the chat route requires the bearer
curl https://<your-workspace>--surya-vllm-serve.modal.run/v1/models \
  -H "Authorization: Bearer $SURYA_MODAL_API_KEY"
```
The first call after an idle period cold-starts a GPU container (~30–90 s incl.
the one-time model download into the persistent volume); subsequent calls are hot.

---

## Tuning (env vars at `modal deploy` time)
All are read in `surya_vllm.py`:

| Env | Default | Meaning |
|---|---|---|
| `SURYA_GPU` | `L4` | GPU type (`L4` 24 GB is plenty for a 650M VLM; `A10G`/`L40S`/`A100` for higher concurrency) |
| `SURYA_SCALEDOWN_WINDOW` | `120` | Idle seconds before the GPU container drains to $0 |
| `SURYA_MIN_CONTAINERS` | `0` | `1` = keep one replica always warm (no cold starts, costs ~1 GPU/h) |
| `SURYA_MAX_CONTAINERS` | `2` | Burst ceiling |
| `SURYA_MAX_CONCURRENT` | `16` | Max in-flight requests per container |
| `SURYA_VLLM_VERSION` | `0.20.1` | vLLM image tag (pinned to Surya-2's tested version) |
| `SURYA_MAX_MODEL_LEN` | `16384` | vLLM context length |
| `SURYA_GPU_MEM_UTIL` | `0.90` | vLLM GPU memory fraction |

Example — keep one replica always hot on an A10G:
```bash
SURYA_GPU=A10G SURYA_MIN_CONTAINERS=1 modal deploy modal_app/surya_vllm.py
```

## Operate
```bash
modal app logs surya-vllm        # stream server logs
modal app list                   # see deployed apps + status
modal app stop surya-vllm        # tear the deployment down
```
