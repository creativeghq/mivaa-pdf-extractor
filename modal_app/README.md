# PaddleOCR-VL on Modal — GPU host for the pipeline's structural pass

This folder deploys **PaddleOCR-VL** (`PaddlePaddle/PaddleOCR-VL-1.6`, 0.9B) to
**Modal** as the catalog pipeline's **layout + OCR backbone**. It's a **two-stage**
document parser run **in one container** by the `paddleocr[doc-parser]` package:

1. **PP-DocLayoutV2** (RT-DETR detector + pointer network) → per-region bounding
   boxes, element labels, and reading order.
2. **PaddleOCR-VL-0.9B** (NaViT encoder + ERNIE-4.5-0.3B) → recognizes the content
   inside each region (text, tables→markdown, formulas→LaTeX, charts).

It **replaced Surya-2** (2026-06-13): the dedicated RT-DETR boxes are tighter
(→ cleaner product crops → better SLIG visual embeddings) and reading order comes
from a dedicated model. This is **NOT** a vLLM `/v1/chat/completions` server — vLLM
serving of PaddleOCR-VL needs nightly builds and only covers the VLM half. We run
the full `PaddleOCRVL` pipeline in-process (`paddlepaddle-gpu`, no vLLM) and expose
a small custom contract that the MIVAA `PaddleOCRManager` speaks:

```
GET  /health           → 200 once models are loaded (unauth; the warmup probe)
POST /parse  (bearer)  → {"image_b64": "...", "mode": "page"|"block"}
                         → {"regions":[{bbox:[x0,y0,x1,y1] px, label, content, order}],
                            "width", "height"}
```

`mode=page` is the structural pass; `mode=block` is per-crop OCR.

Lifecycle: scale-to-zero (`min_containers=0` + `scaledown_window=120`) so it costs
**$0 idle**; the first request after idle cold-starts a GPU container (~90s, model
load + first-call JIT); MIVAA health-probes `/health` as its warmup. GPU is `L4`,
autoscale ceiling `max_containers=4`.

**Already deployed**: app `paddleocr-vl` at
`https://basilakis--paddleocr-vl-paddleservice-web.modal.run` (workspace
`basilakis`). The URL is baked as the `paddleocr_modal_url` config default, so the
**only runtime secret you must set on MIVAA is `PADDLEOCR_MODAL_API_KEY`**. The
steps below are for a fresh deploy / redeploy.

---

## Keys & secrets — quick reference

There are **two separate** secret sets. One lets **MIVAA call** the Modal endpoint;
the other lets **GitHub Actions deploy** the endpoint to Modal.

### A. MIVAA → Modal (runtime) — GitHub Actions secrets on `creativeghq/mivaa-pdf-extractor`

These feed the systemd unit (`deploy.yml` `Environment=` lines). Add at
**Settings → Secrets and variables → Actions**.

| GitHub secret | Required? | Value |
|---|---|---|
| `PADDLEOCR_MODAL_API_KEY` | ✅ required | The exact bearer string stored in the `paddleocr-api-key` Modal secret (the `PADDLEOCR_API_KEY=` value you set at deploy step 1). If you don't have it, rotate it — see "Rotate the bearer" below. |
| `PADDLEOCR_MODAL_URL` | optional | Override the baked default. Only set if you redeploy under a different app/workspace name. Default: `https://basilakis--paddleocr-vl-paddleservice-web.modal.run` |
| `PADDLEOCR_ENABLED` | optional | `true`/`false`. Empty → code default (enabled). |

### B. GitHub Actions → Modal (CI deploy) — same repo's Actions secrets

These let the `deploy-modal` job run `modal deploy` on your behalf. Create the token
pair at **https://modal.com/settings/tokens** (or `modal token new`, then read
`~/.modal.toml`).

| GitHub secret | Required? | Value |
|---|---|---|
| `MODAL_TOKEN_ID` | ✅ required for CI deploy | The `token_id` (starts `ak-…`) from your Modal API token. |
| `MODAL_TOKEN_SECRET` | ✅ required for CI deploy | The `token_secret` (starts `as-…`) from the same token. |

> These values are secrets that only **you / Modal** can mint — they can't be read
> out of the codebase. Generate them in the Modal dashboard and paste them into
> GitHub.

---

## CI auto-deploy (no manual `modal deploy` needed)

The MIVAA workflow (`.github/workflows/deploy.yml`) has a **`deploy-modal`** job
that runs **in parallel** with the server deploy and fires `modal deploy
modal_app/paddleocr_vl.py` **only when `modal_app/**` changed** on the push (a
`dorny/paths-filter` gate). So editing `paddleocr_vl.py` and pushing to `main`
redeploys the Modal endpoint automatically.

- It authenticates with `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (set B above). If
  those secrets are missing, the step fails fast with a pointer to the token page.
- To force a redeploy with no `modal_app/**` change: run the workflow via
  **Actions → Run workflow → `force_modal_deploy: true`** (`workflow_dispatch`).
- The endpoint's own bearer (the `paddleocr-api-key` Modal secret) is referenced by
  the app at deploy time and **never travels through CI**.

---

## One-time / manual deploy

### 0. Prerequisites
```bash
pip install modal
modal token new        # authenticate the CLI to your Modal workspace
```

### 1. Create the API-key secret (the bearer the endpoint requires)
Pick a strong random string — this becomes `PADDLEOCR_MODAL_API_KEY` on the MIVAA side.
```bash
modal secret create paddleocr-api-key PADDLEOCR_API_KEY=$(openssl rand -hex 32)
```

### 2. Deploy
```bash
modal deploy modal_app/paddleocr_vl.py
```
Modal prints a public URL like:
```
https://<your-workspace>--paddleocr-vl-paddleservice-web.modal.run
```
That URL is **`PADDLEOCR_MODAL_URL`**. The `PADDLEOCR_API_KEY` value you set in
step 1 is **`PADDLEOCR_MODAL_API_KEY`**.

### 3. Point MIVAA at Modal
`PADDLEOCR_MODAL_URL` is baked to the deployed app, so the **only** secret you must
add is the bearer key:
```
PADDLEOCR_MODAL_API_KEY=<the value from step 1>
```
Add it as a GitHub Actions secret (→ `deploy.yml` `Environment=` line) or via the
platform `Settings → Keys` DB fallback, then redeploy MIVAA. The next PDF job warms
+ drives PaddleOCR-VL on Modal.

### Rotate the bearer
```bash
modal secret create paddleocr-api-key PADDLEOCR_API_KEY=$(openssl rand -hex 32)  # overwrites
modal deploy modal_app/paddleocr_vl.py                                            # picks up new value
```
Then update the `PADDLEOCR_MODAL_API_KEY` GitHub secret to the new value and redeploy MIVAA.

---

## Verify the endpoint directly
```bash
# health is open (no key) — this is what MIVAA's warmup probe hits
curl https://<your-workspace>--paddleocr-vl-paddleservice-web.modal.run/health

# the /parse route requires the bearer
curl -X POST https://<your-workspace>--paddleocr-vl-paddleservice-web.modal.run/parse \
  -H "Authorization: Bearer $PADDLEOCR_MODAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"<base64 png>","mode":"page"}'
```
The first call after an idle period cold-starts a GPU container (~90 s incl. model
load + first-call JIT into the persistent volume); subsequent calls are hot (~1–3s/page).

---

## Tuning (env vars at `modal deploy` time)
All are read in `paddleocr_vl.py`:

| Env | Default | Meaning |
|---|---|---|
| `PADDLEOCR_GPU` | `L4` | GPU type (`L4` 24 GB fits the 0.9B VLM + detector; bump for higher concurrency) |
| `PADDLEOCR_CUDA_TAG` | `12.6.3-devel-ubuntu22.04` | CUDA base image (matches the cu126 paddle wheel) |
| `PADDLEOCR_SCALEDOWN_WINDOW` | `120` | Idle seconds before the GPU container drains to $0 |
| `PADDLEOCR_MIN_CONTAINERS` | `0` | `1` = keep one replica always warm (no cold starts, costs ~1 GPU/h) |
| `PADDLEOCR_MAX_CONTAINERS` | `4` | Burst ceiling |
| `PADDLEOCR_MAX_CONCURRENT` | `16` | Max in-flight requests per container (keep > 1 so `/health` never queues behind `/parse`) |
| `PADDLEOCR_PADDLE_VERSION` | `3.2.1` | `paddlepaddle-gpu` version |

Example — keep one replica always hot:
```bash
PADDLEOCR_MIN_CONTAINERS=1 modal deploy modal_app/paddleocr_vl.py
```

## Operate
```bash
modal app logs paddleocr-vl        # stream server logs
modal app list                     # see deployed apps + status
modal app stop paddleocr-vl        # tear the deployment down
```

---

# SLIG (SigLIP2) on Modal — GPU host for visual embeddings

`slig.py` deploys **SLIG** — the platform's visual encoder — to Modal, the same
way PaddleOCR-VL is deployed. SLIG produces the 768D `visual` vector in the
7-embedding fusion search (`vecs.image_slig_embeddings`) and also serves zero-shot
classification + image⇄text similarity. The model is the public HF repo
`basiliskan/slig`, which is a **verbatim duplication of
`google/siglip2-base-patch16-512`** — stock SigLIP2 base, **native 768D, no
projection head** (older docstrings saying "SO400M / 1152→768 projection" are
wrong; `config.json` is `model_type: "siglip"` with no `auto_map`).

It exposes the **same `{inputs, parameters}` contract** the HF Inference Endpoint
used (see [`docs/api/slig-inference.md`](../../docs/api/slig-inference.md)), so the
MIVAA SLIG client's payload builders don't change:

```
GET  /health           → 200 once the model is loaded + warmed (unauth probe)
POST /infer  (bearer)  → {"inputs": <str|[str]|{image(s),text(s)}>,
                          "parameters": {"mode": "...", "candidate_labels": [...]}}
   mode ∈ {zero_shot, image_embedding, text_embedding, similarity, auto}
   → image_embedding / text_embedding: [{"embedding": [768 floats]}, ...]
   → zero_shot:  [{"label","score"}, ...] (single) | [[...], ...] (batch)
   → similarity: {"similarity_scores": [[...]], "image_count", "text_count"}
```

**Embedding parity:** the model is loaded exactly as the HF custom `handler.py`
did (stock `transformers` `AutoModel.from_pretrained` + `AutoProcessor`, then
`get_*_features` → L2-norm), so newly-embedded vectors are bit-for-bit comparable
with everything already in `vecs.image_slig_embeddings`. **Validate before
cutover** by re-embedding a few known images and confirming cosine ≈ 1.0 against
their stored vectors — that's the migration go/no-go.

**Warm posture:** ships **scale-to-zero** (`SLIG_MIN_CONTAINERS=0`, $0 idle), same
as PaddleOCR. The model is small (~400M / 1.5 GB, baked into the image), so a cold
start is load-only (~5–15s), no network download. ⚠️ Unlike PaddleOCR (batch-only),
SLIG also serves **realtime search queries** — with scale-to-zero the first query
after idle eats the cold start, so the MIVAA search path must warm-probe before the
embed and degrade gracefully on timeout (drop the visual vector, keep the rest).
Set `SLIG_MIN_CONTAINERS=1` to keep one replica always warm if query latency wins
over $0 idle.

## Deploy

```bash
# No new secret needed — the app reuses the existing `paddleocr-api-key` Modal
# secret (one shared bearer across both apps; only the URL differs per app).
modal deploy modal_app/slig.py
```

Modal prints `https://<workspace>--slig-sligservice-web.modal.run` — that's
**`SLIG_MODAL_URL`**. The bearer is the **same value as `PADDLEOCR_MODAL_API_KEY`**
(reused from the shared `paddleocr-api-key` secret), so on the MIVAA side set
`SLIG_MODAL_API_KEY` to that same value. To give SLIG its own key later, create a
`slig-api-key` secret exposing `SLIG_API_KEY`, add it to `secrets=[...]` in
`slig.py`, and it takes precedence.

CI auto-deploys via the existing `deploy-modal` job (the `dorny/paths-filter` gate
fires on any `modal_app/**` change, so editing `slig.py` and pushing redeploys it
alongside PaddleOCR). MIVAA wiring (`SLIG_MODAL_URL` / `SLIG_MODAL_API_KEY` env,
provider switch) lands in the platform-integration phase — this folder is the
standalone module only.

## Verify directly

```bash
curl https://<workspace>--slig-sligservice-web.modal.run/health
curl -X POST https://<workspace>--slig-sligservice-web.modal.run/infer \
  -H "Authorization: Bearer $SLIG_MODAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs":"a photo of ceramic floor tiles","parameters":{"mode":"text_embedding"}}'
```

## Tuning (env vars at `modal deploy` time)

| Env | Default | Meaning |
|---|---|---|
| `SLIG_GPU` | `L4` | GPU type (`T4` 16 GB is cheaper and fits the ~400M model; `L4`/`A10G` give more batch throughput on big catalog ingests) |
| `SLIG_SCALEDOWN_WINDOW` | `120` | Idle seconds before the GPU container drains to $0 |
| `SLIG_MIN_CONTAINERS` | `0` | `1` = keep one replica always warm (no cold starts on search; costs ~1 GPU/h) |
| `SLIG_MAX_CONTAINERS` | `4` | Burst ceiling for ingest |
| `SLIG_MAX_CONCURRENT` | `16` | Max in-flight requests per container (keep > 1 so `/health` never queues behind `/infer`) |
| `SLIG_MODEL_REPO` | `basiliskan/slig` | Source model repo (public; pulled into the image at build) |
| `SLIG_TORCH_VERSION` | `2.5.1` | torch CUDA wheel version |
| `SLIG_TRANSFORMERS_VERSION` | `4.49.0` | transformers version (siglip2 needs ≥ 4.49; pinned for forward-pass reproducibility) |

## Operate

```bash
modal app logs slig
modal app stop slig
```
