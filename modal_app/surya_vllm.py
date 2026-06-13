"""
Modal deployment: Surya-2 (``datalab-to/surya-ocr-2``) as an OpenAI-compatible
vLLM endpoint — the GPU host for the MIVAA PDF pipeline's structural pass.

This serves the SAME ``/v1/chat/completions`` contract the MIVAA backend already
speaks to HuggingFace, so switching the pipeline to Modal is THREE env vars on
the MIVAA server (see README.md):

    SURYA_PROVIDER=modal
    SURYA_MODAL_URL=<the https URL `modal deploy` prints>
    SURYA_MODAL_API_KEY=<value of the surya-vllm-api-key Modal secret>

Lifecycle parity with the HF endpoint (so the MIVAA SuryaEndpointManager's
warmup / resume / scale-to-zero all behave the same):

  * Scale-to-zero — ``scaledown_window`` drains the GPU container after idle, so
    it costs $0/h when no catalogs are processing (Modal bills per-second of
    container runtime). Default 2 min idle.
  * Auto-wake — the first request after a drain cold-starts a container; the
    MIVAA manager health-probes ``/health`` until it is up (its "warmup").
  * Always-warm option — set SURYA_MIN_CONTAINERS=1 at deploy to keep one GPU
    replica hot and skip cold starts entirely (costs ~1 GPU/h continuously).

Deploy:
    modal secret create surya-vllm-api-key SURYA_VLLM_API_KEY=<random-strong-key>
    modal deploy modal_app/surya_vllm.py

The printed ``https://<workspace>--surya-vllm-serve.modal.run`` URL is
SURYA_MODAL_URL. The ``<random-strong-key>`` you chose is SURYA_MODAL_API_KEY.
"""

import os
import subprocess

import modal

# --------------------------------------------------------------------------- #
# Tunables — override at deploy time via env (e.g. `SURYA_GPU=A10G modal deploy …`)
# --------------------------------------------------------------------------- #
MODEL_REPO = os.environ.get("SURYA_MODEL_REPO", "datalab-to/surya-ocr-2")
# Must match SURYA_MODAL_MODEL_NAME on the MIVAA side (default "surya-ocr-2").
SERVED_MODEL_NAME = os.environ.get("SURYA_SERVED_MODEL_NAME", "surya-ocr-2")
# A 650M VLM fits comfortably on a 24 GB L4 (Modal's cheapest workable GPU for
# vision). Bump to "A10G" / "L40S" / "A100" if you raise concurrency a lot.
GPU = os.environ.get("SURYA_GPU", "L4")
VLLM_PORT = 8000
# Idle seconds before Modal drains the GPU container → $0/h. Short = cheaper,
# longer = fewer cold starts between back-to-back catalogs.
SCALEDOWN_WINDOW = int(os.environ.get("SURYA_SCALEDOWN_WINDOW", "120"))
# 0 = pure scale-to-zero (cheapest, $0 when idle). 1 = keep one replica always hot.
MIN_CONTAINERS = int(os.environ.get("SURYA_MIN_CONTAINERS", "0"))
# Burst ceiling — Modal autoscales 0 → MAX_CONTAINERS under load (more concurrent
# pages / parallel jobs spin more GPU replicas, then drain back to 0 when idle).
MAX_CONTAINERS = int(os.environ.get("SURYA_MAX_CONTAINERS", "4"))
# Requests one container batches before Modal adds another replica. vLLM does
# continuous batching, so one GPU already overlaps several pages.
MAX_CONCURRENT = int(os.environ.get("SURYA_MAX_CONCURRENT", "16"))
# Pin to the vLLM version Surya-2 is tested against (datalab's bench uses 0.20.1).
VLLM_VERSION = os.environ.get("SURYA_VLLM_VERSION", "0.20.1")
CUDA_TAG = os.environ.get("SURYA_CUDA_TAG", "12.8.1-devel-ubuntu22.04")
MAX_MODEL_LEN = os.environ.get("SURYA_MAX_MODEL_LEN", "16384")
GPU_MEM_UTIL = os.environ.get("SURYA_GPU_MEM_UTIL", "0.90")

# Modal's documented vLLM recipe: a CUDA *devel* base (provides nvcc for vLLM's
# runtime cpp_extension compile) + a clean standalone Python via add_python, then
# pip-install vLLM into it. We deliberately do NOT reuse the prebuilt
# `vllm/vllm-openai` image — add_python there overwrites the image's own Python
# and breaks vLLM's deps, and without add_python Modal can't detect its Python.
vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persist weights + the vLLM compile cache across cold starts (download once).
hf_cache = modal.Volume.from_name("surya-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("surya-vllm-cache", create_if_missing=True)

app = modal.App("surya-vllm")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=24 * 60 * 60,  # request timeout ceiling; the server itself is long-lived
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    # `surya-vllm-api-key` → SURYA_VLLM_API_KEY (the bearer vLLM requires on /v1/*).
    # If `datalab-to/surya-ocr-2` is gated on HF, also create a `huggingface-token`
    # secret (HF_TOKEN=...) and add `modal.Secret.from_name("huggingface-token")`.
    secrets=[modal.Secret.from_name("surya-vllm-api-key")],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    """Launch the vLLM OpenAI-compatible server. ``/v1/chat/completions`` is
    bearer-guarded by ``--api-key``; ``/health`` stays open (the MIVAA warmup
    probe relies on that)."""
    api_key = os.environ["SURYA_VLLM_API_KEY"]
    cmd = [
        "vllm", "serve", MODEL_REPO,
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--api-key", api_key,
        "--trust-remote-code",
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
    ]
    print("🚀 starting:", " ".join(c for c in cmd if c != api_key))
    subprocess.Popen(cmd)
