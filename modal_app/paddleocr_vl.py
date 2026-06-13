"""
Modal deployment: PaddleOCR-VL as the catalog pipeline's structural-pass host.

PaddleOCR-VL is a **two-stage** document parser run **in one process** by the
``paddleocr`` package:
  1. PP-DocLayoutV2 (RT-DETR detector + pointer network) → per-region bounding
     boxes, element labels, and reading order.
  2. PaddleOCR-VL-0.9B (NaViT encoder + ERNIE-4.5-0.3B) → recognizes the content
     inside each region (text, tables→markdown, formulas→LaTeX, charts).

It replaced Surya-2 (2026-06-13) because the dedicated RT-DETR detector gives
tighter figure/image boxes (→ cleaner product crops) and a dedicated reading
order, and the VLM adds table/formula/chart recognition.

Unlike the Surya app this is NOT a vLLM ``/v1/chat/completions`` server — vLLM
serving of PaddleOCR-VL needs nightly builds and only covers the VLM half. We
run the full ``PaddleOCRVL`` pipeline in-process (paddlepaddle-gpu, no vLLM) and
expose a small custom contract the MIVAA PaddleOCRManager speaks:

  GET  /health           → 200 once models are loaded (unauth; the warmup probe)
  POST /parse  (bearer)  → {"image_b64": "...", "mode": "page"|"block"}
                           → {"regions":[{"bbox":[x0,y0,x1,y1] px,"label","content",
                                          "order"}], "width", "height"}

Lifecycle parity with the Surya app: scale-to-zero (``min_containers=0`` +
``scaledown_window``) so it costs $0 idle; first request cold-starts a GPU
container; MIVAA health-probes ``/health`` as its warmup.

Deploy:
    modal secret create paddleocr-api-key PADDLEOCR_API_KEY=<random-strong-key>
    modal deploy modal_app/paddleocr_vl.py
The printed URL is PADDLEOCR_MODAL_URL; the key is PADDLEOCR_MODAL_API_KEY.
"""

import base64
import io
import os

import modal

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
GPU = os.environ.get("PADDLEOCR_GPU", "L4")               # 24 GB; VLM ~2-4 GB + small detector
CUDA_TAG = os.environ.get("PADDLEOCR_CUDA_TAG", "12.6.3-devel-ubuntu22.04")  # matches cu126 wheel
SCALEDOWN_WINDOW = int(os.environ.get("PADDLEOCR_SCALEDOWN_WINDOW", "120"))  # idle → $0
MIN_CONTAINERS = int(os.environ.get("PADDLEOCR_MIN_CONTAINERS", "0"))        # 0 = scale-to-zero
MAX_CONTAINERS = int(os.environ.get("PADDLEOCR_MAX_CONTAINERS", "4"))        # autoscale ceiling
# Keep this comfortably > 1 so /health probes are answered instantly and never
# queue behind in-flight /parse work (max_inputs=1 caused a /health pile-up).
MAX_CONCURRENT = int(os.environ.get("PADDLEOCR_MAX_CONCURRENT", "16"))
PADDLE_VERSION = os.environ.get("PADDLEOCR_PADDLE_VERSION", "3.2.1")

# Build: CUDA devel base + standalone Python, paddlepaddle-gpu from Paddle's cu126
# index, then paddleocr[doc-parser] (the PaddleOCR-VL pipeline) from PyPI.
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.11")
    .entrypoint([])
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "ccache")
    .pip_install(
        f"paddlepaddle-gpu=={PADDLE_VERSION}",
        index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .pip_install("paddleocr[doc-parser]", "fastapi[standard]", "pillow", "numpy")
    # PaddleX caches downloaded models under ~/.paddlex; point it at the volume.
    .env({"PADDLE_PDX_CACHE_HOME": "/root/.paddlex"})
)

# Persist the downloaded PP-DocLayoutV2 + PaddleOCR-VL weights across cold starts.
weights = modal.Volume.from_name("paddleocr-weights", create_if_missing=True)

app = modal.App("paddleocr-vl")


@app.cls(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=600,
    volumes={"/root/.paddlex": weights},
    secrets=[modal.Secret.from_name("paddleocr-api-key")],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class PaddleService:
    @modal.enter()
    def load(self):
        """Load the full PaddleOCR-VL pipeline once per container (layout + VLM).

        CRITICAL: force the GPU. PaddleOCR pipelines default to CPU unless
        ``device="gpu"`` is set — a 0.9B VLM on CPU is minutes/page. We set the
        paddle device globally AND pass device to the pipeline, and log the GPU
        state so a silent CPU fallback is visible.
        """
        import paddle
        from paddleocr import PaddleOCRVL

        print(
            f"PADDLE cuda_compiled={paddle.is_compiled_with_cuda()} "
            f"device={paddle.device.get_device()} "
            f"gpu_count={paddle.device.cuda.device_count()}"
        )
        try:
            paddle.set_device("gpu")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ paddle.set_device(gpu) failed: {e}")
        print(f"PADDLE device after set = {paddle.device.get_device()}")

        try:
            self.pipeline = PaddleOCRVL(device="gpu")
        except TypeError:
            # Older signature without a device kwarg — rely on set_device above.
            self.pipeline = PaddleOCRVL()

        # Warmup predict: pays the paddle JIT at container startup so the FIRST
        # real /parse is fast (~3s) instead of timing out the caller (~3 min cold
        # JIT). This blocks @modal.enter ~60-90s, during which /health can't be
        # served — but with max_inputs=16 the handful of queued probes clear
        # instantly once startup finishes (the earlier /health pile-up was the
        # max_inputs=1 bug, not this warmup).
        try:
            import numpy as np
            from PIL import Image, ImageDraw
            # SMALL warmup image — the paddle JIT generalizes across input shapes,
            # so this ~7s warmup makes the first real (full-page) /parse ~3s. A
            # big warmup image instead makes THIS predict JIT-block for minutes
            # (validated: 400x200 warmup → first 1000x1400 /parse = 3.2s).
            im = Image.new("RGB", (400, 200), "white")
            ImageDraw.Draw(im).text((20, 80), "WARMUP 123", fill="black")
            import time as _t
            t0 = _t.time()
            _ = list(self.pipeline.predict(np.array(im)[:, :, ::-1]))
            print(f"✅ warmup predict ok in {_t.time()-t0:.1f}s")
        except Exception as e:  # noqa: BLE001 - warmup is best-effort
            print(f"warmup predict skipped: {e}")
        print("✅ PaddleOCR-VL pipeline loaded (JIT warm)")

    def _parse_image(self, image_bytes: bytes):
        """Run the pipeline on one image; return (regions, width, height)."""
        import tempfile

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        width, height = img.size

        # Pass a real file path — the most robust input form for the pipeline
        # (avoids RGB/BGR ndarray ambiguity that would corrupt OCR).
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
            img.save(tf.name, format="PNG")
            outputs = list(self.pipeline.predict(tf.name))

        regions = []
        for res in outputs:
            data = res.json if hasattr(res, "json") else res
            # Unwrap the {"res": {...}} envelope the pipeline emits.
            payload = data.get("res", data) if isinstance(data, dict) else {}
            for blk in payload.get("parsing_res_list", []) or []:
                bbox = blk.get("block_bbox") or blk.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                # Robust to keys that EXIST with a None value (dict.get returns
                # None, not the default, in that case) — that was a 500-bug.
                order = blk.get("block_order")
                if order is None:
                    order = blk.get("block_id")
                if order is None:
                    order = len(regions)
                regions.append({
                    "bbox": [float(v) for v in bbox],   # pixel x0 y0 x1 y1
                    "label": str(blk.get("block_label") or blk.get("label") or "text"),
                    "content": blk.get("block_content") or blk.get("content") or "",
                    "order": int(order),
                })
        regions.sort(key=lambda r: r["order"])
        return regions, width, height

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel

        api_key = os.environ["PADDLEOCR_API_KEY"]
        api = FastAPI(title="paddleocr-vl")

        class ParseBody(BaseModel):
            image_b64: str
            mode: str = "page"  # "page" (structural pass) | "block" (per-crop OCR)

        def _auth(authorization):
            if authorization != f"Bearer {api_key}":
                raise HTTPException(status_code=401, detail="invalid bearer")

        @api.get("/health")
        def health():
            return {"status": "ok", "model": "paddleocr-vl"}

        @api.post("/parse")
        def parse(body: ParseBody, authorization: str = Header(None)):
            _auth(authorization)
            try:
                image_bytes = base64.b64decode(body.image_b64)
            except Exception:
                raise HTTPException(status_code=400, detail="image_b64 not valid base64")
            regions, width, height = self._parse_image(image_bytes)
            if body.mode == "block":
                # Per-crop OCR: concatenate recognized text in reading order.
                text = "\n".join(r["content"] for r in regions if r.get("content")).strip()
                return {"text": text, "regions": regions, "width": width, "height": height}
            return {"regions": regions, "width": width, "height": height}

        return api
