"""Modal deployment: Idol (speridlabs) — one product photo in, a 3D Gaussian splat (.ply) out."""

import io
import os
import time

import modal

GPU = os.environ.get("IDOL_GPU", "L40S")
SCALEDOWN_WINDOW = int(os.environ.get("IDOL_SCALEDOWN_WINDOW", "120"))
MAX_CONTAINERS = int(os.environ.get("IDOL_MAX_CONTAINERS", "2"))
SDK_COMMIT = os.environ.get("IDOL_SDK_COMMIT", "8db700349a5570aac390e4780fa32e3eed6b587a")
TORCH_VERSION = os.environ.get("IDOL_TORCH_VERSION", "2.7.1")

MODELS = "/models"
IDOL_DIR = f"{MODELS}/idol"
DINOV3_DIR = f"{MODELS}/dinov3"
FLUX2_DIR = f"{MODELS}/flux2"
MAX_IMAGE_BYTES = 12_000_000


DINOV3_REPO = os.environ.get("IDOL_DINOV3_REPO", "cocktailpeanut/dinov3-vith16plus-pretrain-lvd1689m")
DINOV3_REVISION = os.environ.get("IDOL_DINOV3_REVISION", "2d7afabdb47f9219bc0de332edd3e78ed00c79b8")
DINOV3_SHA256 = "3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"


def _download_models():
    """Bake every weight into the image. Only the Apache-2.0 `vae/` folder of FLUX.2-dev is fetched, never the model.

    DINOv3 comes from a byte-identical mirror of Meta's gated repo (same size, same config and LICENSE),
    pinned and checksum-verified; the Meta DINOv3 License applies to it unchanged."""
    import hashlib

    from huggingface_hub import snapshot_download

    token = os.environ["HF_TOKEN"]
    snapshot_download("speridlabs/idol", local_dir=IDOL_DIR, token=token)
    snapshot_download(DINOV3_REPO, revision=DINOV3_REVISION, local_dir=DINOV3_DIR, token=token)
    digest = hashlib.sha256()
    with open(f"{DINOV3_DIR}/model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    if digest.hexdigest() != DINOV3_SHA256:
        raise RuntimeError(f"DINOv3 weights checksum mismatch: {digest.hexdigest()}")
    snapshot_download("black-forest-labs/FLUX.2-dev", local_dir=FLUX2_DIR, allow_patterns=["vae/*"], token=token)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(f"torch=={TORCH_VERSION}", "torchvision", "hf_transfer", "fastapi[standard]")
    .pip_install(f"git+https://github.com/speridlabs/idol-sdk@{SDK_COMMIT}")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download_models, secrets=[modal.Secret.from_name("huggingface-token")])
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "IDOL_DEVICE": "cuda"})
)

app = modal.App("idol")


@app.cls(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    timeout=600,
    secrets=[modal.Secret.from_name("paddleocr-api-key")],
)
@modal.concurrent(max_inputs=1)
class IdolService:
    @modal.enter()
    def load(self):
        from idol import IdolPipeline

        self.pipeline = IdolPipeline.from_pretrained(
            IDOL_DIR, image_encoders={"dinov3": DINOV3_DIR, "flux2_vae": FLUX2_DIR},
        )

    def _generate(self, image_bytes: bytes, mode: str, num_gaussians: int, seed: int) -> dict:
        from PIL import Image

        started = time.time()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        out = self.pipeline(img, mode=mode, seed=seed, num_gaussians=num_gaussians)
        ply_path = "/tmp/out.ply"
        out.gaussian.save_ply(ply_path)
        with open(ply_path, "rb") as f:
            ply = f.read()
        return {"ply": ply, "seconds": round(time.time() - started, 2)}

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
        from fastapi.responses import Response

        api_key = os.environ.get("IDOL_API_KEY") or os.environ["PADDLEOCR_API_KEY"]
        api = FastAPI(title="idol")

        @api.get("/health")
        def health():
            return {"status": "ok", "model": "idol", "sdk": SDK_COMMIT}

        @api.post("/generate")
        async def generate(
            image: UploadFile = File(...),
            mode: str = Form("real"),
            num_gaussians: int = Form(262144),
            seed: int = Form(42),
            authorization: str = Header(None),
        ):
            if authorization != f"Bearer {api_key}":
                raise HTTPException(status_code=401, detail="invalid bearer")
            if mode not in ("real", "synthetic"):
                raise HTTPException(status_code=400, detail="mode must be real or synthetic")
            if not 32_768 <= num_gaussians <= 1_048_576:
                raise HTTPException(status_code=400, detail="num_gaussians must be 32768..1048576")
            data = await image.read()
            if not data or len(data) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail="image missing or larger than 12 MB")
            try:
                result = self._generate(data, mode, num_gaussians, seed)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"generation failed: {e!r}")
            return Response(
                content=result["ply"],
                media_type="application/octet-stream",
                headers={"X-Idol-Seconds": str(result["seconds"])},
            )

        @api.post("/prepared")
        async def prepared(image: UploadFile = File(...), mode: str = Form("real"), authorization: str = Header(None)):
            if authorization != f"Bearer {api_key}":
                raise HTTPException(status_code=401, detail="invalid bearer")
            from PIL import Image

            img = Image.open(io.BytesIO(await image.read())).convert("RGB")
            buf = io.BytesIO()
            self.pipeline.preprocess(img, mode=mode).save(buf, format="PNG")
            return Response(content=buf.getvalue(), media_type="image/png")

        return api
