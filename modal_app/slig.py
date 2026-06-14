"""
Modal deployment: SLIG (SigLIP2) as the pipeline's visual-embedding host.

SLIG is the platform's visual encoder — it produces the 768D ``visual`` vector in
the 7-embedding fusion search (``vecs.image_slig_embeddings``) and also serves
zero-shot classification + image⇄text similarity. The model behind it is the
custom HF repo ``basiliskan/slig``, which is a **verbatim duplication of
``google/siglip2-base-patch16-512``** — a stock SigLIP2 base model with a
**native 768D** output (NO SO400M, NO 1152→768 projection head, despite older
docstrings; ``config.json`` is ``model_type: "siglip"`` with no ``auto_map``).

It moved off **HuggingFace Inference Endpoints** (paid always-allocating GPU +
flaky resume/scale/capacity lifecycle) onto **Modal**, mirroring the PaddleOCR-VL
cutover. Embeddings stay **bit-for-bit identical** to the HF endpoint because we
load the model the exact same way the HF custom ``handler.py`` did — stock
``transformers`` ``AutoModel.from_pretrained`` + ``AutoProcessor`` (the processor
reads the repo's own ``preprocessor_config.json``, so image preprocessing is
identical), then ``get_image_features``/``get_text_features`` → L2-normalize.

This exposes a small custom contract the MIVAA SLIG client speaks — the SAME
``{inputs, parameters}`` request body and the SAME response shapes the HF endpoint
used (see ``docs/api/slig-inference.md``), so the client's payload builders don't
change:

  GET  /health           → 200 once the model is loaded + warmed (unauth probe)
  POST /infer  (bearer)  → {"inputs": <str|[str]|{image(s),text(s)}>,
                            "parameters": {"mode": "...", "candidate_labels": [...]}}
       mode ∈ {zero_shot, image_embedding, text_embedding, similarity, auto}
       → image_embedding / text_embedding: [{"embedding": [768 floats]}, ...]
       → zero_shot:  [{"label","score"}, ...]   (single) | [[...], ...] (batch)
       → similarity: {"similarity_scores": [[...]], "image_count", "text_count"}

Lifecycle: scale-to-zero (``min_containers=0`` + ``scaledown_window``) so it costs
$0 idle; the first request after idle cold-starts a GPU container; MIVAA
health-probes ``/health`` as its warmup. SigLIP2-base is small (~400M, 1.5 GB
weights baked into the image), so a cold start is just model→GPU load (~5-15s),
no network download.

NOTE on the warm posture: unlike PaddleOCR-VL (batch-only PDF jobs), SLIG also
serves REALTIME search queries. With ``min_containers=0`` the first query after
idle eats the cold start — the MIVAA search path must warm-probe before the embed
call and degrade gracefully (drop the visual vector, keep the other fusion
vectors) on timeout. Set ``SLIG_MIN_CONTAINERS=1`` to keep one replica always warm
if query latency matters more than $0 idle.

Deploy:
    modal deploy modal_app/slig.py
The printed URL is SLIG_MODAL_URL. The bearer is SHARED with the PaddleOCR app —
this app reuses the existing ``paddleocr-api-key`` Modal secret, so NO new secret
is needed to deploy (the bearer can be shared across Modal apps; only the URL
differs per app). On the MIVAA side, set ``SLIG_MODAL_API_KEY`` to the same value
as ``PADDLEOCR_MODAL_API_KEY``. To give SLIG its own dedicated key later, create a
``slig-api-key`` secret exposing ``SLIG_API_KEY``, add it to ``secrets=[...]``
below, and it wins via the getenv precedence in ``web()``.
"""

import base64
import io
import os

import modal

# --------------------------------------------------------------------------- #
# Tunables (read at `modal deploy` time)
# --------------------------------------------------------------------------- #
GPU = os.environ.get("SLIG_GPU", "A10G")                     # A10G 24GB — matches the proven PaddleOCR app in this workspace; T4/L4 also fit
SCALEDOWN_WINDOW = int(os.environ.get("SLIG_SCALEDOWN_WINDOW", "120"))   # idle → $0
MIN_CONTAINERS = int(os.environ.get("SLIG_MIN_CONTAINERS", "0"))         # 0 = scale-to-zero; 1 = always warm
MAX_CONTAINERS = int(os.environ.get("SLIG_MAX_CONTAINERS", "4"))         # autoscale ceiling for ingest bursts
# Keep > 1 so /health probes are answered instantly and never queue behind
# in-flight /infer work (max_inputs=1 would cause a /health pile-up).
MAX_CONCURRENT = int(os.environ.get("SLIG_MAX_CONCURRENT", "16"))
MODEL_REPO = os.environ.get("SLIG_MODEL_REPO", "basiliskan/slig")
TORCH_VERSION = os.environ.get("SLIG_TORCH_VERSION", "2.5.1")
# siglip2 support landed in transformers 4.49.0 (the repo was saved with
# 4.49.0.dev0). Pin the family for numeric reproducibility of the forward pass.
TRANSFORMERS_VERSION = os.environ.get("SLIG_TRANSFORMERS_VERSION", "4.49.0")


def _download_model():
    """Bake the model into the image layer so cold starts never hit the network."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO)


# Build: slim Debian + the torch CUDA wheel (bundles CUDA runtime; Modal supplies
# the GPU driver) + transformers/sentencepiece, then snapshot the 1.5 GB weights
# into the image so every cold start is load-only.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
        "accelerate>=0.26.0",
        "safetensors>=0.4.0",
        "sentencepiece>=0.1.99",          # SigLIP2 uses a 256k-vocab sentencepiece tokenizer
        "pillow>=10.0.0",
        # MUST be <1.0: transformers 4.49 imports symbols that huggingface_hub
        # 1.0 removed. Left unpinned, pip grabs the latest (1.19+) and the
        # container crash-loops on `from transformers import ...` at startup.
        "huggingface_hub>=0.26.0,<1.0",
        "hf_transfer",                    # fast 1.5 GB pull at build time
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(_download_model)
    # AFTER the build-time download: pin the runtime fully offline. The weights
    # are baked into the image above, so at request time `from_pretrained` reads
    # purely from the local cache and NEVER contacts huggingface.co — both a
    # reliability win (no runtime dependency on HF being up) and what makes this
    # truly "no HuggingFace at runtime". Must come after run_function or the
    # build-time snapshot_download would itself be blocked.
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
)

app = modal.App("slig")


@app.cls(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=600,
    # Reuse the PaddleOCR app's bearer secret — one shared key across Modal apps,
    # so deploying SLIG needs NO new secret. (Add a dedicated `slig-api-key`
    # secret here later if you want independent rotation; `SLIG_API_KEY` wins.)
    secrets=[modal.Secret.from_name("paddleocr-api-key")],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT)
class SligService:
    @modal.enter()
    def load(self):
        """Load SigLIP2-base once per container, force GPU, warm with a real pass.

        We load EXACTLY as the HF ``handler.py`` did
        (``AutoModel.from_pretrained(..., trust_remote_code=True)`` — a harmless
        no-op here since the repo carries no custom modeling code) so the math is
        identical. ``trust_remote_code`` is kept only for byte-for-byte fidelity
        with the original load path.
        """
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            f"SLIG load: cuda_available={torch.cuda.is_available()} "
            f"device={self.device} repo={MODEL_REPO}",
            flush=True,
        )
        self.model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
        self.model.eval()

        # Warmup as a health gate: one real image embed + one text embed pays the
        # CUDA/JIT cost here so the first /infer is hot, and confirms the model
        # actually serves. SigLIP2 has no nondeterministic cold-start wedge (the
        # PaddleOCR-VL problem), so a plain try/raise is enough — raising recycles
        # a genuinely broken container.
        try:
            from PIL import Image

            t_img = Image.new("RGB", (64, 64), "white")
            _ = self._get_image_embeddings([t_img])
            _ = self._get_text_embeddings(["warmup"])
            print("✅ SLIG warmup ok — model serving", flush=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"SLIG warmup failed on this container — recycling: {e!r}")

    # ----------------------------------------------------------------------- #
    # Model primitives — ported verbatim from the HF handler for exact parity
    # ----------------------------------------------------------------------- #
    def _load_image(self, image_data):
        from PIL import Image
        import requests

        if isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                response = requests.get(image_data, timeout=10)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert("RGB")
            if "," in image_data:
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        raise ValueError(f"Unsupported image format: {type(image_data)}")

    def _get_image_embeddings(self, images):
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

    def _get_text_embeddings(self, texts):
        import torch

        inputs = self.processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

    # ----------------------------------------------------------------------- #
    # The 4 modes + auto-detect — ported verbatim from the HF handler
    # ----------------------------------------------------------------------- #
    def _run(self, data):
        inputs = data.get("inputs", data)
        parameters = data.get("parameters", {}) or {}
        mode = parameters.get("mode", "auto")

        if mode == "auto":
            if isinstance(inputs, dict) and ("image" in inputs or "images" in inputs):
                mode = "similarity"
            elif "candidate_labels" in parameters:
                mode = "zero_shot"
            elif isinstance(inputs, str) and not inputs.startswith(("http", "data:")) and len(inputs) < 500:
                mode = "text_embedding"
            elif isinstance(inputs, list) and all(
                isinstance(i, str) and not i.startswith(("http", "data:")) and len(i) < 500 for i in inputs
            ):
                mode = "text_embedding"
            else:
                mode = "image_embedding"

        if mode == "zero_shot":
            return self._zero_shot(inputs, parameters)
        if mode == "image_embedding":
            return self._image_embedding(inputs)
        if mode == "text_embedding":
            return self._text_embedding(inputs)
        if mode == "similarity":
            return self._similarity(inputs)
        raise ValueError(f"Unknown mode: {mode}")

    def _zero_shot(self, inputs, parameters):
        import torch

        candidate_labels = parameters.get("candidate_labels", ["photo", "illustration", "diagram"])
        if isinstance(candidate_labels, str):
            candidate_labels = [l.strip() for l in candidate_labels.split(",")]

        images = [self._load_image(inputs)] if not isinstance(inputs, list) else [self._load_image(i) for i in inputs]
        image_embeds = self._get_image_embeddings(images)
        text_embeds = self._get_text_embeddings(candidate_labels)

        logits = image_embeds @ text_embeds.T
        probs = torch.softmax(logits, dim=-1)

        results = []
        for prob in probs:
            scores = prob.cpu().tolist()
            result = [
                {"label": l, "score": s}
                for l, s in sorted(zip(candidate_labels, scores), key=lambda x: -x[1])
            ]
            results.append(result)
        return results[0] if len(results) == 1 else results

    def _image_embedding(self, inputs):
        images = [self._load_image(inputs)] if not isinstance(inputs, list) else [self._load_image(i) for i in inputs]
        embeddings = self._get_image_embeddings(images)
        return [{"embedding": emb.cpu().tolist()} for emb in embeddings]

    def _text_embedding(self, inputs):
        texts = [inputs] if isinstance(inputs, str) else inputs
        embeddings = self._get_text_embeddings(texts)
        return [{"embedding": emb.cpu().tolist()} for emb in embeddings]

    def _similarity(self, inputs):
        image_input = inputs.get("image") or inputs.get("images")
        text_input = inputs.get("text") or inputs.get("texts")

        images = (
            [self._load_image(image_input)]
            if not isinstance(image_input, list)
            else [self._load_image(i) for i in image_input]
        )
        texts = [text_input] if isinstance(text_input, str) else text_input

        image_embeds = self._get_image_embeddings(images)
        text_embeds = self._get_text_embeddings(texts)

        similarity = (image_embeds @ text_embeds.T).cpu().tolist()
        return {"similarity_scores": similarity, "image_count": len(images), "text_count": len(texts)}

    # ----------------------------------------------------------------------- #
    # HTTP surface — same {inputs, parameters} contract as the HF endpoint
    # ----------------------------------------------------------------------- #
    @modal.asgi_app()
    def web(self):
        from typing import Any, Dict

        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel

        # Prefer a dedicated SLIG_API_KEY if a `slig-api-key` secret is attached;
        # otherwise fall back to the shared PaddleOCR bearer (the secret wired in
        # `secrets=[...]` above). Either way the value is `SLIG_MODAL_API_KEY` on
        # the MIVAA side.
        api_key = os.environ.get("SLIG_API_KEY") or os.environ["PADDLEOCR_API_KEY"]
        api = FastAPI(title="slig")

        class InferBody(BaseModel):
            inputs: Any
            parameters: Dict[str, Any] = {}

        def _auth(authorization):
            if authorization != f"Bearer {api_key}":
                raise HTTPException(status_code=401, detail="invalid bearer")

        @api.get("/health")
        def health():
            return {"status": "ok", "model": "slig"}

        @api.post("/infer")
        def infer(body: InferBody, authorization: str = Header(None)):
            _auth(authorization)
            try:
                return self._run({"inputs": body.inputs, "parameters": body.parameters})
            except ValueError as e:
                # Bad mode / unsupported input shape — client error, not a 500.
                raise HTTPException(status_code=400, detail=str(e))

        return api
