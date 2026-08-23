"""Modal serverless wrapper for full fukiflow-vision (YOLO + clean + typeset).

Scale-to-zero GPU worker for VPS `VISION_MODE=remote` + `VISION_FULL_URL`.
Does not change `app.main` — same FastAPI routes (`/v1/bubbles`, `/pipeline`, …).

Prereqs (Mac, from this repo):
  pip install modal
  modal setup
  # models/ must exist locally (gitignored) — see README models-v2 download
  modal deploy modal_app.py

Then on the 2GB VPS app unit:
  VISION_MODE=remote
  VISION_URL=http://127.0.0.1:8001
  VISION_FULL_URL=https://<workspace>--fukiflow-vision-fastapi-app.modal.run
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent

# nano YOLO fits T4; idle = $0 when scaledown hits (min_containers default 0).
GPU = "T4"
# Keep warm between chapter upload pages (~1s apart); scale to zero after.
SCALEDOWN_S = 90
# App pipeline timeout is 180s; leave headroom for cold start + large pages.
TIMEOUT_S = 600

_REQUIRED_WEIGHTS = (
    ROOT / "models" / "bubble" / "yolov8n-seg-bubble-v2.pt",
    ROOT / "models" / "panel" / "yolov8n-panel-v3.pt",
)


def _assert_local_weights() -> None:
    """Fail fast on Mac deploy if gitignored weights are missing."""
    missing = [p for p in _REQUIRED_WEIGHTS if not p.is_file()]
    if not missing:
        return
    # Worker image already baked weights under /root/models.
    if Path("/root/models/bubble/yolov8n-seg-bubble-v2.pt").is_file():
        return
    paths = "\n  ".join(str(p) for p in missing)
    raise FileNotFoundError(
        "Modal deploy needs local YOLO weights (gitignored).\n"
        f"Missing:\n  {paths}\n"
        "From repo root:\n"
        "  mkdir -p models/bubble models/panel\n"
        "  gh release download models-v2 -R Taal0/fukiflow-vision "
        "-D models/bubble -p 'yolov8n-seg-bubble-*.pt'\n"
        "  gh release download models-v2 -R Taal0/fukiflow-vision "
        "-D models/panel -p 'yolov8n-panel-*.pt'"
    )


_assert_local_weights()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        # Match requirements.txt; headless OpenCV avoids GUI libs on the worker.
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "python-multipart>=0.0.9",
        "pydantic>=2.5",
        "pillow>=10.0",
        "python-dotenv",
        "pyphen>=0.14",
        "ultralytics>=8.3.0",
        "torch",
        "torchvision",
        "numpy==1.26.4",
        "opencv-python-headless",
        "tqdm",
        "requests",
        "huggingface_hub",
        "safetensors",
        "protobuf>=3.20.2,<6.0.0",
    )
    .env(
        {
            "DEVICE": "cuda",
            "VISION_LIGHT": "0",
            "MODELS_DIR": "/root/models",
        }
    )
    .add_local_dir(str(ROOT / "assets"), remote_path="/root/assets", copy=True)
    .add_local_dir(
        str(ROOT / "models" / "bubble"),
        remote_path="/root/models/bubble",
        copy=True,
    )
    .add_local_dir(
        str(ROOT / "models" / "panel"),
        remote_path="/root/models/panel",
        copy=True,
    )
    .add_local_python_source("app", "vision")
)

app = modal.App("fukiflow-vision", image=image)


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_S,
    memory=8192,
    scaledown_window=SCALEDOWN_S,
)
@modal.asgi_app()
def fastapi_app():
    """HTTPS URL → same FastAPI as local `uvicorn app.main:app`."""
    from app.main import app as web_app

    return web_app
