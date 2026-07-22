# FukiFlow Vision

Stateless FastAPI service for manga **bubble detect** (YOLO-seg) + **manga-ocr**. Clean is a passive no-op stub until the inset-fill revision lands. Bridges translators and editors/letterers — not a full translation pipeline.

## Setup

```bash
cd fukiflow-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Models (not in git) — under MODELS_DIR / models/:
#   bubble/yolov8n-seg-bubble-v1.pt — YOLO-seg (API default)
#   ocr/manga-ocr-base/            — manga-ocr (OCR_MODEL; else HF hub id)
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | `{status, device, models:{bubble,ocr}}` |
| POST | `/v1/bubbles` | multipart `file` → `{page_id, width, height, bubbles[]}` |
| POST | `/v1/ocr` | page + `bubbles_json` → bubbles with `text_ja` |
| POST | `/v1/clean` | **no-op**: original PNG + `editor_regions=[]` + `clean: disabled` |
| POST | `/v1/pipeline` | bubbles → ocr → clean(no-op) in one call |

## Smoke test

```bash
./scripts/smoke_test.sh path/to/page.png
```

## Batch (folder / single page)

Requires the API running. Does **not** modify the input; writes a sibling `*-translated/` folder with an original copy (`*-orig.*`), cleaned PNG (identical while clean is disabled), and `*-bubbles.json`.

```bash
python scripts/batch_clean.py "/path/to/test manga"
python scripts/batch_clean.py path/to/page.png
HOST=http://127.0.0.1:8001 python scripts/batch_clean.py ...
```

## Layout

- `vision/` — bubble detect + manga-ocr (app-agnostic; orchestration stays in routers)
- `app/` — FastAPI wrapper (`engine.py` loads bubble + OCR in lifespan)
- `models/bubble/` — YOLO-seg weights (`BUBBLE_WEIGHT`)
- `app/solidfill.py`, `app/postprocess.py` — reference for the upcoming clean revision (not wired)

## Env

| Variable | Default | Notes |
|----------|---------|-------|
| `BUBBLE_WEIGHT` | `models/bubble/yolov8n-seg-bubble-v1.pt` | YOLO-seg checkpoint |
| `BUBBLE_CONF` | `0.5` | same as bubble-yolo `eval.sh` |
| `BUBBLE_IMGSZ` | `640` | |
| `OCR_MODEL` | `models/ocr/manga-ocr-base` | local dir if present, else HF hub id |
| `OCR_CROP_PAD` | `4` | px pad around bubble bbox |
| `DEVICE` | auto | `cpu` / `mps` / `cuda` |

## Out of scope (this revision)

Real clean (inset fill + uniform check + `editor_regions`), SFX, render, Docker/cloud GPU, auth.
