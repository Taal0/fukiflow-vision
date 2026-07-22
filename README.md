# FukiFlow Vision

Stateless FastAPI service for manga **bubble detect** (YOLO-seg) + **manga-ocr** + **inset-fill clean**. Bridges translators and editors/letterers — not a full translation pipeline.

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
| POST | `/v1/clean` | page + `bubbles_json` → cleaned PNG + `editor_regions` + `clean` (`ok`/`partial`) |
| POST | `/v1/pipeline` | bubbles → ocr → inset-fill clean in one call |

## Smoke test

```bash
./scripts/smoke_test.sh path/to/page.png
```

## Batch (folder / single page)

Requires the API running. Does **not** modify the input. No args → default test manga folder; folder output is a versioned sibling `*-translated v<N>`.

Per page: original copy (`*-orig.*`), cleaned PNG, detect overlay (`*-detect.png`), `*-bubbles.json`. Run-level: `ocr.json`.

```bash
python scripts/batch_clean.py
python scripts/batch_clean.py "/path/to/test manga"
python scripts/batch_clean.py path/to/page.png
HOST=http://127.0.0.1:8001 python scripts/batch_clean.py ...
```

## Layout

- `vision/` — bubble detect + manga-ocr + clean (app-agnostic; orchestration stays in routers)
- `app/` — FastAPI wrapper (`engine.py` loads bubble + OCR + clean in lifespan)
- `models/bubble/` — YOLO-seg weights (`BUBBLE_WEIGHT`)

## Env

| Variable | Default | Notes |
|----------|---------|-------|
| `BUBBLE_WEIGHT` | `models/bubble/yolov8n-seg-bubble-v1.pt` | YOLO-seg checkpoint |
| `BUBBLE_CONF` | `0.5` | same as bubble-yolo `eval.sh` |
| `BUBBLE_IMGSZ` | `640` | |
| `OCR_MODEL` | `models/ocr/manga-ocr-base` | local dir if present, else HF hub id |
| `OCR_CROP_PAD` | `4` | px pad around bubble bbox |
| `CLEAN_INSET` | `10` | erode polygon before fill (keep outline) |
| `CLEAN_RING` | `6` | inner-edge band width for uniform sample |
| `CLEAN_UNIFORM_TOL` | `18` | per-channel abs distance to median |
| `CLEAN_UNIFORM_FRAC` | `0.85` | fraction of ring pixels within tol |
| `CLEAN_MIN_RING_PX` | `30` | skip if inner ring too small |
| `DEVICE` | auto | `cpu` / `mps` / `cuda` |

## Out of scope (this revision)

SFX class / filter, render, Docker/cloud GPU, auth.
