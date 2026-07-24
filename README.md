# FukiFlow Vision

Stateless FastAPI service for manga **bubble detect** (YOLO-seg) + **ink filter** + **inset-fill clean**.
Bridges translators and editors/letterers — not a full translation pipeline.

> OCR (manga-ocr) was parked 2026-07-24 — code snapshot:
> `fukiflow/backend-source-repos/fukiflow-vision-ocr/`.

## Setup

```bash
cd fukiflow-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Models (not in git) — under MODELS_DIR / models/:
#   bubble/yolov8n-seg-bubble-v2.pt — YOLO-seg (API default; from bubble-500)
#   bubble/yolov8n-seg-bubble-v1.pt — rollback
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | `{status, device, models:{bubble,clean}}` |
| POST | `/v1/bubbles` | multipart `file` → `{page_id, width, height, bubbles[]}` (ink DROP applied) |
| POST | `/v1/clean` | page + `bubbles_json` → cleaned PNG + `editor_regions` + `clean` (`ok`/`partial`) |
| POST | `/v1/pipeline` | bubbles → ink_filter → inset-fill clean in one call |

## Smoke test

```bash
./scripts/smoke_test.sh path/to/page.png
```

## Batch (folder / single page)

Requires the API running. Does **not** modify the input. No args → default test manga folder; folder output is a versioned sibling `*-translated v<N>`.

Per page: original copy (`*-orig.*`), cleaned PNG, detect overlay (`*-detect.png`), `*-bubbles.json`.

```bash
python scripts/batch_clean.py
python scripts/batch_clean.py "/path/to/test manga"
python scripts/batch_clean.py path/to/page.png
HOST=http://127.0.0.1:8001 python scripts/batch_clean.py ...
```

## Ink calibration (optional)

```bash
python scripts/measure_ink.py "/path/to/test manga"
# → fixtures/out/ink_measure.csv
```

## Layout

- `vision/` — bubble detect + ink_filter + clean (app-agnostic; orchestration stays in routers)
- `vision/masks.py` — shared polygon / ring / ink helpers
- `app/` — FastAPI wrapper (`engine.py` loads bubble + clean in lifespan)
- `models/bubble/` — YOLO-seg weights (`BUBBLE_WEIGHT`)

## Env

| Variable | Default | Notes |
|----------|---------|-------|
| `BUBBLE_WEIGHT` | `models/bubble/yolov8n-seg-bubble-v2.pt` | YOLO-seg checkpoint (v1 for rollback) |
| `BUBBLE_CONF` | `0.65` | hard detect threshold |
| `BUBBLE_CONF_SOFT_HI` | `0.80` | soft band upper for ink_filter |
| `BUBBLE_IMGSZ` | `640` | |
| `INK_TOL` | `40` | ring-median deviation for ink pixels |
| `INK_MIN_PX` | `50` | total dark px ≥ this → text_present |
| `INK_MIN_CC_AREA` | `12` | any dark CC area ≥ this → text_present |
| `CLEAN_INSET` | `10` | erode polygon before fill (keep outline) |
| `CLEAN_RING` | `6` | inner-edge band width for uniform sample |
| `CLEAN_UNIFORM_TOL` | `18` | per-channel abs distance to median |
| `CLEAN_UNIFORM_FRAC` | `0.85` | fraction of ring pixels within tol |
| `CLEAN_MIN_RING_PX` | `30` | skip if inner ring too small |
| `DEVICE` | auto | `cpu` / `mps` / `cuda` |

## Out of scope (this revision)

OCR, SFX class / filter, render, Docker/cloud GPU, auth.
