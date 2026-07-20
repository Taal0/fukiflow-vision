# FukiFlow Vision

Stateless FastAPI service for manga **detect+OCR** and **inpaint**. Bridges translators and editors/letterers — not a full translation pipeline.

## Setup

```bash
cd fukiflow-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Models (not in git) — copy from the MIT checkout if missing:
# mkdir -p models
# rsync -a ../fukiflow/backend-source-repos/manga-image-translator/models/{detection,ocr,inpainting} ./models/

cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

During local bring-up you can also reuse the existing MIT repo venv (already has torch etc.) by putting `fukiflow-vision` on `PYTHONPATH` and running its `uvicorn`.

```bash
export PYTHONPATH=/path/to/fukiflow-vision
export MODELS_DIR=/path/to/fukiflow-vision/models
/path/to/manga-image-translator/venv/bin/uvicorn app.main:app --port 8001
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | `{status, device, models_loaded}` |
| POST | `/v1/detect-ocr` | multipart `file` → `{page_id, boxes[], mask}` |
| POST | `/v1/inpaint` | page + `mask` (or `boxes_json`) → cleaned PNG |

## Smoke test

```bash
./scripts/smoke_test.sh path/to/page.png
```

## Batch clean (folder / single page)

Requires the API running. Does **not** modify the input; writes a sibling `*-translated/` folder with an original copy (`*-orig.*`) and cleaned PNG per page.

```bash
python scripts/batch_clean.py "/path/to/test manga"
python scripts/batch_clean.py path/to/page.png
HOST=http://127.0.0.1:8001 python scripts/batch_clean.py ...
```

## Layout

- `vision_engine/` — vendored, pruned manga-image-translator (detection / OCR / inpainting / textline_merge / utils / config)
- `app/` — FastAPI wrapper (`engine.py` is the stateless core)
- `models/` — weights (`MODELS_DIR`)

## Out of scope (Faz A)

`/render`, Turkish typesetting, stateful `page_id` cache, Docker/cloud GPU, auth.
