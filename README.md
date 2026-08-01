# FukiFlow Vision

Stateless FastAPI service for manga **bubble detect** (YOLO-seg) + **ink filter** + **panel detect** + **RSP reading order** + **inset-fill clean** + **typeset** (layout JSON).
Bridges translators and editors/letterers — not a full translation pipeline.

> OCR (manga-ocr) was parked 2026-07-24 — code snapshot:
> `fukiflow/backend-source-repos/fukiflow-vision-ocr/`.

## Setup

```bash
cd fukiflow-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Models (not in git) — download release models-v2:
mkdir -p models/bubble models/panel
gh release download models-v2 -R Taal0/fukiflow-vision -D models/bubble -p 'yolov8n-seg-bubble-*.pt'
gh release download models-v2 -R Taal0/fukiflow-vision -D models/panel -p 'yolov8n-panel-*.pt'
#   bubble/yolov8n-seg-bubble-v2.pt — YOLO-seg (API default)
#   bubble/yolov8n-seg-bubble-v1.pt — rollback
#   panel/yolov8n-panel-v3.pt — panel detect
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | `{status, device, models:{bubble,panel,order,clean,typeset}}` |
| POST | `/v1/bubbles` | multipart `file` → bubbles + panels + reading_order (ink DROP applied) |
| POST | `/v1/order` | detect → ink_filter → RSP order (+ optional overlay) |
| POST | `/v1/clean` | page + `bubbles_json` → cleaned PNG + `editor_regions` + `clean` (`ok`/`partial`) |
| POST | `/v1/pipeline` | bubbles → ink_filter → order → inset-fill clean in one call |
| POST | `/v1/typeset` | cleaned page + `bubbles_json` + `texts_json` → layout JSON (+ optional debug PNG) |

`/v1/bubbles` and `/v1/pipeline` include `panels[]`, `order_confidence`, and `needs_review` by default.

## Smoke test

```bash
./scripts/smoke_test.sh path/to/page.png
```

## Batch (folder / single page)

Requires the API running. Does **not** modify the input. No args → default test manga folder; folder output is a versioned sibling `*-translated v<N>`.

Pipeline per page: detect → order → clean → **typeset**. Default TR texts:
`/Users/talo/Desktop/test and result/extracted/manga_texts_tr_reading_order.json`
(`typeset_by_page`; bubble_id mismatch → reading_order remap).

Per page: `*-orig.*`, cleaned PNG, `*-detect.png`, `*-bubbles.json`, `*-typeset.json`, `*-typeset.png`.

```bash
python scripts/batch_clean.py
python scripts/batch_clean.py "/path/to/test manga"
python scripts/batch_clean.py path/to/page.png
TEXTS_JSON=/path/to/manga_texts_tr_reading_order.json python scripts/batch_clean.py ...
SKIP_TYPESET=1 python scripts/batch_clean.py ...   # clean-only
HOST=http://127.0.0.1:8001 python scripts/batch_clean.py ...
```

## Reading-order eval (optional)

API-less local run: overlay PNGs + per-page orphan/fallback/confidence summary.

```bash
python scripts/eval_reading_order.py "/path/to/test manga"
# → fixtures/out/reading_order/*-order.png
python scripts/eval_reading_order.py "/path/to/test manga" --gt gt_order.json
```

## Ink calibration (optional)

```bash
python scripts/measure_ink.py "/path/to/test manga"
# → fixtures/out/ink_measure.csv
```

## Typeset measurement (optional)

```bash
python scripts/measure_typeset.py   # default: ../fukiflow-train/datasets/clean
# → fixtures/out/typeset_measure.csv  (multi-lobe / hyphen / escalation rates)
```

Placeholder font: `assets/fonts/NotoSans-Regular.ttf` (OFL). Override with `TYPESET_FONT`.

## Layout

- `vision/` — bubble detect + ink_filter + panel detect + order + clean + typeset (app-agnostic)
- `vision/panel/` — YOLO panel detector (bbox-only)
- `vision/order/` — RSP panel/bubble ordering, IoB assignment, ambiguity score
- `vision/typeset/` — width profile, wrap/hyphen, font-fit ladder, layout JSON
- `vision/masks.py` — shared polygon / ring / ink helpers
- `app/` — FastAPI wrapper (`engine.py` loads bubble + panel + order + clean + typeset)
- `models/bubble/` — YOLO-seg weights (`BUBBLE_WEIGHT`)
- `models/panel/` — YOLO panel weights (`PANEL_WEIGHT`)
- `assets/fonts/` — OFL placeholder font for typeset

## Env

| Variable | Default | Notes |
|----------|---------|-------|
| `BUBBLE_WEIGHT` | `models/bubble/yolov8n-seg-bubble-v2.pt` | YOLO-seg checkpoint (v1 for rollback) |
| `BUBBLE_CONF` | `0.65` | hard detect threshold |
| `BUBBLE_CONF_SOFT_HI` | `0.80` | soft band upper for ink_filter |
| `BUBBLE_IMGSZ` | `640` | |
| `PANEL_WEIGHT` | `models/panel/yolov8n-panel-v3.pt` | YOLO panel checkpoint |
| `PANEL_CONF` | `0.40` | panel detect threshold |
| `PANEL_IMGSZ` | `640` | |
| `PANEL_NMS_IOU` | `0.60` | post-detect panel dedupe |
| `ORDER_TOL_FRAC` | `0.02` | RSP cut tolerance = `min(W,H) * frac` |
| `ORDER_IOB_CONTAINED` | `0.70` | IoB → contained assignment |
| `ORDER_IOB_PARTIAL` | `0.15` | below → orphan (nearest panel) |
| `ORDER_AMBIG_DELTA` | `0.05` | top-2 IoB gap → ambiguous |
| `ORDER_CONF_MIN` | `0.50` | below → `needs_review` |
| `INK_TOL` | `40` | ring-median deviation for ink pixels |
| `INK_MIN_PX` | `50` | total dark px ≥ this → text_present |
| `INK_MIN_CC_AREA` | `12` | any dark CC area ≥ this → text_present |
| `CLEAN_INSET` | `10` | erode polygon before fill (keep outline) |
| `CLEAN_RING` | `6` | inner-edge band width for uniform sample |
| `CLEAN_UNIFORM_TOL` | `18` | per-channel abs distance to median |
| `CLEAN_UNIFORM_FRAC` | `0.85` | fraction of ring pixels within tol |
| `CLEAN_MIN_RING_PX` | `30` | skip if inner ring too small |
| `TYPESET_FONT` | `assets/fonts/NotoSans-Regular.ttf` | OFL placeholder; no metrics baked into code |
| `TYPESET_MARGIN_RATIO` | `0.008` | typographic margin = page_h × ratio (on top of clean inset) |
| `TYPESET_FONT_MIN_RATIO` | `0.008` | ladder floor (× page_h); shrink-to-fit for narrow bubbles |
| `TYPESET_FONT_MAX_RATIO` | `0.042` | ladder ceiling (× page_h) |
| `TYPESET_FONT_STEPS` | `16` | ladder steps |
| `TYPESET_LEADING` | `1.15` | × (ascent+descent) |
| `TYPESET_WIDTH_SAFETY` | `0.98` | shrink usable line width |
| `TYPESET_HYPHENATE` | `1` | pyphen hyphen fallback |
| `TYPESET_HYPHEN_LANG` | `tr_TR` | unused when dic file set |
| `TYPESET_HYPHEN_DIC` | `assets/hyphen/hyph_tr_TR.dic` | LPPL Turkish patterns (pyphen has no `tr`) |
| `DEVICE` | auto | `cpu` / `mps` / `cuda` |

## Out of scope (this revision)

OCR, AI rerank (only `needs_review` flag), panel polygon/seg, double-page spread, SFX class / filter, Photoshop/IDML export, auto multi-lobe split, Docker/cloud GPU, auth.
