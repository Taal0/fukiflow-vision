# Export contract (PSD / IDML)

PageDoc + layers are the source of truth. Writers bind to
`vision.layers.export.to_export_items(doc) → List[ExportItem]`, not to PageDoc
internals. This iteration ships the contract and fields only — no PSD/IDML writers.

## Coordinates

- Pixel space, origin top-left, **y down** (Photoshop-native).
- IDML converts to points: `pt = px * 72 / dpi`.
- `dpi` lives on `PageDoc.source.dpi` (`PAGE_DPI` env, default **350**).

## Color

- Working color is **RGB**.
- CMYK conversion is deferred to the export step; `source.color_mode` reserves the slot.

## Groups

- `layer.group = "bubble_N"` — Photoshop layer group and IDML group name.
- Clean + text for the same bubble share one group; they are **not** merged.

## Clean layers (`kind=raster` in ExportItem)

- Parametric mask: `inset_polygon` = polygon + inset (rebuild with `erode_mask`).
- Optional `rgba_file` / `raster_file` when a baked patch is preferred.
- Fill is solid `fill_color` RGB.

## Paint layers (`kind=raster` in ExportItem)

- Parametric strokes live on `Layer.strokes` (`fill` | `restore`, color, radius, points).
- Export flattens them to a single RGBA raster (no live stroke list in PSD/IDML).
- `z` sits between clean (100+) and text (200+), typically `150 + bubble_id`.
- `restore` stamps original source pixels; `fill` stamps a solid RGB sampled at stroke start.

## Text layers (`kind=text` — live text)

| Field | Meaning |
|--------|---------|
| `frame` | Axis-aligned text block from line boxes (IDML text frame) |
| `runs` / `lines[]` | **Hard breaks** with explicit baselines / leading |
| `font_pt` / `leading_pt` | `px * 72 / dpi` |
| `font_postscript_name` | InDesign PostScript name (not just family) |
| `font_file` | Typeset-time path (often absolute). **Compose resolves catalog basename first**; absolute path is fallback only. |
| `align` | `center` \| `left` |
| `hyphenated` | If **true**, export **must disable hyphenation** — soft hyphens are already embedded in `lines[].text` |

## Compose vs export

- Preview: `compose(source, enabled layers)` — disable a layer = “delete” without re-running the pipeline.
- Typeset does **not** depend on clean pixels (geometry-only); removing a clean layer does not invalidate text layout.
- PSD/IDML writers consume `ExportItem` only.
