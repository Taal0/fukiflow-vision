#!/usr/bin/env python3
"""Compose a PageDoc onto its source image (UI-less layer enable/disable test).

Usage:
  python scripts/compose_doc.py <stem>-doc.json
  python scripts/compose_doc.py <stem>-doc.json --off cln_3,txt_7
  python scripts/compose_doc.py <stem>-doc.json --only-kind clean
  python scripts/compose_doc.py <stem>-doc.json --export-layers
  python scripts/compose_doc.py <stem>-doc.json -o preview.png

Resolves source from PageDoc.source.file (relative to the doc JSON directory
when not absolute).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Allow `python scripts/compose_doc.py` from repo root or scripts/.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vision.document import PageDoc  # noqa: E402
from vision.layers.compose import compose, render_layer_rgba  # noqa: E402


def _load_rgb(path: Path) -> np.ndarray:
    data = path.read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"error: could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resolve_source(doc: PageDoc, doc_path: Path) -> Path:
    raw = (doc.source.file or "").strip()
    if not raw:
        raise SystemExit("error: PageDoc.source.file is empty")
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    cand = (doc_path.parent / raw).resolve()
    if cand.is_file():
        return cand
    raise SystemExit(f"error: source not found: {raw}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Compose PageDoc layers onto source")
    ap.add_argument("doc", type=Path, help="Path to <stem>-doc.json")
    ap.add_argument(
        "--off",
        default="",
        help="Comma-separated layer ids to disable (e.g. cln_3,txt_7)",
    )
    ap.add_argument(
        "--only-kind",
        choices=("clean", "text"),
        default=None,
        help="Compose only this kind of layer",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: <stem>-compose.png next to doc)",
    )
    ap.add_argument(
        "--export-layers",
        action="store_true",
        help="Also write <stem>-layers/<layer_id>.png RGBA patches",
    )
    ap.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Override font path for text layers",
    )
    args = ap.parse_args(argv[1:])

    doc_path = args.doc.expanduser().resolve()
    if not doc_path.is_file():
        print(f"error: doc not found: {doc_path}", file=sys.stderr)
        return 1

    doc = PageDoc.model_validate(json.loads(doc_path.read_text(encoding="utf-8")))
    src_path = _resolve_source(doc, doc_path)
    source = _load_rgb(src_path)

    off = {s.strip() for s in args.off.split(",") if s.strip()}
    font_path = str(args.font.resolve()) if args.font else None
    if font_path is None:
        # Prefer first text layer's font_file, else repo default.
        for L in doc.layers:
            if L.kind == "text" and L.layout and L.layout.font_file:
                font_path = L.layout.font_file
                break
        if font_path is None:
            default = _ROOT / "assets" / "fonts" / "BubbleSans-Regular.otf"
            if default.is_file():
                font_path = str(default)

    preview = compose(
        source,
        doc,
        font_path=font_path,
        off=off or None,
        only_kind=args.only_kind,
    )

    out = args.output
    if out is None:
        stem = doc_path.name.replace("-doc.json", "").removesuffix(".json")
        out = doc_path.parent / f"{stem}-compose.png"
    else:
        out = out.expanduser().resolve()

    bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        print("error: PNG encode failed", file=sys.stderr)
        return 1
    out.write_bytes(buf.tobytes())
    print(f"wrote {out}")

    if args.export_layers:
        layers_dir = doc_path.parent / f"{doc_path.stem.replace('-doc', '')}-layers"
        # doc stem is often "page-doc" → page-layers
        if doc_path.name.endswith("-doc.json"):
            layers_dir = doc_path.parent / f"{doc_path.name[: -len('-doc.json')]}-layers"
        layers_dir.mkdir(parents=True, exist_ok=True)
        h, w = source.shape[:2]
        n = 0
        for layer in doc.layers:
            rgba = render_layer_rgba(layer, (h, w), font_path=font_path)
            if not np.any(rgba[:, :, 3]):
                continue
            dest = layers_dir / f"{layer.id}.png"
            Image.fromarray(rgba, mode="RGBA").save(dest)
            n += 1
        print(f"wrote {n} layer PNG(s) → {layers_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
