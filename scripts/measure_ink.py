#!/usr/bin/env python3
"""Measure per-bubble ink structure on a test manga folder (read-only).

Runs YOLO detect locally (no API), writes a CSV for threshold calibration.

Usage:
  python scripts/measure_ink.py
  python scripts/measure_ink.py "/path/to/test manga"
  python scripts/measure_ink.py "/path/to/test manga" /tmp/ink.csv
  BUBBLE_CONF=0.65 python scripts/measure_ink.py ...
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from app.engine import create_bubble_detector
from app.settings import get_settings
from vision.bubble.ink_filter import measure_rows, settings_from_app

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_INPUT = Path("/Users/talo/Desktop/test and result/test manga")
DEFAULT_OUT = ROOT / "fixtures" / "out" / "ink_measure.csv"


def collect_images(src: Path) -> list[Path]:
    if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
        return [src]
    if src.is_dir():
        return sorted(
            (p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.name.lower(),
        )
    return []


def load_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str]) -> int:
    src = Path(argv[1]).expanduser().resolve() if len(argv) >= 2 else DEFAULT_INPUT
    out = Path(argv[2]).expanduser().resolve() if len(argv) >= 3 else DEFAULT_OUT

    if not src.exists():
        print(f"error: path does not exist: {src}", file=sys.stderr)
        return 1

    images = collect_images(src)
    if not images:
        print(f"error: no images in {src}", file=sys.stderr)
        return 1

    settings = get_settings()
    detector = create_bubble_detector(settings)
    detector.load()
    ink_settings = settings_from_app(settings)

    print(f"weight={settings.bubble_weight}")
    print(f"conf={settings.bubble_conf}  soft_hi={settings.bubble_conf_soft_hi}")
    print(
        f"ink_tol={settings.ink_tol}  ink_min_px={settings.ink_min_px}  "
        f"ink_min_cc_area={settings.ink_min_cc_area}"
    )
    print(f"input={src}  ({len(images)} image(s))")
    print(f"output={out}")

    rows: list[dict] = []
    for path in images:
        img = load_rgb(path)
        bubbles = detector.predict(img)
        page_rows = measure_rows(img, bubbles, ink_settings, page=path.stem)
        rows.extend(page_rows)
        n_drop = sum(1 for r in page_rows if r["decision"] == "DROP")
        n_fe = sum(1 for r in page_rows if r["decision"] == "FORCE_EDITOR")
        n_fc = sum(1 for r in page_rows if r["decision"] == "FILL_CANDIDATE")
        print(
            f"  {path.name}: n={len(page_rows)}  "
            f"DROP={n_drop} FORCE_EDITOR={n_fe} FILL_CANDIDATE={n_fc}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "page",
        "id",
        "conf",
        "bbox",
        "dark_px",
        "dark_frac",
        "cc_count",
        "max_cc_area",
        "ring_uniform",
        "text_present",
        "decision",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Highlight known problem pages for threshold picking
    focus = {"page_3", "page_8", "page_5", "page_14", "page_15", "page_2"}
    print("\n--- focus pages ---")
    for r in rows:
        if r["page"] in focus:
            print(
                f"  {r['page']} id={r['id']} conf={r['conf']:.3f} "
                f"dark_px={r['dark_px']} max_cc={r['max_cc_area']} "
                f"text={r['text_present']} unif={r['ring_uniform']} → {r['decision']}"
            )

    # Lowest-structure bubbles (punctuation / empty candidates)
    print("\n--- lowest dark_px (15) ---")
    for r in sorted(rows, key=lambda x: (x["dark_px"], x["max_cc_area"]))[:15]:
        print(
            f"  {r['page']} id={r['id']} dark_px={r['dark_px']} "
            f"max_cc={r['max_cc_area']} cc={r['cc_count']} → {r['decision']}"
        )

    print(f"\ndone: {len(rows)} bubbles → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
