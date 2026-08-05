#!/usr/bin/env python3
"""Compose parity / font-portability checks (runnable without CI).

Product contract: server compose must paint text even when layout.font_file is
a foreign absolute path, and paint layers must be deterministic.

Usage:
  python scripts/parity_compose.py
  python scripts/parity_compose.py --mae-threshold 0.5
  python scripts/parity_compose.py --write-golden fixtures/parity/golden.png
  python scripts/parity_compose.py --compare fixtures/parity/golden.png

Exit 0 on pass, 1 on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vision.document import Layer, PageDoc, SourceRef, Stroke  # noqa: E402
from vision.layers.compose import compose  # noqa: E402
from vision.types import BubbleLayout, TypesetLine  # noqa: E402
from vision.typeset.catalog import resolve_portable_font  # noqa: E402


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def _blank_source(h: int = 240, w: int = 320) -> np.ndarray:
    """Light page with a dark ink blob (for restore paint)."""
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    img[80:140, 100:180] = (40, 40, 40)
    return img


def _text_layer(*, font_file: str, font_px: float = 28.0) -> Layer:
    return Layer(
        id="txt_1",
        kind="text",
        name="Text / bubble 1",
        z=200,
        enabled=True,
        bubble_id=1,
        bbox=(30, 30, 200, 90),
        layout=BubbleLayout(
            bubble_id=1,
            status="ok",
            font_px=font_px,
            font_file=font_file,
            lines=[
                TypesetLine(text="Merhaba", x=40.0, baseline_y=60.0, width=120.0),
            ],
        ),
    )


def _paint_layers() -> list[Layer]:
    return [
        Layer(
            id="pnt_1",
            kind="paint",
            name="Paint / bubble 1",
            z=150,
            enabled=True,
            bubble_id=1,
            bbox=(100, 80, 180, 140),
            strokes=[
                Stroke(
                    id="pnt_fill_1",
                    mode="fill",
                    color=(255, 255, 255),
                    radius_px=8.0,
                    points=[[120.0, 100.0], [140.0, 110.0]],
                    bubble_id=1,
                ),
                Stroke(
                    id="pnt_restore_1",
                    mode="restore",
                    radius_px=4.0,
                    points=[[130.0, 105.0]],
                    bubble_id=1,
                ),
            ],
        )
    ]


def _doc(layers: list[Layer], *, h: int = 240, w: int = 320) -> PageDoc:
    return PageDoc(
        page_id="parity",
        source=SourceRef(file="synthetic.png", width=w, height=h),
        bubbles=[],
        layers=layers,
    )


def test_portable_font_matches_catalog(threshold: float) -> None:
    """Dead Mac absolute font_file must compose identical to catalog basename."""
    font_name = "BubbleSans-Regular.otf"
    real = _ROOT / "assets" / "fonts" / font_name
    if not real.is_file():
        raise SystemExit(f"missing font fixture: {real}")

    resolved = resolve_portable_font(
        f"/Users/other-machine/Library/Fonts/{font_name}",
        fallback=None,
    )
    if resolved is None or resolved.resolve() != real.resolve():
        raise AssertionError(
            f"portable resolve failed: got {resolved}, expected {real}"
        )

    source = _blank_source()
    via_catalog = compose(
        source,
        _doc([_text_layer(font_file=font_name)]),
        font_path=str(real),
    )
    via_dead_abs = compose(
        source,
        _doc(
            [
                _text_layer(
                    font_file=f"/Users/other-machine/does-not-exist/{font_name}"
                )
            ]
        ),
        font_path=str(real),
    )
    via_fallback = compose(
        source,
        _doc([_text_layer(font_file="/no/such/TotallyMissing.otf")]),
        font_path=str(real),
    )

    mae_cat = _mae(via_catalog, via_dead_abs)
    if mae_cat > threshold:
        raise AssertionError(
            f"catalog vs dead-abs font MAE {mae_cat:.4f} > {threshold}"
        )

    mae_fb = _mae(via_fallback, source)
    if mae_fb < 0.05:
        raise AssertionError(
            "fallback font_path did not paint text (compose silently skipped)"
        )

    ink = int(np.any(via_dead_abs < 200, axis=2).sum())
    if ink < 20:
        raise AssertionError(
            f"dead-abs font_file produced almost no ink ({ink} dark px)"
        )

    print(f"ok  portable font  MAE(catalog, dead-abs)={mae_cat:.6f}  ink={ink}")


def test_paint_changes_pixels(threshold: float) -> None:
    source = _blank_source()
    base = compose(source, _doc([]))
    painted = compose(source, _doc(_paint_layers()))
    mae = _mae(base, painted)
    if mae < 0.01:
        raise AssertionError("paint layer did not change the compose output")
    again = compose(source, _doc(_paint_layers()))
    mae2 = _mae(painted, again)
    if mae2 > threshold:
        raise AssertionError(f"paint compose not deterministic MAE={mae2}")
    print(f"ok  paint layer    MAE(base, painted)={mae:.6f}")


def test_golden(
    path: Path,
    *,
    write: bool,
    threshold: float,
) -> None:
    source = _blank_source()
    font = str(_ROOT / "assets" / "fonts" / "BubbleSans-Regular.otf")
    layers = [
        *_paint_layers(),
        _text_layer(font_file="/Users/ci-foreign/BubbleSans-Regular.otf"),
    ]
    out = compose(source, _doc(layers), font_path=font)

    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(path)
        print(f"ok  wrote golden   {path}")
        return

    if not path.is_file():
        raise SystemExit(
            f"golden not found: {path}\n"
            f"  create with: python scripts/parity_compose.py --write-golden {path}"
        )
    gold = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    mae = _mae(out, gold)
    mx = _max_abs(out, gold)
    if mae > threshold:
        raise AssertionError(
            f"golden MAE {mae:.4f} > {threshold} (max abs {mx:.1f})"
        )
    print(f"ok  golden         MAE={mae:.6f}  max_abs={mx:.1f}  path={path}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Compose parity / portability checks")
    ap.add_argument(
        "--mae-threshold",
        type=float,
        default=0.5,
        help="Mean absolute error threshold (0–255 scale)",
    )
    ap.add_argument(
        "--write-golden",
        type=Path,
        default=None,
        help="Write current compose output as a golden PNG and exit",
    )
    ap.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Compare compose output against an existing golden PNG",
    )
    args = ap.parse_args(argv[1:])

    try:
        test_portable_font_matches_catalog(args.mae_threshold)
        test_paint_changes_pixels(args.mae_threshold)
        if args.write_golden is not None:
            test_golden(args.write_golden, write=True, threshold=args.mae_threshold)
        elif args.compare is not None:
            test_golden(args.compare, write=False, threshold=args.mae_threshold)
        else:
            default_golden = _ROOT / "fixtures" / "parity" / "golden.png"
            if default_golden.is_file():
                test_golden(
                    default_golden, write=False, threshold=args.mae_threshold
                )
            else:
                print(
                    "skip golden (no fixtures/parity/golden.png — "
                    "pass --write-golden to create)"
                )
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print("all parity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
