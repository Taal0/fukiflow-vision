#!/usr/bin/env python3
"""Visual eval for panel_v3 + RSP reading order (no API).

Runs bubble + panel detectors locally, writes overlay PNGs, prints per-page
orphan / fallback / order_confidence summary. Optional --gt JSON for
exact-match + Kendall tau.

Usage:
  python scripts/eval_reading_order.py
  python scripts/eval_reading_order.py "/path/to/test manga"
  python scripts/eval_reading_order.py "/path/to/test manga" /tmp/order_overlays
  python scripts/eval_reading_order.py "/path/to/test manga" --gt gt_order.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from app.engine import (
    create_bubble_detector,
    create_order_service,
    create_panel_detector,
)
from app.settings import get_settings
from vision.bubble.draw import draw_bubbles
from vision.bubble.ink_filter import classify, settings_from_app

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_INPUT = Path("/Users/talo/Desktop/test and result/test manga")
DEFAULT_OUT = ROOT / "fixtures" / "out" / "reading_order"


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


def kendall_tau(pred: list[int], gt: list[int]) -> float:
    """Kendall tau-b for two permutations of the same ids (pred/gt as ordered id lists)."""
    if len(pred) != len(gt) or not pred:
        return float("nan")
    rank_gt = {bid: i for i, bid in enumerate(gt)}
    try:
        ranks = [rank_gt[bid] for bid in pred]
    except KeyError:
        return float("nan")
    n = len(ranks)
    if n < 2:
        return 1.0
    concord = 0
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            # pairs compared by position in pred vs gt rank
            a = ranks[i] - ranks[j]
            if a == 0:
                continue
            # in pred, i appears before j; in gt, ranks[i] vs ranks[j]
            if a < 0:
                concord += 1
            else:
                discord += 1
    total = concord + discord
    return (concord - discord) / total if total else 1.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Eval panel + RSP reading order overlays")
    p.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="image file or folder",
    )
    p.add_argument(
        "outdir",
        nargs="?",
        default=str(DEFAULT_OUT),
        help="overlay output directory",
    )
    p.add_argument(
        "--gt",
        default=None,
        help="optional GT JSON: {page_stem: [bubble_id, ...]} in reading order",
    )
    args = p.parse_args(argv)

    src = Path(args.input).expanduser().resolve()
    out = Path(args.outdir).expanduser().resolve()
    if not src.exists():
        print(f"error: path does not exist: {src}", file=sys.stderr)
        return 1

    images = collect_images(src)
    if not images:
        print(f"error: no images in {src}", file=sys.stderr)
        return 1

    gt: dict = {}
    if args.gt:
        gt_path = Path(args.gt).expanduser().resolve()
        gt = json.loads(gt_path.read_text(encoding="utf-8"))

    settings = get_settings()
    bubble_det = create_bubble_detector(settings)
    panel_det = create_panel_detector(settings)
    order_svc = create_order_service(settings, panel_detector=panel_det)
    bubble_det.load()
    panel_det.load()
    order_svc.load()
    ink_settings = settings_from_app(settings)

    out.mkdir(parents=True, exist_ok=True)

    print(f"bubble_weight={settings.bubble_weight}")
    print(f"panel_weight={settings.panel_weight}  conf={settings.panel_conf}")
    print(
        f"order_tol_frac={settings.order_tol_frac}  "
        f"nms_iou={settings.panel_nms_iou}  conf_min={settings.order_conf_min}"
    )
    print(f"input={src}  ({len(images)} image(s))")
    print(f"output={out}")

    n_exact = 0
    n_gt = 0
    taus: list[float] = []

    for path in images:
        img = load_rgb(path)
        raw = bubble_det.predict(img)
        bubbles, _ = classify(img, raw, ink_settings)
        result = order_svc.order(img, bubbles)

        overlay = draw_bubbles(img, result.bubbles, panels=result.panels)
        out_path = out / f"{path.stem}-order.png"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        review = " REVIEW" if result.needs_review else ""
        print(
            f"  {path.name}: panels={len(result.panels)} bubbles={len(result.bubbles)} "
            f"orphan={result.n_orphan} ambig={result.n_ambiguous} "
            f"fallback={result.n_fallback}/{result.rsp_calls} "
            f"conf={result.order_confidence:.3f}{review}"
        )

        stem = path.stem
        if stem in gt:
            n_gt += 1
            pred_ids = [b.id for b in sorted(result.bubbles, key=lambda x: x.reading_order or 0)]
            gt_ids = list(gt[stem])
            exact = pred_ids == gt_ids
            if exact:
                n_exact += 1
            tau = kendall_tau(pred_ids, gt_ids)
            if tau == tau:  # not NaN
                taus.append(tau)
            print(
                f"    gt: exact={exact} kendall_tau={tau:.3f} "
                f"pred={pred_ids} gt={gt_ids}"
            )

    if n_gt:
        mean_tau = sum(taus) / len(taus) if taus else float("nan")
        print(
            f"\nGT summary: exact={n_exact}/{n_gt}  "
            f"mean_kendall_tau={mean_tau:.3f}"
        )

    print(f"\ndone: {len(images)} page(s) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
