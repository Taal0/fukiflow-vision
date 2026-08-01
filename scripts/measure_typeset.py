#!/usr/bin/env python3
"""Measure multi-lobe / hyphen / typeset-escalation rates on clean pages.

Runs YOLO detect locally (no API), builds width profiles, then optional layout
fit with synthetic Turkish text sized to bubble area.

Usage:
  python scripts/measure_typeset.py
  python scripts/measure_typeset.py /path/to/datasets/clean
  python scripts/measure_typeset.py /path/to/datasets/clean /tmp/typeset_measure.csv
  TYPESET_HYPHENATE=0 python scripts/measure_typeset.py ...
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from app.engine import create_bubble_detector, create_typeset_service
from app.settings import get_settings
from vision.types import BubbleText

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_INPUT = (
    ROOT.parent / "fukiflow-train" / "datasets" / "clean"
).resolve()
DEFAULT_OUT = ROOT / "fixtures" / "out" / "typeset_measure.csv"

# Synthetic TR lines of increasing length — picked by bubble area percentile.
_TR_POOL = [
    "Tamam.",
    "Ne?",
    "Bekle bir saniye.",
    "Hiçbir şey anlamıyorum.",
    "Bu kadar hızlı olamaz!",
    "Sen gerçekten bunu mu düşünüyorsun?",
    "Konuşamayız mı artık? Lütfen dinle.",
    "Yarın görüşürüz o zaman, söz veriyorum.",
    "Özür dilerim, yanıldım; bunu daha önce fark etmeliydim.",
    (
        "Bütün bunlar birdenbire mi oldu, yoksa uzun zamandır mı "
        "hazırlanıyordu? Anlatır mısın?"
    ),
]


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


def _pick_text(area: float, areas: list[float]) -> str:
    if not areas:
        return _TR_POOL[4]
    ranked = sorted(areas)
    # Percentile index into pool.
    try:
        idx = ranked.index(min(ranked, key=lambda a: abs(a - area)))
    except ValueError:
        idx = 0
    frac = idx / max(1, len(ranked) - 1)
    return _TR_POOL[min(len(_TR_POOL) - 1, int(round(frac * (len(_TR_POOL) - 1))))]


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
    typesetter = create_typeset_service(settings)
    typesetter.load()

    margin_ratio = settings.typeset_margin_ratio
    clean_inset = settings.clean_inset
    lobe_ratio = settings.typeset_multi_lobe_min_area_ratio

    print(f"weight={settings.bubble_weight}")
    print(f"font={settings.typeset_font}")
    print(f"hyphenate={settings.typeset_hyphenate} lang={settings.typeset_hyphen_lang}")
    print(f"clean_inset={clean_inset} margin_ratio={margin_ratio}")
    print(f"multi_lobe_min_area_ratio={lobe_ratio}")
    print(f"input={src}  ({len(images)} image(s))")
    print(f"output={out}")

    rows: list[dict] = []
    n_multi = 0
    n_bubbles = 0
    necks: list[float] = []
    sec_ratios: list[float] = []
    n_ignored_fragments = 0
    n_hyphen = 0
    n_ok = 0
    n_editor = 0
    reason_counts: dict[str, int] = {}

    for path in images:
        img = load_rgb(path)
        h, w = img.shape[:2]
        margin_px = max(0, int(round(h * margin_ratio)))
        bubbles = detector.predict(img)
        areas: list[float] = []
        for b in bubbles:
            x1, y1, x2, y2 = b.bbox
            areas.append(float(max(0, x2 - x1) * max(0, y2 - y1)))

        texts = [
            BubbleText(bubble_id=int(b.id), text=_pick_text(areas[i], areas))
            for i, b in enumerate(bubbles)
        ]
        # Mark all unknown (no prior clean) — measure fit/lobe, not clean skip.
        stamped = [b.model_copy(update={"clean_status": "filled"}) for b in bubbles]
        result = typesetter.typeset(img, stamped, texts)

        for b, lay in zip(stamped, result.layouts):
            prof = result.profiles.get(int(b.id))
            multi = bool(prof.multi_lobe) if prof else False
            neck = float(prof.neck_depth) if prof else 0.0
            cc_count = int(prof.cc_count) if prof else 0
            primary = int(prof.primary_area) if prof else 0
            secondary = int(prof.secondary_area) if prof else 0
            sec_ratio = float(prof.secondary_area_ratio) if prof else 0.0
            ignored = int(prof.ignored_fragments) if prof else 0
            n_bubbles += 1
            if multi:
                n_multi += 1
            necks.append(neck)
            if sec_ratio > 0:
                sec_ratios.append(sec_ratio)
            n_ignored_fragments += ignored
            if lay.hyphenated:
                n_hyphen += 1
            if lay.status == "ok":
                n_ok += 1
            else:
                n_editor += 1
                reason_counts[lay.reason or "editor"] = (
                    reason_counts.get(lay.reason or "editor", 0) + 1
                )

            rows.append(
                {
                    "page": path.stem,
                    "id": b.id,
                    "multi_lobe": int(multi),
                    "cc_count": cc_count,
                    "primary_area": primary,
                    "secondary_area": secondary,
                    "secondary_area_ratio": f"{sec_ratio:.4f}",
                    "ignored_fragments": ignored,
                    "neck_depth": f"{neck:.4f}",
                    "status": lay.status,
                    "reason": lay.reason or "",
                    "hyphenated": int(lay.hyphenated),
                    "font_px": f"{lay.font_px:.2f}" if lay.font_px else "",
                    "n_lines": len(lay.lines),
                }
            )

        n_page_multi = sum(
            1
            for b in stamped
            if result.profiles.get(int(b.id))
            and result.profiles[int(b.id)].multi_lobe
        )
        n_page_ed = sum(1 for lay in result.layouts if lay.status == "editor")
        print(
            f"  {path.name}: n={len(bubbles)} multi={n_page_multi} "
            f"editor={n_page_ed} typeset={result.status}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "page",
        "id",
        "multi_lobe",
        "cc_count",
        "primary_area",
        "secondary_area",
        "secondary_area_ratio",
        "ignored_fragments",
        "neck_depth",
        "status",
        "reason",
        "hyphenated",
        "font_px",
        "n_lines",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("\n=== summary ===")
    if n_bubbles == 0:
        print("no bubbles")
        return 0

    multi_rate = 100.0 * n_multi / n_bubbles
    hyphen_rate = 100.0 * n_hyphen / n_bubbles
    esc_rate = 100.0 * n_editor / n_bubbles
    print(f"bubbles={n_bubbles} pages={len(images)}")
    print(f"multi_lobe={n_multi} ({multi_rate:.2f}%)")
    print(f"ignored_tail_fragments={n_ignored_fragments}")
    if necks:
        qs = statistics.quantiles(necks, n=4) if len(necks) >= 4 else necks
        print(
            f"neck_depth: mean={statistics.mean(necks):.3f} "
            f"median={statistics.median(necks):.3f} "
            f"p75={qs[-1] if qs else 0:.3f} max={max(necks):.3f}"
        )
    if sec_ratios:
        qs = statistics.quantiles(sec_ratios, n=4) if len(sec_ratios) >= 4 else sec_ratios
        print(
            f"secondary_area_ratio (cc≥2): n={len(sec_ratios)} "
            f"mean={statistics.mean(sec_ratios):.3f} "
            f"median={statistics.median(sec_ratios):.3f} "
            f"p75={qs[-1] if qs else 0:.3f} max={max(sec_ratios):.3f}"
        )
        # How many would escalate at common relative cutoffs.
        for cut in (0.10, 0.15, 0.20, 0.25, 0.30):
            n_cut = sum(1 for r in sec_ratios if r >= cut)
            print(f"  ratio≥{cut:.2f}: {n_cut} ({100.0 * n_cut / n_bubbles:.2f}% of bubbles)")
    print(f"hyphenated={n_hyphen} ({hyphen_rate:.2f}%)")
    print(f"escalated={n_editor} ({esc_rate:.2f}%)  ok={n_ok}")
    if reason_counts:
        print("reasons:", ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items())))
    print(f"\ndone → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
