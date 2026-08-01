"""Build PageDoc layers from clean fills + typeset layouts."""

from __future__ import annotations

from typing import List, Optional, Sequence

from vision.clean.fill import CleanFill, CleanOutcome
from vision.document import DocFlags, Layer, MaskSpec, PageDoc, SourceRef
from vision.masks import as_bbox
from vision.types import Bubble, BubbleLayout, BubbleText
from vision.typeset.service import TypesetResult


def clean_layers(fills: Sequence[CleanFill]) -> List[Layer]:
    """One enabled clean layer per successful inset-fill."""
    out: List[Layer] = []
    for f in fills:
        bid = int(f.bubble_id)
        bbox = as_bbox(f.bbox)
        out.append(
            Layer(
                id=f"cln_{bid}",
                kind="clean",
                name=f"Clean / bubble {bid}",
                group=f"bubble_{bid}",
                bubble_id=bid,
                enabled=True,
                z=100 + bid,
                source_step="clean.inset_fill.v1",
                bbox=bbox,
                mask=MaskSpec(
                    type="inset_polygon",
                    polygon=list(f.polygon),
                    inset=int(f.inset),
                    bbox=bbox,
                ),
                fill_color=(int(f.color[0]), int(f.color[1]), int(f.color[2])),
            )
        )
    return out


def text_layers(
    layouts: Sequence[BubbleLayout],
    bubbles: Sequence[Bubble],
) -> List[Layer]:
    """One text layer per layout; escalated bubbles stay disabled with reason on layout."""
    by_id = {int(b.id): b for b in bubbles}
    out: List[Layer] = []
    for lay in layouts:
        bid = int(lay.bubble_id)
        b = by_id.get(bid)
        bbox = as_bbox(b.bbox) if b is not None else (0, 0, 0, 0)
        enabled = lay.status == "ok" and bool(lay.lines)
        frame = lay.frame
        out.append(
            Layer(
                id=f"txt_{bid}",
                kind="text",
                name=f"Text / bubble {bid}",
                group=f"bubble_{bid}",
                bubble_id=bid,
                enabled=enabled,
                z=200 + bid,
                source_step="typeset.v1",
                bbox=bbox,
                layout=lay,
                frame=frame,
            )
        )
    return out


def build_doc(
    source: SourceRef,
    bubbles: Sequence[Bubble],
    texts: Sequence[BubbleText],
    clean_outcome: CleanOutcome,
    typeset_result: Optional[TypesetResult] = None,
    *,
    page_id: str,
) -> PageDoc:
    """Assemble PageDoc from clean + optional typeset results."""
    cl = clean_layers(clean_outcome.fills)
    tl: List[Layer] = []
    typeset_status = "skipped"
    typeset_editor_ids: List[int] = []
    if typeset_result is not None:
        tl = text_layers(typeset_result.layouts, bubbles)
        typeset_status = typeset_result.status
        typeset_editor_ids = [
            int(L.bubble_id)
            for L in typeset_result.layouts
            if L.status == "editor"
        ]

    layers = sorted(cl + tl, key=lambda L: (L.z, L.id))
    return PageDoc(
        schema_version=1,
        page_id=page_id,
        source=source,
        bubbles=list(bubbles),
        texts=list(texts),
        layers=layers,
        flags=DocFlags(
            clean=clean_outcome.status,
            typeset=typeset_status,
            editor_bubble_ids=list(clean_outcome.editor_bubble_ids),
            typeset_editor_bubble_ids=typeset_editor_ids,
        ),
    )
