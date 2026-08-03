"""Font catalog: scan assets/fonts for .ttf/.otf faces."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from vision.typeset.font import FontFace


class FontInfo(BaseModel):
    key: str
    label: str
    file: str
    postscript_name: str


def _fonts_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"


def _slug(stem: str) -> str:
    # ComicNeue-Regular → comic_neue; BubbleSans-Regular → bubble_sans
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", stem)
    s = re.sub(r"[-_\s]+", "_", s)
    s = re.sub(r"_regular$", "", s, flags=re.IGNORECASE)
    return s.lower().strip("_")


def _label(stem: str, family: str) -> str:
    # Prefer human family from the face; fall back to spaced stem.
    if family and family.strip():
        return family.strip()
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)
    spaced = re.sub(r"[-_]+", " ", spaced)
    spaced = re.sub(r"\s+Regular$", "", spaced, flags=re.IGNORECASE)
    return spaced.strip() or stem


@lru_cache(maxsize=1)
def list_fonts() -> tuple[FontInfo, ...]:
    root = _fonts_dir()
    if not root.is_dir():
        return ()
    out: List[FontInfo] = []
    seen_keys: set[str] = set()
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".ttf", ".otf"}:
            continue
        try:
            face = FontFace(path)
        except Exception:
            continue
        key = _slug(path.stem)
        if key in seen_keys:
            key = _slug(path.name)  # include extension uniqueness
        seen_keys.add(key)
        out.append(
            FontInfo(
                key=key,
                label=_label(path.stem, face.family),
                file=path.name,
                postscript_name=face.postscript_name,
            )
        )
    return tuple(out)


def clear_font_catalog() -> None:
    list_fonts.cache_clear()


def resolve_font_path(
    font: Optional[str],
    *,
    default: Optional[Path] = None,
) -> Path:
    """Resolve catalog key or filename to an absolute font path."""
    root = _fonts_dir()
    if not font or not str(font).strip():
        if default is not None:
            return Path(default).resolve()
        raise FileNotFoundError("no font specified and no default")

    raw = str(font).strip()
    # Absolute / existing path
    cand = Path(raw)
    if cand.is_file():
        return cand.resolve()

    # Filename in assets/fonts
    by_name = root / raw
    if by_name.is_file():
        return by_name.resolve()

    # Catalog key
    for info in list_fonts():
        if info.key == raw or info.file == raw:
            return (root / info.file).resolve()

    # Stem match (ComicNeue-Regular)
    for path in root.glob("*"):
        if path.suffix.lower() in {".ttf", ".otf"} and path.stem == raw:
            return path.resolve()

    raise FileNotFoundError(f"font not found: {font}")
