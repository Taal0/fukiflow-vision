"""Greedy word-boundary wrap with pluggable hyphenation fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

# word → break offsets (indexes into the word where a hyphen may be inserted)
Hyphenator = Callable[[str], Sequence[int]]


def pyphen_hyphenator(
    lang: str = "tr_TR",
    *,
    dic_path: Optional[Union[str, Path]] = None,
) -> Hyphenator:
    """Adapter around pyphen.

    pyphen's wheel does not ship Turkish patterns, so prefer ``dic_path``
    (bundled ``assets/hyphen/hyph_tr_TR.dic``). ``lang`` is a fallback for
    locales that *are* bundled (e.g. ``en_US``).
    """
    import pyphen

    if dic_path is not None:
        dic = pyphen.Pyphen(filename=str(Path(dic_path).resolve()))
    else:
        dic = pyphen.Pyphen(lang=lang)

    def positions(word: str) -> List[int]:
        if not word:
            return []
        return list(dic.positions(word))

    return positions


def _split_affixes(token: str) -> Tuple[str, str, str]:
    """Split leading/trailing non-letters so hyphenation sees the core word."""
    if not token:
        return "", "", ""
    i = 0
    while i < len(token) and not token[i].isalpha():
        i += 1
    j = len(token)
    while j > i and not token[j - 1].isalpha():
        j -= 1
    return token[:i], token[i:j], token[j:]


def _hyphenate_fit(
    word: str,
    max_w: float,
    measure: Callable[[str], float],
    hyphenator: Hyphenator,
) -> Optional[Tuple[str, str]]:
    """Longest prefix that fits as ``prefix-``; remainder is the rest of the word.

    Tries break points from longest to shortest. Returns None if nothing fits.
    Punctuation affixes stay attached to the core (prefix on head, suffix on tail).
    """
    lead, core, trail = _split_affixes(word)
    if len(core) < 4:
        return None
    positions = sorted(hyphenator(core), reverse=True)
    for i in positions:
        if i <= 0 or i >= len(core):
            continue
        head = lead + core[:i] + "-"
        if measure(head) <= max_w:
            return head, core[i:] + trail
    return None


def _wrap_paragraph(
    words: List[str],
    line_widths: Sequence[float],
    measure: Callable[[str], float],
    *,
    li: int,
    hyphenator: Optional[Hyphenator] = None,
) -> Optional[Tuple[List[str], bool, int]]:
    """Greedy-wrap one paragraph's words starting at slot ``li``.

    Returns ``(lines, hyphenated, next_li)`` or ``None`` if it doesn't fit.
    """
    lines: List[str] = []
    hyphenated = False
    wi = 0
    current = ""

    while wi < len(words):
        if li >= len(line_widths):
            return None
        max_w = float(line_widths[li])
        if max_w <= 0:
            return None

        word = words[wi]
        candidate = word if not current else f"{current} {word}"
        if measure(candidate) <= max_w:
            current = candidate
            wi += 1
            continue

        # Current line has content → commit and retry word on next line.
        if current:
            lines.append(current)
            current = ""
            li += 1
            continue

        # Word alone does not fit this empty line → hyphenate or fail.
        if hyphenator is not None:
            split = _hyphenate_fit(word, max_w, measure, hyphenator)
            if split is not None:
                head, rest = split
                lines.append(head)
                hyphenated = True
                words[wi] = rest
                li += 1
                current = ""
                continue

        # Cannot place this word on this line width.
        return None

    if current:
        if li >= len(line_widths):
            # Shouldn't happen: we had room while accumulating.
            return None
        lines.append(current)
        li += 1

    return lines, hyphenated, li


def wrap_lines(
    text: str,
    line_widths: Sequence[float],
    measure: Callable[[str], float],
    *,
    hyphenator: Optional[Hyphenator] = None,
) -> Tuple[Optional[List[str]], bool]:
    """Greedy wrap into ``len(line_widths)`` slots (or fewer if text ends early).

    A literal ``\\n`` in ``text`` is honored as a forced line break — the
    letterer's manual line splits (typed in the sidebar box) always win;
    each resulting segment is still auto-wrapped if it doesn't fit its slot.

    Returns ``(lines, hyphenated)`` or ``(None, hyphenated)`` if text does not
    fit the provided width schedule (too few lines / a token never fits).
    """
    if not text.strip():
        return [], False
    if not line_widths:
        return None, False

    paragraphs = text.split("\n")
    lines: List[str] = []
    hyphenated = False
    li = 0

    for para in paragraphs:
        words = para.split()
        if not words:
            # Explicit blank line — still consumes one vertical slot.
            if li >= len(line_widths):
                return None, hyphenated
            lines.append("")
            li += 1
            continue

        result = _wrap_paragraph(words, line_widths, measure, li=li, hyphenator=hyphenator)
        if result is None:
            return None, hyphenated
        para_lines, para_hyphenated, li = result
        lines.extend(para_lines)
        hyphenated = hyphenated or para_hyphenated

    return lines, hyphenated
