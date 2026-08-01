#!/usr/bin/env python3
"""Batch process manga pages via the local vision API (bubbles → clean → typeset).

Usage:
  python scripts/batch_clean.py
  python scripts/batch_clean.py /path/to/page.png
  python scripts/batch_clean.py "/path/to/test manga"
  python scripts/batch_clean.py "/path/to/test manga" /path/to/custom-out
  HOST=http://127.0.0.1:8001 python scripts/batch_clean.py ...
  TEXTS_JSON=/path/to/manga_texts_tr_reading_order.json python scripts/batch_clean.py ...
  SKIP_TYPESET=1 python scripts/batch_clean.py ...

Does not modify the input. Default input (no args):
  /Users/talo/Desktop/test and result/test manga
Default output (folder input): sibling ``<name>-translated v<N>`` with N = max existing + 1.
Default TEXTS_JSON:
  /Users/talo/Desktop/test and result/extracted/manga_texts_tr_reading_order.json

Writes per page:
  <stem>-orig.<ext>    copy of the source
  <stem>.png            cleaned result
  <stem>-detect.png     original + YOLO polygons (reading_order labels)
  <stem>-bubbles.json   bubbles + clean/editor_regions
  <stem>-typeset.json   layout JSON + typeset status/regions
  <stem>-typeset.png    debug: text on cleaned page (+ editor marks)

On name collision, writes *-overwritten / *-overwritten-N instead of overwriting.
Stdlib only (no pip deps) so it can run against a remote API from another machine.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
HOST = os.environ.get("HOST", "http://127.0.0.1:8001").rstrip("/")
TIMEOUT = float(os.environ.get("BATCH_TIMEOUT", "600"))
DEFAULT_INPUT = Path("/Users/talo/Desktop/test and result/test manga")
DEFAULT_TEXTS_JSON = Path(
    "/Users/talo/Desktop/test and result/test manga text/manga_texts_tr_reading_order.json"
)
SKIP_TYPESET = os.environ.get("SKIP_TYPESET", "").lower() in ("1", "true", "yes")
_VERSIONED_OUT_RE = re.compile(r"^(.+)-translated v(\d+)$")


def usage() -> None:
    print(
        "Usage: batch_clean.py [image-or-folder] [output-folder]\n"
        f"  default input: {DEFAULT_INPUT}\n"
        f"  TEXTS_JSON={DEFAULT_TEXTS_JSON}\n"
        "  SKIP_TYPESET=1 to skip layout step\n"
        "  HOST=http://127.0.0.1:8001  (default)",
        file=sys.stderr,
    )


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def collect_inputs(src: Path) -> list[Path]:
    if is_image(src):
        return [src]
    if src.is_dir():
        return sorted(
            (p for p in src.iterdir() if is_image(p)),
            key=lambda p: p.name.lower(),
        )
    return []


def next_versioned_out_dir(src: Path) -> Path:
    """``<parent>/<name>-translated v<N>`` with N = max existing + 1."""
    parent = src.parent
    base = src.name
    max_n = 0
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        m = _VERSIONED_OUT_RE.match(child.name)
        if not m or m.group(1) != base:
            continue
        max_n = max(max_n, int(m.group(2)))
    return parent / f"{base}-translated v{max_n + 1}"


def default_out_dir(src: Path) -> Path:
    if src.is_dir():
        return next_versioned_out_dir(src)
    return src.parent / f"{src.stem}-translated"


def unique_path(path: Path) -> Path:
    """Return path, or path with -overwritten / -overwritten-N if taken."""
    if not path.exists():
        return path
    candidate = path.with_name(f"{path.stem}-overwritten{path.suffix}")
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}-overwritten-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _multipart(
    fields: dict[str, tuple[str, bytes, str] | str | bytes],
) -> tuple[bytes, str]:
    """Build multipart body. Values are either file triples
    (filename, data, content_type) or plain form strings/bytes (no filename)."""
    boundary = f"----ffbatch{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        if isinstance(value, tuple):
            filename, data, content_type = value
            chunks.append(
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode()
            )
            chunks.append(data)
        else:
            data = value.encode("utf-8") if isinstance(value, str) else value
            chunks.append(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            )
            chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def http_json(
    method: str,
    url: str,
    body: bytes | None = None,
    content_type: str | None = None,
) -> Any:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    req = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype or raw[:1] == b"{":
                return json.loads(raw.decode())
            return raw
    except HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"request failed {url}: {e.reason}") from e


def check_health(*, need_typeset: bool) -> None:
    url = f"{HOST}/health"
    try:
        data = http_json("GET", url)
    except Exception as e:
        raise SystemExit(f"error: health check failed ({url}): {e}") from e
    if not isinstance(data, dict) or data.get("status") != "ok":
        raise SystemExit(f"error: service not ready: {data}")
    models = data.get("models") or {}
    if not models.get("bubble"):
        raise SystemExit(f"error: models not loaded: {data}")
    if need_typeset and not models.get("typeset"):
        raise SystemExit(f"error: typeset service not loaded: {data}")


def load_texts_bundle(path: Path | None) -> dict[str, Any] | None:
    """Load manga_texts_tr_reading_order.json (or compatible)."""
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"error: TEXTS_JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("error: TEXTS_JSON must be a JSON object")
    by_page = data.get("typeset_by_page") or {}
    pages: dict[str, dict] = {}
    for pg in data.get("pages") or []:
        if not isinstance(pg, dict):
            continue
        fname = pg.get("file") or ""
        stem = Path(str(fname)).stem if fname else f"page_{pg.get('page')}"
        if stem:
            pages[stem] = pg
    return {"typeset_by_page": by_page, "pages": pages, "path": str(path)}


def resolve_texts(
    bundle: dict[str, Any] | None,
    stem: str,
    bubbles: list[dict],
) -> tuple[list[dict[str, Any]], str]:
    """Return (texts_json entries, how they were resolved)."""
    if bundle is None:
        return [], "none"

    raw = list(bundle.get("typeset_by_page", {}).get(stem) or [])
    bubble_ids = {int(b["id"]) for b in bubbles if "id" in b}
    matched = [
        {"bubble_id": int(t["bubble_id"]), "text": str(t.get("text") or "")}
        for t in raw
        if "bubble_id" in t and int(t["bubble_id"]) in bubble_ids
    ]
    if matched:
        return matched, "bubble_id"

    # Detect ids drifted vs the texts file → pair by reading_order.
    page = (bundle.get("pages") or {}).get(stem)
    if not page:
        return matched, "missing"

    page_texts = sorted(
        (page.get("texts") or []),
        key=lambda t: (t.get("reading_order") is None, t.get("reading_order") or 0),
    )
    ordered_bubbles = sorted(
        [b for b in bubbles if b.get("reading_order") is not None],
        key=lambda b: b["reading_order"],
    )
    out: list[dict[str, Any]] = []
    for b, t in zip(ordered_bubbles, page_texts):
        out.append(
            {
                "bubble_id": int(b["id"]),
                "text": str(t.get("text") or ""),
            }
        )
    if out:
        return out, "reading_order"
    return [], "empty"


def process_one(
    src: Path,
    out_dir: Path,
    *,
    texts_bundle: dict[str, Any] | None,
    do_typeset: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_dest = unique_path(out_dir / f"{src.stem}-orig{src.suffix}")
    clean_dest = unique_path(out_dir / f"{src.stem}.png")
    detect_dest = unique_path(out_dir / f"{src.stem}-detect.png")
    bubbles_dest = unique_path(out_dir / f"{src.stem}-bubbles.json")

    shutil.copy2(src, orig_dest)

    img_bytes = src.read_bytes()
    mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"

    body, ctype = _multipart(
        {
            "file": (src.name, img_bytes, mime),
            "page_id": src.stem,
        }
    )
    payload = http_json("POST", f"{HOST}/v1/pipeline", body=body, content_type=ctype)
    if not isinstance(payload, dict) or "bubbles" not in payload:
        raise RuntimeError("pipeline response missing bubbles")
    if not payload.get("cleaned_image"):
        raise RuntimeError("pipeline response missing cleaned_image")
    if not payload.get("bubbles_overlay"):
        raise RuntimeError("pipeline response missing bubbles_overlay")

    cleaned = base64.b64decode(payload["cleaned_image"])
    clean_dest.write_bytes(cleaned)
    detect_dest.write_bytes(base64.b64decode(payload["bubbles_overlay"]))

    bubbles = payload.get("bubbles") or []
    editor_regions = payload.get("editor_regions") or []
    bubbles_dest.write_text(
        json.dumps(
            {
                "page_id": payload.get("page_id"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "clean": payload.get("clean", "ok"),
                "editor_regions": editor_regions,
                "bubbles": bubbles,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    clean_flag = payload.get("clean", "ok")
    typeset_note = ""

    if do_typeset:
        texts, how = resolve_texts(texts_bundle, src.stem, bubbles)
        typeset_dest = unique_path(out_dir / f"{src.stem}-typeset.json")
        typeset_png_dest = unique_path(out_dir / f"{src.stem}-typeset.png")

        tbody, tctype = _multipart(
            {
                "file": (clean_dest.name, cleaned, "image/png"),
                "bubbles_json": json.dumps(bubbles, ensure_ascii=False),
                "texts_json": json.dumps(texts, ensure_ascii=False),
                "page_id": src.stem,
                "include_debug": "true",
            }
        )
        tpayload = http_json(
            "POST", f"{HOST}/v1/typeset", body=tbody, content_type=tctype
        )
        if not isinstance(tpayload, dict) or "layouts" not in tpayload:
            raise RuntimeError("typeset response missing layouts")

        if tpayload.get("debug_image"):
            typeset_png_dest.write_bytes(base64.b64decode(tpayload["debug_image"]))

        layouts = tpayload.get("layouts") or []
        n_ok = sum(1 for L in layouts if L.get("status") == "ok")
        n_ed = sum(1 for L in layouts if L.get("status") == "editor")
        typeset_dest.write_text(
            json.dumps(
                {
                    "page_id": tpayload.get("page_id"),
                    "width": tpayload.get("width"),
                    "height": tpayload.get("height"),
                    "typeset": tpayload.get("typeset", "ok"),
                    "typeset_regions": tpayload.get("typeset_regions") or [],
                    "texts_resolve": how,
                    "texts": texts,
                    "layouts": layouts,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        typeset_note = (
            f"  typeset={tpayload.get('typeset')} ok={n_ok} editor={n_ed} "
            f"texts={len(texts)}({how})  "
            f"typeset→{typeset_dest.name}  preview→{typeset_png_dest.name}"
        )

    print(
        f"OK  {src.name}  bubbles={len(bubbles)}  clean={clean_flag}  "
        f"editor={len(editor_regions)}  "
        f"orig→{orig_dest.name}  clean→{clean_dest.name}  "
        f"detect→{detect_dest.name}  json→{bubbles_dest.name}"
        f"{typeset_note}"
    )


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] in {"-h", "--help"}:
        usage()
        return 2

    if len(argv) >= 2:
        src = Path(argv[1]).expanduser().resolve()
    else:
        src = DEFAULT_INPUT.expanduser().resolve()

    out_override = Path(argv[2]).expanduser().resolve() if len(argv) >= 3 else None

    if not src.exists():
        print(f"error: path does not exist: {src}", file=sys.stderr)
        return 1
    if src.is_file() and not is_image(src):
        print(
            f"error: not a supported image ({', '.join(sorted(IMAGE_EXTS))}): {src}",
            file=sys.stderr,
        )
        return 1

    images = collect_inputs(src)
    if not images:
        kind = "folder" if src.is_dir() else "path"
        print(f"error: no images found in {kind}: {src}", file=sys.stderr)
        return 1

    out_dir = out_override if out_override is not None else default_out_dir(src)
    if src.is_dir() and out_dir.resolve() == src.resolve():
        print("error: output folder must differ from input folder", file=sys.stderr)
        return 1

    do_typeset = not SKIP_TYPESET
    texts_path: Path | None = None
    texts_bundle: dict[str, Any] | None = None
    if do_typeset:
        env_texts = os.environ.get("TEXTS_JSON", "").strip()
        texts_path = (
            Path(env_texts).expanduser().resolve()
            if env_texts
            else DEFAULT_TEXTS_JSON.expanduser().resolve()
        )
        texts_bundle = load_texts_bundle(texts_path)

    print(f"host={HOST}")
    print(f"input={src}  ({len(images)} image(s))")
    print(f"output={out_dir}")
    if do_typeset:
        print(f"texts={texts_path}")
    else:
        print("typeset=skipped")
    check_health(need_typeset=do_typeset)
    print("health=ok")

    ok = 0
    failed: list[str] = []
    for img in images:
        try:
            process_one(
                img,
                out_dir,
                texts_bundle=texts_bundle,
                do_typeset=do_typeset,
            )
            ok += 1
        except Exception as e:
            print(f"FAIL  {img.name}: {e}", file=sys.stderr)
            failed.append(img.name)

    print(f"done: ok={ok} fail={len(failed)} → {out_dir}")
    if failed:
        print("failed:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
