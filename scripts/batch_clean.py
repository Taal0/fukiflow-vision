#!/usr/bin/env python3
"""Batch process manga pages via the local vision API (bubbles → ocr → clean).

Usage:
  python scripts/batch_clean.py
  python scripts/batch_clean.py /path/to/page.png
  python scripts/batch_clean.py "/path/to/test manga"
  python scripts/batch_clean.py "/path/to/test manga" /path/to/custom-out
  HOST=http://127.0.0.1:8001 python scripts/batch_clean.py ...

Does not modify the input. Default input (no args):
  /Users/talo/Desktop/test and result/test manga
Default output (folder input): sibling ``<name>-translated v<N>`` with N = max existing + 1.

Writes per page:
  <stem>-orig.<ext>   copy of the source
  <stem>.png           cleaned result
  <stem>-detect.png    original + YOLO polygons (labels match ocr.json n)
  <stem>-bubbles.json  bubbles + text_ja + clean/editor_regions

Plus run-level:
  ocr.json             { "<page_stem>": [{"n": 1, "text": "..."}, ...], ... }

On name collision, writes *-overwritten / *-overwritten-N instead of overwriting.
Stdlib only (no pip deps) so it can run against a remote API from another machine.
"""

from __future__ import annotations

import atexit
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
_VERSIONED_OUT_RE = re.compile(r"^(.+)-translated v(\d+)$")


def usage() -> None:
    print(
        "Usage: batch_clean.py [image-or-folder] [output-folder]\n"
        f"  default input: {DEFAULT_INPUT}\n"
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


def http_json(method: str, url: str, body: bytes | None = None, content_type: str | None = None) -> Any:
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


def check_health() -> None:
    url = f"{HOST}/health"
    try:
        data = http_json("GET", url)
    except Exception as e:
        raise SystemExit(f"error: health check failed ({url}): {e}") from e
    if not isinstance(data, dict) or data.get("status") != "ok":
        raise SystemExit(f"error: service not ready: {data}")
    models = data.get("models") or {}
    if not models.get("bubble") or not models.get("ocr"):
        raise SystemExit(f"error: models not loaded: {data}")


def process_one(src: Path, out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_dest = unique_path(out_dir / f"{src.stem}-orig{src.suffix}")
    clean_dest = unique_path(out_dir / f"{src.stem}.png")
    detect_dest = unique_path(out_dir / f"{src.stem}-detect.png")
    bubbles_dest = unique_path(out_dir / f"{src.stem}-bubbles.json")

    shutil.copy2(src, orig_dest)

    img_bytes = src.read_bytes()
    mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"

    body, ctype = _multipart({"file": (src.name, img_bytes, mime)})
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

    ocr_entries = [
        {"n": i + 1, "text": b.get("text_ja")}
        for i, b in enumerate(bubbles)
        if isinstance(b, dict)
    ]

    clean_flag = payload.get("clean", "ok")
    print(
        f"OK  {src.name}  bubbles={len(bubbles)}  clean={clean_flag}  "
        f"editor={len(editor_regions)}  "
        f"orig→{orig_dest.name}  clean→{clean_dest.name}  "
        f"detect→{detect_dest.name}  json→{bubbles_dest.name}"
    )
    return ocr_entries


def write_ocr_json(out_dir: Path, ocr_by_page: dict[str, list[dict[str, Any]]]) -> None:
    if not ocr_by_page:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "ocr.json"
    dest.write_text(
        json.dumps(ocr_by_page, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"ocr.json → {dest}  ({len(ocr_by_page)} page(s))")


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

    print(f"host={HOST}")
    print(f"input={src}  ({len(images)} image(s))")
    print(f"output={out_dir}")
    check_health()
    print("health=ok")

    ocr_by_page: dict[str, list[dict[str, Any]]] = {}
    ocr_written = False

    def _flush_ocr() -> None:
        nonlocal ocr_written
        if ocr_written or not ocr_by_page:
            return
        write_ocr_json(out_dir, ocr_by_page)
        ocr_written = True

    atexit.register(_flush_ocr)

    ok = 0
    failed: list[str] = []
    try:
        for img in images:
            try:
                ocr_by_page[img.stem] = process_one(img, out_dir)
                ok += 1
            except Exception as e:
                print(f"FAIL  {img.name}: {e}", file=sys.stderr)
                failed.append(img.name)
    finally:
        _flush_ocr()

    print(f"done: ok={ok} fail={len(failed)} → {out_dir}")
    if failed:
        print("failed:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
