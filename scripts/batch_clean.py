#!/usr/bin/env python3
"""Batch clean manga pages via the local vision API.

Usage:
  python scripts/batch_clean.py /path/to/page.png
  python scripts/batch_clean.py "/path/to/test manga"
  python scripts/batch_clean.py "/path/to/test manga" /path/to/custom-out
  HOST=http://127.0.0.1:8001 python scripts/batch_clean.py ...

Does not modify the input. Writes to a sibling *-translated/ folder (or arg2):
  <stem>-orig.<ext>   copy of the source
  <stem>.png           cleaned result

On name collision, writes *-overwritten / *-overwritten-N instead of overwriting.
Stdlib only (no pip deps) so it can run against a remote API from another machine.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
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


def usage() -> None:
    print(
        "Usage: batch_clean.py <image-or-folder> [output-folder]\n"
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


def default_out_dir(src: Path) -> Path:
    if src.is_dir():
        return src.parent / f"{src.name}-translated"
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
    fields: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----ffbatch{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, (filename, data, content_type) in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
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


def http_bytes(url: str, body: bytes, content_type: str) -> bytes:
    req = Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read()
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
    if not isinstance(data, dict) or data.get("status") != "ok" or not data.get("models_loaded"):
        raise SystemExit(f"error: service not ready: {data}")


def process_one(src: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_dest = unique_path(out_dir / f"{src.stem}-orig{src.suffix}")
    clean_dest = unique_path(out_dir / f"{src.stem}.png")

    shutil.copy2(src, orig_dest)

    img_bytes = src.read_bytes()
    mime = mimetypes.guess_type(src.name)[0] or "application/octet-stream"

    body, ctype = _multipart(
        {"file": (src.name, img_bytes, mime)},
    )
    payload = http_json("POST", f"{HOST}/v1/detect-ocr", body=body, content_type=ctype)
    if not isinstance(payload, dict) or not payload.get("mask"):
        raise RuntimeError("detect-ocr response missing mask")
    mask_bytes = base64.b64decode(payload["mask"])
    n_boxes = len(payload.get("boxes") or [])

    body, ctype = _multipart(
        {
            "file": (src.name, img_bytes, mime),
            "mask": ("mask.png", mask_bytes, "image/png"),
        },
    )
    cleaned = http_bytes(f"{HOST}/v1/inpaint", body, ctype)
    clean_dest.write_bytes(cleaned)
    print(
        f"OK  {src.name}  boxes={n_boxes}  "
        f"orig→{orig_dest.name}  clean→{clean_dest.name}"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in {"-h", "--help"}:
        usage()
        return 2

    src = Path(argv[1]).expanduser().resolve()
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

    ok = 0
    failed: list[str] = []
    for img in images:
        try:
            process_one(img, out_dir)
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
