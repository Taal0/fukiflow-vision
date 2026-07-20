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
  <stem>-editor.json   (only if X-Editor-Regions non-empty)
  <stem>-editor.png    cleaned + red boxes (same condition)

On name collision, writes *-overwritten / *-overwritten-N instead of overwriting.
Stdlib only (no pip deps) so it can run against a remote API from another machine.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import struct
import sys
import uuid
import zlib
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


def http_bytes_and_headers(
    url: str, body: bytes, content_type: str
) -> tuple[bytes, dict[str, str]]:
    req = Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.read(), headers
    except HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"request failed {url}: {e.reason}") from e


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _decode_png_rgb(data: bytes) -> tuple[int, int, bytearray]:
    """Decode 8-bit RGB/RGBA non-interlaced PNG → (w, h, RGB bytes)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos = 8
    width = height = None
    color_type = None
    bit_depth = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif tag == b"IDAT":
            idat.extend(chunk)
        elif tag == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in (2, 6):
        raise ValueError("unsupported PNG (need 8-bit RGB/RGBA)")
    raw = zlib.decompress(bytes(idat))
    bpp = 3 if color_type == 2 else 4
    stride = 1 + width * bpp
    if len(raw) != stride * height:
        raise ValueError("bad PNG size")
    prev = bytearray(width * bpp)
    out = bytearray(width * height * 3)
    for y in range(height):
        row = raw[y * stride : (y + 1) * stride]
        filt, scan = row[0], bytearray(row[1:])
        if filt == 1:  # Sub
            for i in range(bpp, len(scan)):
                scan[i] = (scan[i] + scan[i - bpp]) & 0xFF
        elif filt == 2:  # Up
            for i in range(len(scan)):
                scan[i] = (scan[i] + prev[i]) & 0xFF
        elif filt == 3:  # Average
            for i in range(len(scan)):
                left = scan[i - bpp] if i >= bpp else 0
                scan[i] = (scan[i] + ((left + prev[i]) // 2)) & 0xFF
        elif filt == 4:  # Paeth
            for i in range(len(scan)):
                a = scan[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                scan[i] = (scan[i] + pr) & 0xFF
        elif filt != 0:
            raise ValueError(f"unsupported PNG filter {filt}")
        prev = scan
        dst = y * width * 3
        if bpp == 3:
            out[dst : dst + width * 3] = scan
        else:
            for x in range(width):
                si = x * 4
                di = dst + x * 3
                out[di : di + 3] = scan[si : si + 3]
    return width, height, out


def _encode_png_rgb(width: int, height: int, rgb: bytes | bytearray) -> bytes:
    raw = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw.append(0)  # filter None
        start = y * row_bytes
        raw.extend(rgb[start : start + row_bytes])
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            _png_chunk(b"IEND", b""),
        ]
    )


def draw_editor_boxes(png_bytes: bytes, regions: list[list[int]]) -> bytes:
    """Draw red rectangles on a cleaned PNG (stdlib)."""
    w, h, rgb = _decode_png_rgb(png_bytes)
    thickness = 3
    for box in regions:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        for t in range(thickness):
            yy1, yy2 = min(h - 1, y1 + t), max(0, y2 - t)
            xx1, xx2 = min(w - 1, x1 + t), max(0, x2 - t)
            for x in range(xx1, xx2 + 1):
                for y in (yy1, yy2):
                    i = (y * w + x) * 3
                    rgb[i : i + 3] = b"\xff\x00\x00"
            for y in range(yy1, yy2 + 1):
                for x in (xx1, xx2):
                    i = (y * w + x) * 3
                    rgb[i : i + 3] = b"\xff\x00\x00"
    return _encode_png_rgb(w, h, rgb)


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
    boxes = payload.get("boxes") or []
    n_boxes = len(boxes)

    fields: dict[str, tuple[str, bytes, str] | str | bytes] = {
        "file": (src.name, img_bytes, mime),
        "mask": ("mask.png", mask_bytes, "image/png"),
    }
    if boxes:
        # Plain form field (Form str) — not a file part, or FastAPI 422s.
        fields["boxes_json"] = json.dumps(boxes)
    body, ctype = _multipart(fields)
    cleaned, headers = http_bytes_and_headers(f"{HOST}/v1/inpaint", body, ctype)
    clean_dest.write_bytes(cleaned)

    editor_regions: list[Any] = []
    raw_regions = headers.get("x-editor-regions", "").strip()
    if raw_regions:
        try:
            parsed = json.loads(raw_regions)
            if isinstance(parsed, list):
                editor_regions = parsed
        except json.JSONDecodeError:
            pass

    editor_n = len(editor_regions)
    extra = ""
    if editor_n:
        editor_json = unique_path(out_dir / f"{src.stem}-editor.json")
        editor_json.write_text(
            json.dumps({"editor_regions": editor_regions}, indent=2) + "\n",
            encoding="utf-8",
        )
        editor_png = unique_path(out_dir / f"{src.stem}-editor.png")
        try:
            editor_png.write_bytes(draw_editor_boxes(cleaned, editor_regions))
            extra = f"  editor={editor_n}→{editor_json.name},{editor_png.name}"
        except Exception as e:
            extra = f"  editor={editor_n}→{editor_json.name} (png failed: {e})"
    else:
        extra = "  editor=0"

    print(
        f"OK  {src.name}  boxes={n_boxes}  "
        f"orig→{orig_dest.name}  clean→{clean_dest.name}{extra}"
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
