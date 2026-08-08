"""Human-readable request/response dumps for the vision terminal."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

log = logging.getLogger("uvicorn.error")

_SKIP_PREFIXES = ("/docs", "/redoc", "/openapi", "/favicon")
_BUBBLE_LIMIT = 20
_PANEL_LIMIT = 12
_LAYOUT_LIMIT = 12
_RULE = "=" * 48

# path → short operation title for terminal scanning
_OPS: dict[str, str] = {
    "/v1/bubbles": "BUBBLE DETECT",
    "/v1/order": "READING ORDER",
    "/v1/clean": "CLEAN FILL",
    "/v1/pipeline": "FULL PIPELINE",
    "/v1/typeset": "TYPESET",
    "/v1/compose": "COMPOSE",
    "/health": "HEALTH",
    "/v1/typeset/fonts": "FONT CATALOG",
}


def _op_title(path: str) -> str:
    if path in _OPS:
        return _OPS[path]
    if path.startswith("/v1/typeset/fonts/"):
        return "FONT FILE"
    return path.upper().strip("/") or "REQUEST"


def _bbox(item: Mapping[str, Any]) -> str:
    b = item.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        return f"[{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}]"
    return "?"


def _conf(item: Mapping[str, Any], key: str = "confidence") -> str:
    raw = item.get(key)
    if isinstance(raw, (int, float)):
        return f"{float(raw):.3f}"
    return "?"


def _fmt_bubble(b: Mapping[str, Any]) -> str:
    bits = [
        f"#{b.get('id', '?')}",
        f"conf={_conf(b)}",
        f"bbox={_bbox(b)}",
    ]
    if b.get("reading_order") is not None:
        bits.append(f"order={b['reading_order']}")
    if b.get("panel_id") is not None:
        bits.append(f"panel={b['panel_id']}")
    cs = b.get("clean_status")
    if cs and cs != "unknown":
        bits.append(f"clean={cs}")
    am = b.get("assign_mode")
    if am and am != "unassigned":
        bits.append(am)
    return " ".join(bits)


def _fmt_panel(p: Mapping[str, Any]) -> str:
    bits = [
        f"#{p.get('id', '?')}",
        f"conf={_conf(p)}",
        f"bbox={_bbox(p)}",
    ]
    if p.get("reading_order"):
        bits.append(f"order={p['reading_order']}")
    if p.get("is_synthetic"):
        bits.append("synthetic")
    return " ".join(bits)


def _fmt_layout(L: Mapping[str, Any]) -> str:
    bits = [
        f"b{L.get('bubble_id', '?')}",
        f"status={L.get('status', '?')}",
    ]
    if L.get("reason"):
        bits.append(f"reason={L['reason']}")
    if isinstance(L.get("font_px"), (int, float)):
        bits.append(f"px={float(L['font_px']):.2f}")
    if L.get("overflow"):
        bits.append("overflow")
    return " ".join(bits)


def _list_block(title: str, lines: list[str], total: int, limit: int) -> list[str]:
    if total == 0:
        return [f"  {title}(0): (none)"]
    out = [f"  {title}({total}):"]
    for line in lines[:limit]:
        out.append(f"    {line}")
    if total > limit:
        out.append(f"    … +{total - limit} more")
    return out


def _result_line(path: str, data: Mapping[str, Any], status_code: int) -> str:
    """One plain-English verdict for the operation."""
    if status_code >= 400:
        detail = data.get("detail", "error")
        text = detail if isinstance(detail, str) else str(detail)
        return f"RESULT: failed ({status_code}) — {text[:160]}"

    if path == "/v1/bubbles" or path == "/v1/order" or path == "/v1/pipeline":
        bubbles = data.get("bubbles")
        n = len(bubbles) if isinstance(bubbles, list) else 0
        if n == 0:
            verdict = "no bubbles detected"
        elif n == 1:
            b0 = bubbles[0] if isinstance(bubbles[0], Mapping) else {}
            verdict = f"1 bubble detected (conf={_conf(b0)})"
        else:
            confs = [
                _conf(b)
                for b in bubbles
                if isinstance(b, Mapping)
            ][:6]
            verdict = f"{n} bubbles detected (conf={', '.join(confs)})"

        panels = data.get("panels")
        if isinstance(panels, list) and panels:
            real = sum(
                1
                for p in panels
                if isinstance(p, Mapping) and not p.get("is_synthetic")
            )
            if real:
                verdict += f", {real} panel(s)"
            elif any(
                isinstance(p, Mapping) and p.get("is_synthetic") for p in panels
            ):
                verdict += ", panel fallback (synthetic)"

        if path == "/v1/pipeline" and "clean" in data:
            verdict += f", clean={data['clean']}"
        if data.get("needs_review"):
            verdict += " — needs review"
        return f"RESULT: {verdict}"

    if path == "/v1/clean":
        clean = data.get("clean", "?")
        editors = data.get("editor_regions")
        n_ed = len(editors) if isinstance(editors, list) else 0
        if n_ed:
            return f"RESULT: clean={clean}, {n_ed} region(s) skipped (editor)"
        return f"RESULT: clean={clean}"

    if path == "/v1/typeset":
        layouts = data.get("layouts")
        n = len(layouts) if isinstance(layouts, list) else 0
        ok = (
            sum(
                1
                for L in layouts
                if isinstance(L, Mapping) and L.get("status") == "ok"
            )
            if isinstance(layouts, list)
            else 0
        )
        return f"RESULT: typeset={data.get('typeset', '?')} — {ok}/{n} layouts ok"

    if path == "/health":
        return f"RESULT: status={data.get('status', '?')}"

    if "detail" in data:
        detail = data["detail"]
        text = detail if isinstance(detail, str) else str(detail)
        return f"RESULT: {text[:160]}"

    return "RESULT: ok"


def format_json_dump(
    data: Any, *, path: str = "", status_code: int = 200
) -> list[str]:
    """Multi-line human dump of vision JSON (skips base64 blobs)."""
    if isinstance(data, list):
        if data and isinstance(data[0], Mapping) and "key" in data[0]:
            keys = [str(x.get("key")) for x in data[:15] if isinstance(x, Mapping)]
            extra = f" … +{len(data) - 15}" if len(data) > 15 else ""
            return [
                f"RESULT: {len(data)} fonts in catalog",
                f"  fonts: {', '.join(keys)}{extra}",
            ]
        return [f"RESULT: {len(data)} items"]

    if not isinstance(data, Mapping):
        return [f"  {type(data).__name__}"]

    lines: list[str] = [_result_line(path, data, status_code)]

    head: list[str] = []
    if data.get("page_id") is not None:
        head.append(f"page={data['page_id']}")
    if "width" in data and "height" in data:
        head.append(f"size={data['width']}x{data['height']}")
    if "order_confidence" in data:
        try:
            head.append(f"order_conf={float(data['order_confidence']):.3f}")
        except (TypeError, ValueError):
            pass
    if "device" in data:
        head.append(f"device={data['device']}")
    if head:
        lines.append("  " + "  ".join(head))

    models = data.get("models")
    if isinstance(models, Mapping):
        ready = ", ".join(f"{k}={'on' if v else 'off'}" for k, v in models.items())
        lines.append(f"  models: {ready}")

    bubbles = data.get("bubbles")
    if isinstance(bubbles, list):
        blines = [_fmt_bubble(b) for b in bubbles if isinstance(b, Mapping)]
        lines.extend(_list_block("bubbles", blines, len(bubbles), _BUBBLE_LIMIT))

    panels = data.get("panels")
    if isinstance(panels, list):
        plines = [_fmt_panel(p) for p in panels if isinstance(p, Mapping)]
        lines.extend(_list_block("panels", plines, len(panels), _PANEL_LIMIT))

    editors = data.get("editor_regions")
    if isinstance(editors, list) and editors:
        shown = []
        for e in editors[:8]:
            if isinstance(e, (list, tuple)) and len(e) >= 4:
                shown.append(f"[{int(e[0])},{int(e[1])},{int(e[2])},{int(e[3])}]")
        extra = f" … +{len(editors) - 8}" if len(editors) > 8 else ""
        lines.append(f"  editor_regions({len(editors)}): {', '.join(shown)}{extra}")

    layouts = data.get("layouts")
    if isinstance(layouts, list):
        llines = [_fmt_layout(L) for L in layouts if isinstance(L, Mapping)]
        lines.extend(_list_block("layouts", llines, len(layouts), _LAYOUT_LIMIT))

    for key in ("clean_layers", "text_layers"):
        layers = data.get(key)
        if isinstance(layers, list) and layers:
            lines.append(f"  {key}={len(layers)}")

    return lines


def format_body_dump(
    *,
    path: str,
    status_code: int,
    headers: Mapping[str, str],
    body: bytes,
) -> Optional[list[str]]:
    ctype = (headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype == "application/json":
        if not body:
            return ["RESULT: empty json"]
        try:
            return format_json_dump(
                json.loads(body), path=path, status_code=status_code
            )
        except Exception:
            return [f"RESULT: json {len(body)}B (parse failed)"]

    if ctype.startswith("image/"):
        parts: list[str] = []
        if body:
            parts.append(
                f"{len(body) // 1024}KB" if len(body) >= 1024 else f"{len(body)}B"
            )
        clean = headers.get("x-clean")
        if clean:
            parts.append(f"clean={clean}")
        editors = headers.get("x-editor-regions")
        n_ed = 0
        if editors:
            try:
                n_ed = len(json.loads(editors))
                parts.append(f"editor_regions={n_ed}")
            except Exception:
                pass
        if path == "/v1/clean":
            if clean == "ok" and not n_ed:
                result = "RESULT: clean fill ok"
            elif clean:
                result = f"RESULT: clean={clean}"
                if n_ed:
                    result += f", {n_ed} region(s) skipped (editor)"
            else:
                result = "RESULT: PNG returned"
        elif path == "/v1/compose":
            result = "RESULT: compose PNG ready"
        else:
            result = "RESULT: image returned"
        detail = "  " + "  ".join(parts) if parts else None
        return [result] + ([detail] if detail else [])

    if ctype.startswith("font/") or ctype in (
        "application/octet-stream",
        "application/font-sfnt",
        "application/x-font-ttf",
    ):
        return [
            f"RESULT: font file ({len(body)}B)" if body else "RESULT: font file"
        ]

    return None


def _header_map(raw_headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw_headers:
        out[k.decode("latin-1").lower()] = v.decode("latin-1")
    return out


class ResponseSummaryMiddleware:
    """Pure ASGI middleware — dumps labeled, separated request results."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        method = scope.get("method") or "?"
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            await self.app(scope, receive, send)
            return

        status_code = 0
        raw_headers: list[tuple[bytes, bytes]] = []
        chunks: list[bytes] = []

        async def send_wrapper(message: Any) -> None:
            nonlocal status_code, raw_headers
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                raw_headers = list(message.get("headers") or [])
                await send(message)
                return
            if message["type"] == "http.response.body":
                chunks.append(message.get("body") or b"")
                if not message.get("more_body", False):
                    body = b"".join(chunks)
                    headers = _header_map(raw_headers)
                    title = _op_title(path)
                    dump = format_body_dump(
                        path=path,
                        status_code=status_code,
                        headers=headers,
                        body=body,
                    )
                    lines = [
                        _RULE,
                        f" {title}",
                        _RULE,
                        f"{method} {path} → {status_code}",
                    ]
                    if dump:
                        lines.extend(dump)
                    lines.append(_RULE)
                    log.info("\n".join(lines))
                await send(message)
                return
            await send(message)

        await self.app(scope, receive, send_wrapper)
