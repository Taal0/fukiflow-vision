#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-http://127.0.0.1:8001}"
IMG="${1:-}"

if [[ -z "$IMG" ]]; then
  echo "Usage: $0 path/to/page.png" >&2
  exit 1
fi

OUT="$ROOT/fixtures/out"
mkdir -p "$OUT"

echo "== health =="
curl -sf "$HOST/health" | tee "$OUT/health.json"
echo

echo "== bubbles =="
curl -sf -X POST "$HOST/v1/bubbles" \
  -F "file=@${IMG}" \
  -o "$OUT/bubbles.json"
python3 - <<'PY' "$OUT/bubbles.json"
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
print(f"page_id={data['page_id']} bubbles={len(data['bubbles'])} {data['width']}x{data['height']}")
for i, b in enumerate(data["bubbles"][:8]):
    print(f"  [{i}] conf={b['confidence']:.3f} bbox={b['bbox']} cls={b.get('cls')}")
if len(data["bubbles"]) > 8:
    print(f"  ... +{len(data['bubbles'])-8} more")
PY

echo "== ocr =="
python3 - <<'PY' "$OUT/bubbles.json" "$OUT/ocr_bubbles.json"
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps(data["bubbles"]), encoding="utf-8")
print(f"wrote {sys.argv[2]} ({len(data['bubbles'])} bubbles)")
PY
curl -sf -X POST "$HOST/v1/ocr" \
  -F "file=@${IMG}" \
  -F "bubbles_json=$(cat "$OUT/ocr_bubbles.json")" \
  -o "$OUT/ocr.json"
python3 - <<'PY' "$OUT/ocr.json"
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
print(f"bubbles={len(data['bubbles'])}")
for i, b in enumerate(data["bubbles"][:8]):
    print(f"  [{i}] text={b.get('text_ja')!r}")
if len(data["bubbles"]) > 8:
    print(f"  ... +{len(data['bubbles'])-8} more")
PY

echo "== pipeline =="
curl -sf -X POST "$HOST/v1/pipeline" \
  -F "file=@${IMG}" \
  -o "$OUT/pipeline.json"
python3 - <<'PY' "$OUT/pipeline.json" "$OUT/cleaned.png" "$OUT/detect.png"
import json, base64, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_bytes(base64.b64decode(data["cleaned_image"]))
if data.get("bubbles_overlay"):
    Path(sys.argv[3]).write_bytes(base64.b64decode(data["bubbles_overlay"]))
print(
    f"bubbles={len(data['bubbles'])} clean={data.get('clean')} "
    f"→ {sys.argv[2]} detect→{sys.argv[3]}"
)
PY
echo "OK → $OUT/"
