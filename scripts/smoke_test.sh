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

echo "== detect-ocr =="
curl -sf -X POST "$HOST/v1/detect-ocr" \
  -F "file=@${IMG}" \
  -o "$OUT/detect.json"
python3 - <<'PY' "$OUT/detect.json" "$OUT/mask.png"
import json, base64, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text())
print(f"page_id={data['page_id']} boxes={len(data['boxes'])} {data['width']}x{data['height']}")
for i, b in enumerate(data["boxes"][:8]):
    print(f"  [{i}] conf={b['confidence']:.3f} bbox={b['bbox']} text={b['text_ja']!r}")
if len(data["boxes"]) > 8:
    print(f"  ... +{len(data['boxes'])-8} more")
Path(sys.argv[2]).write_bytes(base64.b64decode(data["mask"]))
print(f"wrote {sys.argv[2]}")
PY

echo "== inpaint =="
curl -sf -X POST "$HOST/v1/inpaint" \
  -F "file=@${IMG}" \
  -F "mask=@${OUT}/mask.png" \
  -o "$OUT/cleaned.png"
file "$OUT/cleaned.png"
echo "OK → $OUT/"
