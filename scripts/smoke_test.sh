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

echo "== pipeline =="
curl -sf -X POST "$HOST/v1/pipeline" \
  -F "file=@${IMG}" \
  -o "$OUT/pipeline.json"
python3 - <<'PY' "$OUT/pipeline.json" "$OUT/cleaned.png" "$OUT/detect.png" "$OUT/texts.json"
import json, base64, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_bytes(base64.b64decode(data["cleaned_image"]))
if data.get("bubbles_overlay"):
    Path(sys.argv[3]).write_bytes(base64.b64decode(data["bubbles_overlay"]))
# Fixture texts keyed by bubble id (sample TR lines, cycled).
samples = [
    "Ne oluyor burada?",
    "Hiçbir şey anlamıyorum.",
    "Bekle… Şunu bir daha söyle.",
    "Bu kadar hızlı olamaz!",
    "Gidelim. Zamanımız kalmadı.",
    "Sen gerçekten bunu mu düşünüyorsun?",
    "Özür dilerim. Yanıldım.",
    "Konuşamayız mı artık?",
]
texts = []
for i, b in enumerate(data.get("bubbles") or []):
    texts.append({"bubble_id": b["id"], "text": samples[i % len(samples)]})
Path(sys.argv[4]).write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
print(
    f"bubbles={len(data['bubbles'])} clean={data.get('clean')} "
    f"→ {sys.argv[2]} detect→{sys.argv[3]}"
)
PY

echo "== typeset =="
python3 - <<'PY' "$OUT/pipeline.json" "$OUT/bubbles_for_typeset.json"
import json, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(
    json.dumps(data.get("bubbles") or [], ensure_ascii=False),
    encoding="utf-8",
)
PY
curl -sf -X POST "$HOST/v1/typeset" \
  -F "file=@${OUT}/cleaned.png" \
  -F "bubbles_json=<${OUT}/bubbles_for_typeset.json" \
  -F "texts_json=<${OUT}/texts.json" \
  -F "include_debug=true" \
  -o "$OUT/typeset.json"
python3 - <<'PY' "$OUT/typeset.json" "$OUT/typeset_debug.png"
import json, base64, sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
if data.get("debug_image"):
    Path(sys.argv[2]).write_bytes(base64.b64decode(data["debug_image"]))
n_ok = sum(1 for L in data.get("layouts") or [] if L.get("status") == "ok")
n_ed = sum(1 for L in data.get("layouts") or [] if L.get("status") == "editor")
print(
    f"typeset={data.get('typeset')} ok={n_ok} editor={n_ed} "
    f"regions={len(data.get('typeset_regions') or [])} → {sys.argv[2]}"
)
PY
echo "OK → $OUT/"
