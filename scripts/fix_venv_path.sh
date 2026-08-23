#!/usr/bin/env bash
# One-time fix when the repo moved (benim/ → products/) and `source .venv/bin/activate`
# points at a dead path. Does NOT reinstall packages.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OLD="/Users/talo/Developer/benim/fukiflow-vision/.venv"
NEW="$ROOT/.venv"

if [[ -d .venv.broken-20260823 ]]; then
  echo "Restoring full .venv from .venv.broken-20260823 …"
  rm -rf .venv
  mv .venv.broken-20260823 .venv
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv/bin/python — create a venv first or restore backup." >&2
  exit 1
fi

echo "Patching paths: $OLD → $NEW"
for f in .venv/pyvenv.cfg .venv/bin/activate .venv/bin/activate.csh .venv/bin/activate.fish; do
  if [[ -f "$f" ]]; then
    sed -i '' "s|${OLD}|${NEW}|g" "$f"
  fi
done

while IFS= read -r f; do
  sed -i '' "s|${OLD}|${NEW}|g" "$f"
done < <(grep -rl "$OLD" .venv/bin 2>/dev/null || true)

echo "OK. Test:"
echo "  source .venv/bin/activate"
echo "  modal --version"
