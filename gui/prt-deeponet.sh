#!/usr/bin/env bash
# PRT-DeepONet Studio launcher for Linux and macOS.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
for c in "$ROOT/3D/.venv/bin/python" "$ROOT/.venv/bin/python" python3 python; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY="$c"; break; fi
done
echo "Starting PRT-DeepONet Studio with $PY"
exec "$PY" "$HERE/prt_gui.py"
