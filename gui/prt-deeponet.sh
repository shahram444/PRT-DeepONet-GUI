#!/usr/bin/env bash
# PRT-DeepONet Studio launcher for Linux and macOS.
#
# AUDIT GUILAUNCH-08. With "set -u" active and PY assigned only inside the
# loop, a machine with no Python at all died on an unset-variable error from
# bash rather than telling the user what was wrong. PY now starts empty and the
# failure is explained.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

PY=""
for c in "$ROOT/3D/.venv/bin/python" "$ROOT/.venv/bin/python" python3 python; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY="$c"; break; fi
done

if [ -z "$PY" ]; then
    echo "No Python interpreter was found." >&2
    echo >&2
    echo "Looked for, in this order:" >&2
    echo "   $ROOT/3D/.venv/bin/python" >&2
    echo "   $ROOT/.venv/bin/python" >&2
    echo "   python3 on the PATH" >&2
    echo "   python  on the PATH" >&2
    echo >&2
    echo "Install Python 3.10 or newer, or create a virtual environment at" >&2
    echo "$ROOT/.venv, then run this again." >&2
    exit 1
fi

echo "Starting PRT-DeepONet Studio with $PY"
exec "$PY" "$HERE/prt_gui.py"
