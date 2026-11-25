#!/bin/sh
# TIL Studio launcher for Linux/macOS: opens the IDE in your browser.
#   ./til-ide.sh [FOLDER]        (default: the current folder)
# Same as `til ide [FOLDER]`. Needs only Python 3.9+.
DIR="$(cd "$(dirname "$0")" && pwd)"
if command -v til >/dev/null 2>&1; then
    exec til ide "$@"
fi
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
    echo "TIL Studio needs Python 3.9+: https://www.python.org/downloads/" >&2
    exit 1
fi
exec "$PY" "$DIR/../src/til.py" ide "$@"
