#!/bin/bash
# Double-click me (macOS): installs everything on first run, then starts the Myrmex app.
cd "$(dirname "$0")"
REPO="$(pwd)"
VENV="$HOME/.venvs/myrmex"
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import PySide6, myrmex" 2>/dev/null; then
  echo "First run: installing Myrmex (one time, a few minutes)..."
  bash "$REPO/tools/mac_setup.sh" || { echo "setup failed"; read -r -p "Enter to close"; exit 1; }
fi
exec "$VENV/bin/python" -m myrmex.app
