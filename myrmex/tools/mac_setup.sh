#!/usr/bin/env bash
# Myrmex one-time setup on macOS: Python env for the live engine, Blender add-on, Ableton Remote Script.
# Safe to re-run.  Usage:  bash tools/mac_setup.sh [--blender 5.2]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER_VER="5.2"
if [[ "${1:-}" == "--blender" && -n "${2:-}" ]]; then BLENDER_VER="$2"; fi
VENV="$HOME/.venvs/myrmex"

echo "== Myrmex setup ($REPO)"

# 1. uv + Python 3.12 venv (python-rtmidi has no macOS wheels for 3.13 yet)
if ! command -v uv >/dev/null 2>&1; then
  echo "-- installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "-- creating $VENV (Python 3.12)"
  uv venv --python 3.12 "$VENV"
fi
echo "-- installing myrmex[live,app] into $VENV"
uv pip install --python "$VENV/bin/python" -e "$REPO[live,app]"

# 1b. Myrmex.app (a thin bundle around the venv) in ~/Applications + a shortcut on the Desktop
APP="$HOME/Applications/Myrmex.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Myrmex</string>
  <key>CFBundleDisplayName</key><string>Myrmex</string>
  <key>CFBundleIdentifier</key><string>app.myrmex.live</string>
  <key>CFBundleVersion</key><string>0.3</string>
  <key>CFBundleShortVersionString</key><string>0.3</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Myrmex</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>Myrmex listens to the music input to animate the character.</string>
</dict></plist>
PLIST
cat > "$APP/Contents/MacOS/Myrmex" <<LAUNCH
#!/bin/bash
exec "$VENV/bin/python" -m myrmex.app "\$@"
LAUNCH
chmod +x "$APP/Contents/MacOS/Myrmex"
ln -sfn "$APP" "$HOME/Desktop/Myrmex.app" 2>/dev/null || true
echo "-- app: $APP (shortcut on the Desktop)"

# 2. Blender add-on (symlink, so updates of the repo are picked up)
ADDONS="$HOME/Library/Application Support/Blender/$BLENDER_VER/scripts/addons"
mkdir -p "$ADDONS"
if [[ -e "$ADDONS/myrmex_blender" && ! -L "$ADDONS/myrmex_blender" ]]; then
  echo "!! $ADDONS/myrmex_blender exists and is not a symlink - leaving it alone"
else
  ln -sfn "$REPO/blender/myrmex_blender" "$ADDONS/myrmex_blender"
  echo "-- Blender add-on linked: $ADDONS/myrmex_blender  (enable 'Myrmex' in Blender > Settings > Add-ons)"
fi

# 3. Ableton Remote Script (copied: Live does not follow symlinks reliably)
RS="$HOME/Music/Ableton/User Library/Remote Scripts"
if [[ -d "$HOME/Music/Ableton" ]]; then
  mkdir -p "$RS"
  rm -rf "$RS/Myrmex"
  cp -R "$REPO/ableton/remote_script/Myrmex" "$RS/Myrmex"
  echo "-- Remote Script installed: $RS/Myrmex  (restart Live, Settings > Link, Tempo & MIDI > Control Surface: Myrmex)"
else
  echo "-- Ableton User Library not found (~/Music/Ableton); copy ableton/remote_script/Myrmex there manually"
fi

# 4. Check what the engine can see
echo "-- checking ports"
"$VENV/bin/myrmex" ports || true

cat <<MSG

Done.  Start Myrmex.app (Desktop / ~/Applications):
  Live tab: the engine runs; "Open character in Blender" shows her live.
  Ableton: LINK on (+ Start Stop Sync) or Control Surface: Myrmex -> Play.
Guide: $REPO/docs/REALTIME.md
MSG
