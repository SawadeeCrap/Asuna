#!/bin/bash
# Double-click me (macOS): turn a Hunyuan3D GLB into a live-ready character.
cd "$(dirname "$0")"
REPO="$(pwd)"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
if [[ ! -x "$BLENDER" ]]; then
  APP="$(mdfind "kMDItemCFBundleIdentifier == 'org.blenderfoundation.blender'" | head -1)"
  BLENDER="$APP/Contents/MacOS/Blender"
fi
GLB="$1"
if [[ -z "$GLB" ]]; then
  echo "Drag the .glb file into this window and press Enter:"
  read -r GLB
  GLB="${GLB%\"}"; GLB="${GLB#\"}"; GLB="${GLB%\'}"; GLB="${GLB#\'}"; GLB="${GLB//\\ / }"
fi
NAME="$(basename "${GLB%.*}")"
OUT="$HOME/Myrmex/${NAME}_live.blend"
mkdir -p "$HOME/Myrmex"
read -r -p "Height in metres [1.70]: " H; H="${H:-1.70}"
"$BLENDER" -b --python "$REPO/blender/scripts/prepare_character.py" -- --glb "$GLB" --out "$OUT" --height "$H"
echo
echo "Ready: $OUT"
echo "Start it:  \"$REPO/Myrmex.command\" \"$OUT\""
read -r -p "Enter to close"
