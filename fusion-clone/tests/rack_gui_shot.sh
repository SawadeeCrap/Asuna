#!/bin/bash
# Starts a real Rack WITH its window and the same patch as rack_smoke.sh (Fundamental VCO -> Fusion Clone), waits, and takes a screenshot:
#
#   [RACK_SHOT_OUT=dir] tests/rack_gui_shot.sh PACKAGE.vcvplugin /path/to/Rack [SECONDS] [ZOOM]
#
# Linux: a virtual X display (Xvfb, Mesa software OpenGL) is started when DISPLAY is not set. macOS: the desktop of the runner.
# What it exercises that the headless test cannot: the drawing code of the panel (display, spectrum bars, labels) and everything a window needs.
# PASS = Rack is still alive after SECONDS, the module was created and its log has no warning/error mentioning Fusion Clone. With RACK_SHOT_OUT
# set, the full-size screenshot is copied to that directory as rack-<platform>-full.png.
set -u
PKG=${1:?usage: rack_gui_shot.sh PACKAGE.vcvplugin /path/to/Rack [SECONDS] [ZOOM]}
RACK=${2:?usage: rack_gui_shot.sh PACKAGE.vcvplugin /path/to/Rack [SECONDS] [ZOOM]}
SECS=${3:-15}
ZOOM=${4:-1.0}     # Rack's zoom setting is log2 of the zoom factor: 0 = 100 %, 1 = 200 %
PKG=$(cd "$(dirname "$PKG")" && pwd)/$(basename "$PKG")
RACK=$(cd "$(dirname "$RACK")" && pwd)/$(basename "$RACK")

case "$(uname -s)-$(uname -m)" in
	Darwin-arm64) ARCH=mac-arm64 ;;
	Darwin-x86_64) ARCH=mac-x64 ;;
	Linux-x86_64) ARCH=lin-x64 ;;
	*) echo "unsupported platform $(uname -s)-$(uname -m)"; exit 2 ;;
esac

WORK=$(mktemp -d)
USER_DIR=$WORK/user
mkdir -p "$USER_DIR/plugins-$ARCH" "$USER_DIR/autosave" "$WORK/patch"
cp "$PKG" "$USER_DIR/plugins-$ARCH/"
cat > "$WORK/patch/patch.json" <<EOF
{
  "version": "2.0.0",
  "gridOffset": [0.0, 0.0],
  "modules": [
    {"id": 1, "plugin": "Fundamental", "model": "VCO", "params": [], "pos": [0, 0]},
    {"id": 2, "plugin": "FusionClone", "model": "FusionClone", "pos": [12, 0],
     "params": [{"id": 0, "value": 16.0}, {"id": 9, "value": 3.0}, {"id": 10, "value": 1.0}]}
  ],
  "cables": [
    {"id": 1, "outputModuleId": 1, "outputId": 2, "inputModuleId": 2, "inputId": 0}
  ]
}
EOF
cp "$WORK/patch/patch.json" "$USER_DIR/autosave/patch.json"
(cd "$WORK/patch" && tar -c patch.json | zstd -q -19 -o "$WORK/shot.vcv") || { echo "could not create the patch archive (need tar and zstd)"; exit 2; }
# fill the (virtual) screen with the Rack window
# (zoom is Rack's log2 zoom, a setting of Rack, not of the patch; the start-up tip dialog would cover part of a zoomed-in module)
echo '{"windowSize": [1600, 1000], "windowPos": [0, 0], "zoom": '"$ZOOM"', "showTipsOnLaunch": false}' > "$USER_DIR/settings.json"

XVFB=
if [ "$(uname -s)" = Linux ] && [ -z "${DISPLAY:-}" ]; then
	command -v Xvfb >/dev/null 2>&1 || { echo "Xvfb is needed"; exit 2; }
	Xvfb :99 -screen 0 1600x1000x24 +extension GLX +render -noreset >/dev/null 2>&1 &
	XVFB=$!
	export DISPLAY=:99
	sleep 2
fi

echo "== Rack: $RACK"
echo "== starting it with a window for $SECS s (user folder $USER_DIR, DISPLAY=${DISPLAY:-none})"
cd "$(dirname "$RACK")" || exit 2
"$RACK" -u "$USER_DIR" "$WORK/shot.vcv" > "$WORK/rack.out" 2>&1 &
PID=$!
CRASHED=0
for ((i = 0; i < SECS; i++)); do
	sleep 1
	if ! kill -0 "$PID" 2>/dev/null; then CRASHED=1; break; fi
done

SHOT=$WORK/full.png
if [ "$CRASHED" = 0 ]; then
	echo "Rack process after $SECS s:"; ps -o pid=,cputime=,etime=,rss= -p "$PID" || true
	if [ "$(uname -s)" = Darwin ]; then
		screencapture -x "$SHOT" || echo "screencapture failed"
		sips -g pixelWidth -g pixelHeight "$SHOT" 2>/dev/null | tail -2
		# keep the file small: never wider than 1600 px
		w=$(sips -g pixelWidth "$SHOT" 2>/dev/null | awk '/pixelWidth/{print $2}')
		if [ -n "$w" ] && [ "$w" -gt 1600 ]; then sips --resampleWidth 1600 "$SHOT" --out "$SHOT.small.png" >/dev/null 2>&1 && mv "$SHOT.small.png" "$SHOT"; fi
	else
		if command -v import >/dev/null 2>&1; then import -display "$DISPLAY" -window root "$SHOT" || echo "import failed"; else echo "ImageMagick 'import' is missing"; fi
	fi
	kill -TERM "$PID" 2>/dev/null
	for ((i = 0; i < 10; i++)); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
	kill -0 "$PID" 2>/dev/null && kill -KILL "$PID"
	wait "$PID" 2>/dev/null
else
	wait "$PID"; echo "Rack exited by itself after $i s with status $?"
fi
[ -n "$XVFB" ] && kill "$XVFB" 2>/dev/null

echo; echo "================ Rack stdout/stderr (last 40 lines)"; tail -40 "$WORK/rack.out"
echo; echo "================ log.txt (last 80 lines)"; tail -80 "$USER_DIR/log.txt" 2>/dev/null || echo "(no log.txt)"
echo; echo "================ lines of log.txt about Fusion Clone, warnings and errors"
grep -iE "fusion ?clone|\[(warn|fatal)" "$USER_DIR/log.txt" 2>/dev/null || echo "(none)"

if [ -s "$SHOT" ]; then
	echo; echo "screenshot: $(ls -l "$SHOT" | awk '{print $5}') bytes"
	if [ -n "${RACK_SHOT_OUT:-}" ]; then
		mkdir -p "$RACK_SHOT_OUT" && cp "$SHOT" "$RACK_SHOT_OUT/rack-$ARCH-full.png" && echo "copied to $RACK_SHOT_OUT/rack-$ARCH-full.png"
	fi
else
	echo; echo "no screenshot was taken"
fi

FAIL=0
if [ "$CRASHED" = 1 ]; then echo "  FAIL  Rack kept running for $SECS s"; FAIL=1; else echo "  ok    Rack kept running for $SECS s with its window"; fi
if grep -iE "fusion ?clone" "$USER_DIR/log.txt" 2>/dev/null | grep -qE "\[(warn|fatal)"; then echo "  FAIL  warnings/errors mention Fusion Clone"; FAIL=1; else echo "  ok    no warnings or errors mention Fusion Clone"; fi
if grep -q "Fusion Clone: module added" "$USER_DIR/log.txt" 2>/dev/null; then echo "  ok    the module was created"; else echo "  FAIL  the module was not created"; FAIL=1; fi
[ "$FAIL" = 0 ] && echo "RACK GUI TEST: PASS" || echo "RACK GUI TEST: FAIL"
rm -rf "$WORK"
exit "$FAIL"
