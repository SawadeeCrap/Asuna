#!/bin/bash
# End-to-end smoke test inside a REAL Rack: installs a built Fusion Clone package into a scratch user folder, starts Rack without a window
# (headless, -h) on a patch that feeds Fusion Clone from a Fundamental VCO (when Rack ships Fundamental), lets it run for a few seconds, stops
# it with SIGINT and reads Rack's log.
#
#   tests/rack_smoke.sh PACKAGE.vcvplugin /path/to/Rack [SECONDS]      (default 20 s)
#
# PASS means: Rack loaded the plugin, created the module from the patch, kept running for SECONDS without crashing, and (when the VCO exists)
# the module reported that it locked onto the oscillator. The audio is not recorded: this proves loading, instantiation, and crash-freedom in
# the real engine, not sound quality. It needs `zstd` and `tar`. Needs a Rack whose major version matches the package (2.x).
set -u
PKG=${1:?usage: rack_smoke.sh PACKAGE.vcvplugin /path/to/Rack [SECONDS]}
RACK=${2:?usage: rack_smoke.sh PACKAGE.vcvplugin /path/to/Rack [SECONDS]}
SECS=${3:-20}
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

# VOICES = 16, QUALITY = ULTRA, ALGORITHM = FUSION: the heaviest configuration. Input: the saw output (id 2) of a Fundamental VCO at C4.
cat > "$WORK/patch/patch.json" <<'EOF'
{
  "version": "2.0.0",
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
(cd "$WORK/patch" && tar -c patch.json | zstd -q -19 -o "$WORK/smoke.vcv") || { echo "could not create the patch archive (need tar and zstd)"; exit 2; }

XVFB=
if [ "$(uname -s)" = Linux ] && [ -z "${DISPLAY:-}" ] && command -v Xvfb >/dev/null 2>&1; then
	Xvfb :99 -screen 0 1280x800x24 >/dev/null 2>&1 &
	XVFB=$!
	export DISPLAY=:99
	sleep 1
fi

echo "== Rack: $RACK"
echo "== running headless for $SECS s (user folder $USER_DIR)"
cd "$(dirname "$RACK")" || exit 2   # on Linux Rack finds its res/ folder relative to the working directory
"$RACK" -h -u "$USER_DIR" "$WORK/smoke.vcv" > "$WORK/rack.out" 2>&1 &
PID=$!
CRASHED=0
for ((i = 0; i < SECS; i++)); do
	sleep 1
	if ! kill -0 "$PID" 2>/dev/null; then CRASHED=1; break; fi
done
if [ "$CRASHED" = 1 ]; then
	wait "$PID"; STATUS=$?
	echo "Rack exited by itself after $i s with status $STATUS"
else
	kill -INT "$PID" 2>/dev/null
	for ((i = 0; i < 15; i++)); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
	if kill -0 "$PID" 2>/dev/null; then echo "Rack did not stop on SIGINT, killing it"; kill -KILL "$PID"; fi
	wait "$PID" 2>/dev/null; STATUS=$?
	echo "Rack still running after $SECS s; stopped with SIGINT (status $STATUS)"
fi
[ -n "$XVFB" ] && kill "$XVFB" 2>/dev/null

LOG=$USER_DIR/log.txt
echo; echo "================ Rack stdout/stderr"; cat "$WORK/rack.out"
echo; echo "================ $LOG"; cat "$LOG" 2>/dev/null || echo "(no log.txt)"
echo; echo "================ lines about Fusion Clone"
cat "$LOG" "$WORK/rack.out" 2>/dev/null | grep -iE "fusion ?clone" || echo "(none)"
echo

FAIL=0
ok()   { echo "  ok    $1"; }
bad()  { echo "  FAIL  $1"; FAIL=1; }
note() { echo "  note  $1"; }
have() { cat "$LOG" "$WORK/rack.out" 2>/dev/null | grep -qE "$1"; }

if [ "$CRASHED" = 1 ]; then bad "Rack kept running for $SECS s (it exited by itself with status $STATUS)"; else ok "Rack kept running for $SECS s without crashing"; fi
if have "Fusion Clone: module added"; then ok "the plugin was loaded and the module was created from the patch"; else bad "no 'Fusion Clone: module added' line: plugin not loaded or module not created"; fi
if cat "$LOG" 2>/dev/null | grep -iE "fusion ?clone" | grep -qE "\[(warn|fatal)"; then bad "warnings/errors mention Fusion Clone"; else ok "no warnings or errors mention Fusion Clone"; fi
if have "Fusion Clone: module removed"; then
	if have "module removed; last state LOCKED"; then ok "on shutdown the module reported that it was LOCKED onto the VCO";
	else bad "the module did not end up LOCKED onto the VCO (its last state is in the lines above; the patch needs the Fundamental plugin that ships with Rack)"; fi
	if have "safety-net hits 0"; then ok "the engine's NaN/Inf safety net never fired"; else bad "the engine's safety net fired (or its counter is missing)"; fi
else
	note "no 'module removed' line: this Rack does not call onRemove on exit, so the final engine state is unknown"
fi
echo
if [ "$FAIL" = 0 ]; then echo "RACK SMOKE TEST: PASS"; else echo "RACK SMOKE TEST: FAIL"; fi
rm -rf "$WORK"
exit "$FAIL"
