#!/bin/bash
# Headless test of the real FusionClone Module class against Rack's own engine classes (Module, ParamQuantity, Quantity, jansson).
# It instantiates the module without any GUI, checks the parameter / CV mapping, process() output, bypass, and patch save/load
# reproducibility (see test_module.cpp). Only the pieces of Rack that Module::toJson()/fromJson() touch are compiled; the application layer is
# replaced by rack_headless_stubs.cpp.
#
#   tests/build_module_test.sh RACK_SRC [DEP_SRC] [BUILD_DIR]
#
#   RACK_SRC   a checkout of https://github.com/VCVRack/Rack (v2) with its third-party headers installed in RACK_SRC/dep/include
#              (run `make dep` in the Rack checkout once; that also fetches the submodules).
#   DEP_SRC    where the third-party *sources* live (default RACK_SRC/dep): needs jansson/src, tinyexpr/tinyexpr.c, nanovg/src/nanovg.c.
#   DEPINC     (environment) where the third-party *headers* live (default RACK_SRC/dep/include, i.e. after `make dep`; an SDK download has them too).
#   BUILD_DIR  scratch directory for objects (default ./build-module-test).
#
# Uses the portable built-in FFT (no pffft needed). Exit status = test result.
set -e
RACK=${1:?path to a Rack v2 source checkout (with dep/include populated)}
DEPSRC=${2:-$RACK/dep}
BUILD=${3:-./build-module-test}
HERE=$(cd "$(dirname "$0")/.." && pwd)
DEPINC=${DEPINC:-$RACK/dep/include}
mkdir -p "$BUILD/obj"
CXX=${CXX:-g++}
CC=${CC:-gcc}
case "$(uname -s)" in Darwin) ARCHDEF=-DARCH_MAC ;; MINGW*|MSYS*) ARCHDEF=-DARCH_WIN ;; *) ARCHDEF=-DARCH_LIN ;; esac
FL="-std=c++11 -O1 -g -DGLFW_INCLUDE_NONE -DVERSION=\"2.0.0\" $ARCHDEF -I$RACK/include -I$DEPINC -iquote $HERE/src"

compile() { # compile <compiler> <flags> <src> <obj>
	if [ ! -f "$4" ] || [ "$3" -nt "$4" ]; then echo "  cc $(basename "$3")"; $1 $2 -c "$3" -o "$4"; fi
}

echo "== Rack engine sources"
for f in engine/LightInfo engine/Module engine/ParamQuantity engine/PortInfo Quantity common logger random string; do
	compile "$CXX" "$FL" "$RACK/src/$f.cpp" "$BUILD/obj/r_$(basename $f).o"
done

echo "== third-party sources (jansson, tinyexpr, nanovg)"
JCFG=$BUILD/jansson_private_config.h
if [ ! -f "$JCFG" ]; then
	cat > "$JCFG" <<'EOF'
#define HAVE_STDINT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1
#define HAVE_LOCALE_H 1
#define HAVE_SETLOCALE 1
#define HAVE_STRTOLL 1
#define HAVE_SNPRINTF 1
#define HAVE___BUILTIN_EXPECT 1
#define HAVE_ATOMIC_BUILTINS 1
#define HAVE_SYNC_BUILTINS 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_CLOCK_GETTIME 1
#define HAVE_SYS_TIME_H 1
#define HAVE_TIME_H 1
#define INITIAL_HASHTABLE_ORDER 3
#define USE_URANDOM 1
#define USE_WINDOWS_CRYPTOAPI 0
#define HAVE_ENDIAN_H 1
EOF
fi
for f in dtoa dump error hashtable hashtable_seed load memory pack_unpack strbuffer strconv utf value version; do
	compile "$CC" "-O1 -DHAVE_CONFIG_H -I$BUILD -I$DEPSRC/jansson/src -I$DEPINC" "$DEPSRC/jansson/src/$f.c" "$BUILD/obj/j_$f.o"
done
compile "$CC" "-O1 -I$DEPSRC/tinyexpr" "$DEPSRC/tinyexpr/tinyexpr.c" "$BUILD/obj/tinyexpr.o"
compile "$CC" "-O1 -DNANOVG_GL2 -I$DEPSRC/nanovg/src -I$DEPINC" "$DEPSRC/nanovg/src/nanovg.c" "$BUILD/obj/nanovg.o"

echo "== stubs and test"
compile "$CXX" "$FL" "$HERE/tests/rack_headless_stubs.cpp" "$BUILD/obj/stubs.o"
$CXX $FL -o "$BUILD/test_module" "$HERE/tests/test_module.cpp" "$BUILD"/obj/*.o -lpthread
"$BUILD/test_module"
