#!/bin/bash
# Compile-check the plugin sources against a Rack source checkout (headers only, no linking). Useful in CI or when no full SDK is around.
#   tools/rack_compile_check.sh /path/to/Rack /path/to/sdk-dep-include [/path/to/pffft/include]
# The second argument must contain the third-party headers of Rack's dep/include (jansson.h, nanovg.h, GLFW/glfw3.h, pffft.h, ...).
set -e
RACK=${1:?path to Rack source or SDK}
DEP=${2:-$RACK/dep/include}
HERE=$(cd "$(dirname "$0")/.." && pwd)
for f in "$HERE"/src/*.cpp; do
	echo "syntax-check $f"
	g++ -std=c++11 -fsyntax-only -Wall -Wextra -Wno-unused-parameter -march=nehalem -DGLFW_INCLUDE_NONE -DFC_FFT_PFFFT \
		-I"$RACK/include" -I"$DEP" -I"$HERE/src" "$f"
done
echo "OK"
