# VCV Rack 2 plugin build. Usage:
#     make RACK_DIR=/path/to/Rack-SDK            # build plugin.so / plugin.dylib / plugin.dll
#     make RACK_DIR=/path/to/Rack-SDK install    # copy into your Rack user plugins folder
#     make RACK_DIR=/path/to/Rack-SDK dist       # build a distributable .vcvplugin package
RACK_DIR ?= ../..

# The DSP core is framework independent and is compiled with pffft (bundled with Rack, exported by libRack) as its FFT backend.
# Rack's build system uses -std=c++11, so the whole core is written in C++11.
# `-iquote src` (not `-Isrc`): src/ must only be searched for "quoted" includes. With -Isrc, Rack's own <dsp/...> headers would be looked up in
# our src/dsp/ first, and on a case-insensitive file system (macOS, Windows) <dsp/common.hpp> would silently resolve to a file of ours whose
# name differs only in case. (For the same reason our headers are called Fc*.hpp.)
FLAGS += -iquote src -DFC_FFT_PFFFT
CFLAGS +=
CXXFLAGS +=
LDFLAGS +=

SOURCES += src/plugin.cpp src/FusionClone.cpp

DISTRIBUTABLES += res
DISTRIBUTABLES += $(wildcard LICENSE*)
DISTRIBUTABLES += README.md

include $(RACK_DIR)/plugin.mk
