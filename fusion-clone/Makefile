# VCV Rack 2 plugin build. Usage:
#     make RACK_DIR=/path/to/Rack-SDK            # build plugin.so / plugin.dylib / plugin.dll
#     make RACK_DIR=/path/to/Rack-SDK install    # copy into your Rack user plugins folder
#     make RACK_DIR=/path/to/Rack-SDK dist       # build a distributable .vcvplugin package
RACK_DIR ?= ../..

# The DSP core is framework independent and is compiled with pffft (bundled with Rack, exported by libRack) as its FFT backend.
# Rack's build system uses -std=c++11, so the whole core is written in C++11.
FLAGS += -Isrc -DFC_FFT_PFFFT
CFLAGS +=
CXXFLAGS +=
LDFLAGS +=

SOURCES += src/plugin.cpp src/FusionClone.cpp

DISTRIBUTABLES += res
DISTRIBUTABLES += $(wildcard LICENSE*)
DISTRIBUTABLES += README.md

include $(RACK_DIR)/plugin.mk
