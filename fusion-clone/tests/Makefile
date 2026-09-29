# DSP unit / integration tests. They need nothing but a C++11 compiler (built-in FFT) — no Rack.
#
#   make              build and run every test
#   make quick        only the fast component tests (FFT, sinc, Hilbert, pitch tracker, cycle analyser)
#   make engine       only the engine test suite (a few minutes)
#   make lock         lock-robustness matrix (waveform x pitch); `./test_lock full` covers all four quality modes
#   make dynamics     vibrato, SUB sweep, WAVE switch, note gap
#   make realtime     allocation-free / finite / per-sample cost check of the audio path
#   make PFFFT=/path/to/pffft   use pffft instead of the built-in FFT (what the plugin uses)
#   make clean
#
# The headless Rack module test (test_module.cpp) is built separately by build_module_test.sh because it needs a Rack source checkout.
CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -march=native -Wall -Wextra -Wno-unused-parameter
ifdef PFFFT
  PFFFT_INC := $(firstword $(wildcard $(PFFFT)/include/pffft $(PFFFT)))
  CXXFLAGS += -DFC_FFT_PFFFT -I$(PFFFT_INC)
  PFFFT_OBJS := pffft.o pffft_common.o
endif

QUICK := test_fft test_sinc test_hilbert test_pitch test_analyzer
ALL := $(QUICK) test_lock test_dynamics test_artifacts test_engine test_realtime

.PHONY: all quick engine lock dynamics artifacts realtime clean
.SECONDARY:
all: $(ALL:%=run-%)
quick: $(QUICK:%=run-%)
engine: run-test_engine
lock: run-test_lock
dynamics: run-test_dynamics
artifacts: run-test_artifacts
realtime: run-test_realtime

run-%: %
	@echo "=== $<"
	@./$<

pffft.o: ; $(CC) -O3 -c -msse2 $(PFFFT)/src/pffft.c -I$(PFFFT_INC) -I$(PFFFT)/src -o $@
pffft_common.o: ; $(CC) -O3 -c $(PFFFT)/src/pffft_common.c -I$(PFFFT_INC) -I$(PFFFT)/src -o $@

DSP := $(wildcard ../src/dsp/*.hpp)
RESEARCH := $(wildcard ../research/common/*.hpp)
test_%: test_%.cpp $(DSP) $(RESEARCH) $(PFFFT_OBJS) ; $(CXX) $(CXXFLAGS) test_$*.cpp $(PFFFT_OBJS) -o $@

clean: ; rm -f $(ALL) pffft.o pffft_common.o
