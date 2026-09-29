# Standalone tools (no Rack needed): CLI renderer, benchmark, engine tests.
#   make            build everything with the portable built-in FFT
#   make PFFFT=/path/to/pffft   build against pffft (faster; sources: pffft.c pffft_common.c ... from Rack's dep/ or github.com/marton78/pffft)
CXX ?= g++
# -march=native and -msse2 are x86 options; Apple silicon / aarch64 builds use the compiler's default target (NEON is always available)
ifneq ($(filter arm64 aarch64,$(shell uname -m)),)
  ARCH_CXX :=
  ARCH_CC :=
else
  ARCH_CXX := -march=native
  ARCH_CC := -msse2
endif
CXXFLAGS ?= -std=c++11 -O3 $(ARCH_CXX) -Wall -Wextra -Wno-unused-parameter
ifdef PFFFT
  PFFFT_INC := $(firstword $(wildcard $(PFFFT)/include/pffft $(PFFFT)))
  CXXFLAGS += -DFC_FFT_PFFFT -I$(PFFFT_INC)
  PFFFT_OBJS := pffft.o pffft_common.o
endif
all: fusionclone_cli bench
pffft.o: ; $(CC) -O3 -c $(ARCH_CC) $(PFFFT)/src/pffft.c -I$(PFFFT_INC) -I$(PFFFT)/src -o $@
pffft_common.o: ; $(CC) -O3 -c $(PFFFT)/src/pffft_common.c -I$(PFFFT_INC) -I$(PFFFT)/src -o $@
HDRS := $(wildcard ../src/dsp/*.hpp ../research/common/*.hpp)
fusionclone_cli: fusionclone_cli.cpp $(HDRS) $(PFFFT_OBJS) ; $(CXX) $(CXXFLAGS) fusionclone_cli.cpp $(PFFFT_OBJS) -o $@
bench: bench.cpp $(HDRS) $(PFFFT_OBJS) ; $(CXX) $(CXXFLAGS) bench.cpp $(PFFFT_OBJS) -o $@
clean: ; rm -f fusionclone_cli bench pffft.o pffft_common.o
