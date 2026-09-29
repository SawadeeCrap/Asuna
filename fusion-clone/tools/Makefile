# Standalone tools (no Rack needed): CLI renderer, benchmark, engine tests.
#   make            build everything with the portable built-in FFT
#   make PFFFT=/path/to/pffft   build against pffft (faster; sources: pffft.c pffft_common.c ... from Rack's dep/ or github.com/marton78/pffft)
CXX ?= g++
CXXFLAGS ?= -std=c++11 -O3 -march=native -Wall -Wextra -Wno-unused-parameter
ifdef PFFFT
  PFFFT_INC := $(firstword $(wildcard $(PFFFT)/include/pffft $(PFFFT)))
  CXXFLAGS += -DFC_FFT_PFFFT -I$(PFFFT_INC)
  PFFFT_OBJS := pffft.o pffft_common.o
endif
all: fusionclone_cli bench
pffft.o: ; $(CC) -O3 -c -msse2 $(PFFFT)/src/pffft.c -I$(PFFFT_INC) -I$(PFFFT)/src -o $@
pffft_common.o: ; $(CC) -O3 -c $(PFFFT)/src/pffft_common.c -I$(PFFFT_INC) -I$(PFFFT)/src -o $@
fusionclone_cli: fusionclone_cli.cpp $(PFFFT_OBJS) ; $(CXX) $(CXXFLAGS) $^ -o $@
bench: bench.cpp $(PFFFT_OBJS) ; $(CXX) $(CXXFLAGS) $^ -o $@
clean: ; rm -f fusionclone_cli bench pffft.o pffft_common.o
