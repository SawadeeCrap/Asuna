# Prototype harness: the candidate architectures A-K, the synthetic source model and the experiments that led to the architecture
# (docs/ARCHITECTURE.md section 4). Everything is offline C++11; the production engine (../src/dsp) takes part as candidate J.
#
#   make                          build the experiments
#   make -jN results              run all of them and refresh results/*.txt (about 40 CPU-minutes; the sinusoidal-tracking and
#                                 heterodyne candidates are slow by design)
#   make PFFFT=/path/to/pffft     use pffft instead of the built-in FFT
#   make clean
CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -march=native -Wall -Wextra -Wno-unused-parameter
ifdef PFFFT
  PFFFT_INC := $(firstword $(wildcard $(PFFFT)/include/pffft $(PFFFT)))
  CXXFLAGS += -DFC_FFT_PFFFT -I$(PFFFT_INC)
  PFFFT_OBJS := pffft.o pffft_common.o
endif

EXPS := exp_sanity exp_compare exp_transient exp_texture exp_engine_checks exp_fft_study
HDRS := $(wildcard common/*.hpp candidates/*.hpp ../src/dsp/*.hpp)

all: $(EXPS)
pffft.o: ; $(CC) -O3 -c -msse2 $(PFFFT)/src/pffft.c -I$(PFFFT_INC) -I$(PFFFT)/src -o $@
pffft_common.o: ; $(CC) -O3 -c $(PFFFT)/src/pffft_common.c -I$(PFFFT_INC) -I$(PFFFT)/src -o $@
exp_%: exp_%.cpp $(HDRS) $(PFFFT_OBJS) ; $(CXX) $(CXXFLAGS) $< $(PFFFT_OBJS) -o $@

# one output file per scene of the architecture comparison (each is an independent job)
scene_saw110_N2      := saw 110 Hz N=2
scene_saw110_N16     := saw 110 Hz N=16
scene_saw41_N4       := saw 41.2 Hz N=4
scene_saw20_N4       := saw 20 Hz N=4
scene_saw440_N8      := saw 440 Hz N=8
scene_saw1760_N8     := saw 1760 Hz N=8
scene_sub110_N8      := saw+sub(.5) 110 Hz N=8
scene_doppler110_N8  := saw+Doppler detune .3, 110 Hz N=8
scene_pulse220_N8    := pulse10%+tube .4, 220 Hz N=8
SCENES := saw110_N2 saw110_N16 saw41_N4 saw20_N4 saw440_N8 saw1760_N8 sub110_N8 doppler110_N8 pulse220_N8

results/compare_%.txt: exp_compare
	@mkdir -p results
	./exp_compare "$(scene_$*)" > $@
results/sanity.txt: exp_sanity ; @mkdir -p results; ./exp_sanity > $@
results/transient.txt: exp_transient ; @mkdir -p results; ./exp_transient > $@
results/texture.txt: exp_texture ; @mkdir -p results; ./exp_texture > $@
results/engine_checks.txt: exp_engine_checks ; @mkdir -p results; ./exp_engine_checks > $@
results/fft_study.txt: exp_fft_study ; @mkdir -p results; ./exp_fft_study > $@

results: $(SCENES:%=results/compare_%.txt) results/sanity.txt results/transient.txt results/texture.txt results/engine_checks.txt results/fft_study.txt
.PHONY: all results clean
clean: ; rm -f $(EXPS) pffft.o pffft_common.o
