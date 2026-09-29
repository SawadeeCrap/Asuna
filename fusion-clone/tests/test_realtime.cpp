// Real-time safety checks for the engine's process() path.
//   R1  no heap allocation, ever, inside process() (global operator new/delete are replaced by counting versions that are armed only around
//       process()) — over lock, unlock, pitch steps, glides, parameter sweeps, quality switches, VOICES changes, hostile input.
//   R2  worst-case duration of a single process() call per quality mode (analysis, table builds and FFTs are the expensive events) and the
//       99.9th percentile, reported against the audio budget of one sample.
//   R3  no NaN / Inf on any output sample.
//
// Timing figures depend on the machine; only the allocation and NaN checks fail the test. The spikes are reported so that regressions
// (an FFT that is no longer time-sliced, say) are visible.
#include "../research/common/fusion_source.hpp"
#include "../src/dsp/Engine.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete" // the replaced operator new/delete below are a matched malloc/free pair
#endif

static bool g_armed = false;
static long g_allocs = 0;

void* operator new(std::size_t n) {
	if (g_armed)
		g_allocs++;
	void* p = std::malloc(n ? n : 1);
	if (!p)
		throw std::bad_alloc();
	return p;
}
void* operator new[](std::size_t n) {
	if (g_armed)
		g_allocs++;
	void* p = std::malloc(n ? n : 1);
	if (!p)
		throw std::bad_alloc();
	return p;
}
void operator delete(void* p) noexcept {
	if (g_armed && p)
		g_allocs++; // freeing inside process() is as forbidden as allocating
	std::free(p);
}
void operator delete[](void* p) noexcept {
	if (g_armed && p)
		g_allocs++;
	std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
	if (g_armed && p)
		g_allocs++;
	std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
	if (g_armed && p)
		g_allocs++;
	std::free(p);
}

using namespace research;
static const double FS = 48000.0;
static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct Stats {
	std::vector<double> us;
	void add(double v) { us.push_back(v); }
	double pct(double p) {
		std::vector<double> s = us;
		std::sort(s.begin(), s.end());
		return s.empty() ? 0.0 : s[std::min(s.size() - 1, (size_t) (p * (double) s.size()))];
	}
	double mx() const { return us.empty() ? 0.0 : *std::max_element(us.begin(), us.end()); }
	double mean() const {
		double a = 0;
		for (double v : us)
			a += v;
		return us.empty() ? 0.0 : a / (double) us.size();
	}
};

int main() {
	// ---- R1 + R3: a long, busy scenario with the allocation counter armed around every process() call ---------------------------
	printf("R1/R3  allocation-free and finite over lock, steps, glides, parameter changes, quality switches, hostile input\n");
	{
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p;
		p.voices = 12;
		e.setParams(p);
		SourceSpec a;
		a.fs = FS; a.f0 = 110.0; a.seconds = 1.5; a.seed = 2; a.noiseDb = -80;
		SourceSpec b = a;
		b.f0 = 165.0;
		SourceSpec c = a;
		c.f0 = 55.0; c.sub = 0.5; c.wSaw = 0.0; c.wPulse = 1.0; c.pulseWidth = 0.2; c.tube = 0.4;
		std::vector<float> x;
		for (const SourceSpec* s : {&a, &b, &c}) {
			std::vector<float> seg = renderSource(*s);
			x.insert(x.end(), seg.begin(), seg.end());
		}
		// a glide, a gap of silence, DC, noise, garbage
		for (int i = 0; i < 24000; i++)
			x.push_back(0.f);
		for (int i = 0; i < 24000; i++)
			x.push_back(0.5f);
		uint32_t rng = 12345;
		for (int i = 0; i < 24000; i++) {
			rng = rng * 1664525u + 1013904223u;
			x.push_back(((float) (rng >> 8) / 8388608.f - 1.f) * 2.f);
		}
		x.push_back(1e9f);
		x.push_back(-1e9f);
		x.push_back(NAN);
		x.push_back(INFINITY);
		for (int i = 0; i < 48000; i++)
			x.push_back(0.f);
		bool finite = true;
		const size_t n = x.size();
		g_allocs = 0;
		for (size_t i = 0; i < n; i++) {
			g_armed = true; // setParams() runs on the audio thread in the module too
			if (i % 4800 == 0) { // parameter traffic, every 100 ms
				fc::EngineParams q = p;
				q.voices = 1 + (int) ((i / 4800) % 16);
				q.spread = (float) ((i / 4800) % 7) / 6.f;
				q.drift = (float) ((i / 4800) % 5) / 4.f;
				q.width = (float) ((i / 4800) % 3) / 2.f;
				q.mix = (i / 4800) % 11 == 10 ? 0.f : 1.f;
				q.algorithm = (int) ((i / 4800) % 4 == 3);
				q.quality = (int) ((i / 48000) % 4); // switches quality once a second
				q.originalOnly = (i / 4800) % 13 == 12;
				q.character = (float) ((i / 4800) % 4) / 3.f;
				e.setParams(q);
			}
			float l = 0, r = 0;
			e.process(x[i], l, r);
			g_armed = false;
			if (!(l == l) || !(r == r) || std::fabs(l) > 1e6f || std::fabs(r) > 1e6f)
				finite = false;
		}
		CHECK(g_allocs == 0, "%ld heap allocations/frees inside setParams()/process() over %.1f s of busy input", g_allocs, (double) n / FS);
		CHECK(finite, "every output sample finite and bounded");
	}

	// ---- R2: per-call cost ---------------------------------------------------------------------------------------------------------
	// The engine is deterministic, so the same input can be run several times and each sample's cost taken as the minimum over the runs: that
	// removes scheduler noise and interrupts (which show up as spikes of a millisecond or more on a shared machine) and leaves the compute cost.
	const int kRuns = 3;
	printf("R2  per-sample cost of process() in us (min of %d identical runs per sample; budget at 48 kHz: 20.8 us per sample, 256-sample block = 5.3 ms)\n", kRuns);
	printf("      quality   pitch     mean us   99.9%% us   99.99%% us   max us   samples > 100 us   > 250 us\n");
	static const char* qn[] = {"ECO", "BALANCED", "HIGH", "ULTRA"};
	for (int q = 0; q < 4; q++) {
		for (double f0 : {20.0, 110.0, 880.0}) {
			SourceSpec s;
			s.fs = FS; s.f0 = f0; s.seconds = 3.0; s.seed = 4; s.noiseDb = -80; s.sub = 0.4;
			std::vector<float> x = renderSource(s);
			for (float& v : x)
				v *= 4.f;
			std::vector<double> best(x.size(), 1e30);
			for (int run = 0; run < kRuns; run++) {
				fc::Engine e;
				e.prepare(FS);
				fc::EngineParams p;
				p.voices = 16;
				p.quality = q;
				p.character = 0.5f;
				p.algorithm = fc::ALGO_FUSION;
				e.setParams(p);
				for (size_t i = 0; i < x.size(); i++) {
					float l, r;
					const auto t0 = std::chrono::steady_clock::now();
					e.process(x[i], l, r);
					const auto t1 = std::chrono::steady_clock::now();
					best[i] = std::min(best[i], std::chrono::duration<double, std::micro>(t1 - t0).count());
				}
			}
			Stats st;
			int n100 = 0, n250 = 0;
			for (size_t i = (size_t) (0.5 * FS); i < x.size(); i++) {
				st.add(best[i]);
				n100 += best[i] > 100.0;
				n250 += best[i] > 250.0;
			}
			printf("      %-9s %5.0f Hz  %8.3f  %9.2f  %10.2f  %8.2f  %10d  %14d\n", qn[q], f0, st.mean(), st.pct(0.999), st.pct(0.9999), st.mx(), n100, n250);
			fflush(stdout);
		}
	}
	printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
	return failures ? 1 : 0;
}
