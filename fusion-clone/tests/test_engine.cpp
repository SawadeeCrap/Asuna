// Engine acceptance tests. Each block maps to one of the 15 acceptance tests of the design brief (see docs/ARCHITECTURE.md §Acceptance).
// Uses only the portable FFT and the in-repo source model so it builds with a plain `g++ -std=c++11`.
#include "../research/common/fusion_source.hpp"
#include "../research/common/metrics.hpp"
#include "../src/dsp/Engine.hpp"
#include <cstdio>

using namespace research;
static const double FS = 48000;
static int fails = 0, total = 0;
#define CHECK(cond, ...) do { total++; if (!(cond)) { fails++; printf("  FAIL  "); printf(__VA_ARGS__); printf("\n"); } else { printf("  ok    "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::vector<float> source(double f0, double seconds, double sub = 0.0, uint32_t seed = 5) {
	SourceSpec s; s.fs = FS; s.f0 = f0; s.seconds = seconds; s.seed = seed; s.noiseDb = -85; s.sub = sub; s.startPhase = 0.0;
	auto x = renderSource(s);
	for (auto& v : x) v *= 4.f; // ~ -2 dBFS at 5 V
	return x;
}

struct Out { std::vector<float> l, r; };
static Out run(fc::Engine& e, const std::vector<float>& x) {
	Out o; o.l.resize(x.size()); o.r.resize(x.size());
	for (size_t i = 0; i < x.size(); i++) e.process(x[i], o.l[i], o.r[i]);
	return o;
}
static fc::EngineParams base(int voices) {
	fc::EngineParams p; p.voices = voices; p.spread = 0.5f; p.drift = 0.f; p.character = 0.f; p.harmonic = 0.f; p.phase = 1.f; p.width = 0.f; p.mix = 1.f; p.summing = 0.f;
	return p;
}
static Out runWith(const fc::EngineParams& p, const std::vector<float>& x, fc::Engine** keep = nullptr) {
	static fc::Engine* last = nullptr;
	delete last;
	last = new fc::Engine();
	last->prepare(FS);
	last->setParams(p);
	Out o = run(*last, x);
	if (keep) *keep = last;
	return o;
}
static double relErr(const std::vector<float>& a, const std::vector<float>& b, size_t from) {
	double e = 0, r = 0;
	for (size_t i = from; i < a.size(); i++) { e += (double) (a[i] - b[i]) * (a[i] - b[i]); r += (double) b[i] * b[i]; }
	return e / std::max(r, 1e-30);
}

int main() {
	const double f0 = 110;
	auto x = source(f0, 24);
	const size_t a = (size_t) (8 * FS);
	double rmsIn = rms(x, a);
	const int km = 8; // beat-texture band: harmonic 8 (beats fast enough to average)
	double bandLo = (km - 0.35) * f0, bandHi = (km + 0.35) * f0;
	printf("TEST 1  VOICES = 1 -> output is the input\n");
	{
		fc::EngineParams p = base(1); Out o = runWith(p, x);
		CHECK(relErr(o.l, x, (size_t) (0.5 * FS)) < 1e-12, "identity (rel. error %.1e), zero latency, no processing", relErr(o.l, x, (size_t) (0.5 * FS)));
	}
	printf("TESTS 2-5  VOICES = 2 / 4 / 8 / 16: increasing density, independent-oscillator statistics, no chorus/supersaw signature\n");
	double rmsN[17] = {0};
	for (int N : {2, 4, 8, 16}) {
		fc::EngineParams p = base(N); Out o = runWith(p, x);
		rmsN[N] = rms(o.l, a);
		Psd pi = welch(x, FS, 1 << 17, a), po = welch(o.l, FS, 1 << 17, a);
		Clusters ci = harmonicClusters(pi, f0, 12, 0.03), co = harmonicClusters(po, f0, 12, 0.03);
		double spreadOut = 0; for (int k = 4; k < 12; k++) spreadOut += co.spread[k]; spreadOut /= 8;
		double spreadIn = 0; for (int k = 4; k < 12; k++) spreadIn += ci.spread[k]; spreadIn /= 8;
		EnvStats ev = bandEnvelopeStats(o.l, FS, bandLo, bandHi, a, o.l.size());
		double rec = envRecurrence(o.l, FS, bandLo, bandHi, a, o.l.size());
		double ih = interHarmonicDb(po, f0, 12) - interHarmonicDb(pi, f0, 12);
		double rip = combRippleDb(harmonicProfileDb(po, f0, 20), harmonicProfileDb(pi, f0, 20));
		CHECK(spreadOut > 1.5 * spreadIn + 0.5, "N=%-2d clone lines widen the harmonic clusters: spread %.2f -> %.2f cents", N, spreadIn, spreadOut);
		CHECK(ih < 6.0, "N=%-2d no spurious inter-harmonic energy (%.1f dB vs input)", N, ih);
		CHECK(rip < 2.0, "N=%-2d no comb filtering across harmonics (ripple %.2f dB)", N, rip);
		if (N >= 4) CHECK(ev.cv > 0.3 && ev.cv < 0.75 && rec < 0.9, "N=%-2d independent-oscillator beat texture: envelope CV %.2f (Rayleigh 0.52), recurrence %.2f (<0.9; supersaw ~1.0)", N, ev.cv, rec);
		CHECK(rmsN[N] / rmsIn > 0.6 && rmsN[N] / rmsIn < 1.7, "N=%-2d loudness follows the density law: %+.1f dB vs the original (not +%.0f dB)", N, db(rmsN[N] / rmsIn), 20 * std::log10((double) N));
	}
	printf("TEST 6  SPREAD = 0 -> every clone sits on the original pitch, no modulation\n");
	{
		fc::EngineParams p = base(8); p.spread = 0.f; fc::Engine* e = nullptr; Out o = runWith(p, x, &e);
		double maxc = 0; for (int v = 1; v < 8; v++) maxc = std::max(maxc, std::fabs(e->voiceCents(v)));
		CHECK(maxc < 1e-6, "all clone ratios are exactly 1.0 (max |cents| = %.2e)", maxc);
		Psd po = welch(o.l, FS, 1 << 17, a); Clusters co = harmonicClusters(po, f0, 12, 0.03), ci = harmonicClusters(welch(x, FS, 1 << 17, a), f0, 12, 0.03);
		CHECK(co.spread[8] < ci.spread[8] * 1.5 + 0.2, "cluster width unchanged (%.2f vs %.2f cents): no uncontrolled modulation", co.spread[8], ci.spread[8]);
	}
	printf("TEST 7  DRIFT = 0 -> no random pitch movement\n");
	{
		fc::EngineParams p = base(8); p.drift = 0.f; fc::Engine* e = nullptr; fc::Engine eng; eng.prepare(FS); eng.setParams(p);
		double mn[16], mx[16]; for (int v = 1; v < 8; v++) mn[v] = 1e9, mx[v] = -1e9;
		for (size_t i = 0; i < (size_t) (10 * FS); i++) { float l, r; eng.process(x[i], l, r); if (i > (size_t) FS) for (int v = 1; v < 8; v++) { double c = eng.voiceCents(v); mn[v] = std::min(mn[v], c); mx[v] = std::max(mx[v], c); } }
		double worst = 0; for (int v = 1; v < 8; v++) worst = std::max(worst, mx[v] - mn[v]);
		CHECK(worst < 1e-9, "voice pitches are constant over 9 s (max excursion %.2e cents)", worst);
		(void) e;
	}
	printf("TEST 8  DRIFT = 100 -> slow, bounded, independent analog-like instability\n");
	{
		fc::EngineParams p = base(8); p.drift = 1.f; p.spread = 0.f; fc::Engine eng; eng.prepare(FS); eng.setParams(p);
		// The drift is an Ornstein-Uhlenbeck process: diffusive (non-differentiable) at short lags by construction, so "slow" is judged at the
		// 1 s scale, where a vibrato-like modulation (5 Hz, +-10 cents = 300 cents/s peak) would be two orders of magnitude faster.
		std::vector<double> c[8]; double maxAbs = 0;
		for (size_t i = 0; i < (size_t) (240 * FS); i++) { float l, r; eng.process(0.f, l, r);
			if (i % 4800 == 0 && i > (size_t) (30 * FS)) { for (int v = 1; v < 8; v++) { double cc = 1200.0 * std::log2(eng.voiceRatioTarget(v)); c[v].push_back(cc); maxAbs = std::max(maxAbs, std::fabs(cc)); } } }
		double maxSlope = 0; for (int v = 1; v < 8; v++) for (size_t k = 10; k < c[v].size(); k++) maxSlope = std::max(maxSlope, std::fabs(c[v][k] - c[v][k - 10]) / 1.0);
		double corr = 0, s1 = 0, s2 = 0; for (size_t k = 0; k < c[1].size(); k++) { corr += c[1][k] * c[2][k]; s1 += c[1][k] * c[1][k]; s2 += c[2][k] * c[2][k]; }
		corr /= std::sqrt(s1 * s2 + 1e-30);
		CHECK(maxAbs < 5.0 && maxAbs > 0.2, "drift is bounded (|max| %.2f cents <= 3 sigma) and present", maxAbs);
		CHECK(maxSlope < 3.0, "drift is slow: max pitch velocity over 1 s = %.2f cents/s (vibrato would be > 100)", maxSlope);
		CHECK(std::fabs(corr) < 0.85, "voices drift independently (correlation of two voices %.2f)", corr);
	}
	printf("TESTS 9/10  PHASE = 0 -> maximum phase relationship, PHASE = 100 -> maximum useful divergence\n");
	{
		auto sx = source(f0, 4);
		fc::EngineParams p = base(8); p.spread = 0.f; p.phase = 0.f; Out o0 = runWith(p, sx);
		p.phase = 1.f; Out o1 = runWith(p, sx);
		double r0 = rms(o0.l, (size_t) (2 * FS)) / rms(sx, (size_t) (2 * FS)), r1 = rms(o1.l, (size_t) (2 * FS)) / rms(sx, (size_t) (2 * FS));
		double coherent = std::pow(8.0, 0.575);
		CHECK(std::fabs(r0 / coherent - 1.0) < 0.12, "PHASE=0: clones are phase-locked to the source (gain %.2f, coherent sum %.2f)", r0, coherent);
		CHECK(r1 < 0.7 * r0, "PHASE=100: start phases are decorrelated (gain %.2f vs %.2f at PHASE=0, %.1f dB lower)", r1, r0, db(r0 / r1));
	}
	printf("TESTS 11/12  HARMONIC = 0 -> voices preserve the source spectral identity, HARMONIC = 100 -> subtle individuality\n");
	{
		auto sx = source(f0, 3);
		auto spectrumOf = [&](fc::Engine& e, int v, std::vector<double>& mag) {
			const float* t; int Nc; if (!e.debugVoiceTable(v, t, Nc)) return false;
			fc::RealFFT fft(Nc); std::vector<float> in(Nc), sp(Nc); for (int i = 0; i < Nc; i++) in[i] = t[i]; fft.forward(in.data(), sp.data());
			mag.assign(60, 0.0); for (int k = 1; k <= 60 && k < Nc / 2; k++) mag[k - 1] = std::sqrt((double) sp[2 * k] * sp[2 * k] + (double) sp[2 * k + 1] * sp[2 * k + 1]);
			return true; };
		fc::EngineParams p = base(4); p.harmonic = 0.f; p.character = 0.f; fc::Engine e0; e0.prepare(FS); e0.setParams(p); run(e0, sx);
		p.harmonic = 1.f; fc::Engine e1; e1.prepare(FS); e1.setParams(p); run(e1, sx);
		std::vector<double> m01, m02, m11, m12; bool ok = spectrumOf(e0, 1, m01) && spectrumOf(e0, 2, m02) && spectrumOf(e1, 1, m11) && spectrumOf(e1, 2, m12);
		CHECK(ok, "per-voice period tables available");
		double d0 = 0, d1 = 0, dv = 0; int n = 0;
		for (int k = 0; k < 60; k++) if (m01[k] > 1e-6) { d0 += std::pow(20 * std::log10(m02[k] / m01[k]), 2); d1 += std::pow(20 * std::log10(m11[k] / m01[k]), 2); dv += std::pow(20 * std::log10(m12[k] / m11[k]), 2); n++; }
		d0 = std::sqrt(d0 / n); d1 = std::sqrt(d1 / n); dv = std::sqrt(dv / n);
		CHECK(d0 < 0.05, "HARMONIC=0: two clone tables are spectrally identical to the source (%.3f dB rms)", d0);
		CHECK(d1 > 0.1 && d1 < 2.5, "HARMONIC=100: clone spectrum differs from the source by a subtle %.2f dB rms (same instrument)", d1);
		CHECK(dv > 0.1, "HARMONIC=100: each voice has its own individuality (voice1 vs voice2 %.2f dB rms)", dv);
	}
	printf("TEST 13  WIDTH = 0 -> mono-compatible; WIDTH > 0 keeps the mono fold-down unchanged\n");
	{
		auto sx = source(f0, 3);
		fc::EngineParams p = base(12); p.width = 0.f; Out o0 = runWith(p, sx);
		double dlr = 0; for (size_t i = 0; i < o0.l.size(); i++) dlr = std::max(dlr, (double) std::fabs(o0.l[i] - o0.r[i]));
		CHECK(dlr == 0.0, "WIDTH=0: L == R exactly");
		p.width = 1.f; Out o1 = runWith(p, sx);
		double dm = 0, sc = 0, dside = 0; for (size_t i = 0; i < o1.l.size(); i++) { dm = std::max(dm, (double) std::fabs(0.5f * (o1.l[i] + o1.r[i]) - o0.l[i])); sc = std::max(sc, (double) std::fabs(o0.l[i])); dside = std::max(dside, (double) std::fabs(o1.l[i] - o1.r[i])); }
		CHECK(dm < 1e-4 * sc, "WIDTH=100: (L+R)/2 equals the WIDTH=0 mono signal (max dev %.1e of peak)", dm / sc);
		CHECK(dside > 0.01 * sc, "WIDTH=100: voices are actually spread (side signal %.1f %% of peak) but only slightly", 100 * dside / sc);
	}
	printf("TESTS 14/15  MIX = 0 -> original only, MIX = 100 -> processed multi-voice result\n");
	{
		auto sx = source(f0, 3);
		fc::EngineParams p = base(16); p.mix = 0.f; Out o = runWith(p, sx);
		CHECK(relErr(o.l, sx, (size_t) (0.5 * FS)) < 1e-12, "MIX=0: output equals the original (rel. error %.1e)", relErr(o.l, sx, (size_t) (0.5 * FS)));
		p.mix = 1.f; o = runWith(p, sx);
		CHECK(relErr(o.l, sx, (size_t) (1.5 * FS)) > 0.05, "MIX=100: processed multi-voice result differs from the input (rel. difference %.2f)", relErr(o.l, sx, (size_t) (1.5 * FS)));
	}
	printf("ROBUSTNESS  hostile inputs, sample rates, pitch step, original path\n");
	{
		for (double sr : {44100.0, 88200.0, 96000.0, 192000.0}) {
			SourceSpec s; s.fs = sr; s.f0 = 110; s.seconds = 3; s.noiseDb = -85; auto y = renderSource(s); for (auto& v : y) v *= 4.f;
			fc::Engine e; e.prepare(sr); fc::EngineParams p = base(8); e.setParams(p);
			bool fin = true; size_t lockAt = 0; for (size_t i = 0; i < y.size(); i++) { float l, r; e.process(y[i], l, r); if (!(l == l) || std::fabs(l) > 20) fin = false; if (!lockAt && e.lockedNow()) lockAt = i; }
			CHECK(fin && lockAt > 0 && lockAt < (size_t) (0.25 * sr), "sample rate %6.0f Hz: locks in %.0f ms, output finite", sr, 1000.0 * lockAt / sr);
		}
		fc::Engine e; e.prepare(FS); fc::EngineParams p = base(16); e.setParams(p);
		fc::Rng rr(3); bool fin = true; float pk = 0;
		for (size_t i = 0; i < (size_t) (4 * FS); i++) { float in = (i < 48000) ? 0.f : (i < 96000 ? 50.f : (i < 144000 ? 0.5f * rr.bipolar() : 1e9f * ((i & 1) ? 1 : -1))); float l, r; e.process(in, l, r); if (!(l == l)) fin = false; pk = std::max(pk, std::fabs(l)); }
		CHECK(fin && e.guardHits() == 0, "silence, DC, noise and +-1e9 garbage never produce NaN/Inf (safety-net hits: %d)", e.guardHits());
		auto sx = source(f0, 6, 0.0);
		fc::EngineParams pp = base(8); Out o = runWith(pp, sx);
		fc::Engine eng2; eng2.prepare(FS); eng2.setParams(pp); bool ok = true;
		for (size_t i = 0; i < sx.size(); i++) { float l, r; eng2.process(sx[i], l, r); (void) l; (void) r; }
		ok = eng2.status().latencySamples == 0.f;
		CHECK(ok, "original path latency is exactly 0 samples");
	}
	printf("\n%d / %d checks passed\n", total - fails, total);
	return fails ? 1 : 0;
}
