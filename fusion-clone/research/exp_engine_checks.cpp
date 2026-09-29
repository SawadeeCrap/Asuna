// Engine scenario checks: lock time, level, pitch step, glide, note-on, hostile inputs. Prints a table.
#include "common/fusion_source.hpp"
#include "common/metrics.hpp"
#include "common/wav.hpp"
#include "../src/dsp/Engine.hpp"
#include <cstdio>

using namespace research;
static const double FS = 48000;

struct RunResult { std::vector<float> l, r; double lockTime = -1; int relocks = 0; double maxJump = 0; bool bad = false; };

static RunResult run(const std::vector<float>& x, const fc::EngineParams& p, double f0hint = 0) {
	fc::Engine e; e.prepare(FS); e.setParams(p);
	RunResult rr; rr.l.resize(x.size()); rr.r.resize(x.size());
	bool wasLocked = false; float prev = 0;
	for (size_t i = 0; i < x.size(); i++) {
		e.process(x[i], rr.l[i], rr.r[i]);
		if (!(rr.l[i] == rr.l[i]) || std::fabs(rr.l[i]) > 50) rr.bad = true;
		bool lk = e.lockedNow();
		if (lk && !wasLocked) { if (rr.lockTime < 0) rr.lockTime = i / FS; else rr.relocks++; }
		wasLocked = lk;
		rr.maxJump = std::max(rr.maxJump, (double) std::fabs(rr.l[i] - prev)); prev = rr.l[i];
	}
	return rr;
}

static std::vector<float> scale(std::vector<float> x, float g) { for (auto& v : x) v *= g; return x; }

int main() {
	fc::EngineParams p; p.voices = 8; p.spread = 0.5f; p.drift = 0.2f; p.character = 0.f; p.harmonic = 0.3f; p.phase = 1.f; p.summing = 0.f;
	printf("=== 1. steady state across pitch (saw, N=8) ===\n");
	printf("%8s | lock(s) | out/in RMS dB (expect +1.4 dB +/- interference) | maxJump/inMaxJump | NaN\n", "f0");
	for (double f0 : {16.35, 20.0, 41.2, 110.0, 440.0, 1760.0, 3520.0}) {
		SourceSpec s; s.fs = FS; s.f0 = f0; s.seconds = 6; s.seed = 9; s.noiseDb = -80;
		auto x = scale(renderSource(s), 4.f);
		RunResult r = run(x, p);
		double jin = 0; for (size_t i = 1; i < x.size(); i++) jin = std::max(jin, (double) std::fabs(x[i] - x[i - 1]));
		printf("%8.1f | %7.3f | %+6.2f | %5.2f | %s\n", f0, r.lockTime, db(rms(r.l, 3 * 48000) / rms(x, 3 * 48000)), r.maxJump / jin, r.bad ? "BAD" : "ok");
	}
	printf("\n=== 2. waveform / source variants at 110 Hz, N=8 ===\n");
	struct V { const char* n; double saw, tri, pulse, sine, pw, sub; int det; double dk, tube; };
	V vs[] = {{"saw",1,0,0,0,.5,0,0,0,0},{"tri",0,1,0,0,.5,0,0,0,0},{"pulse10%",0,0,1,0,.1,0,0,0,0},{"sine",0,0,0,1,.5,0,0,0,0},
	          {"saw+sub.6",1,0,0,0,.5,.6,0,0,0},{"saw+dopplerDet.3",1,0,0,0,.5,0,DETUNE_DOPPLER,.3,0},{"saw+ssbDet.3",1,0,0,0,.5,0,DETUNE_SSB,.3,0},
	          {"pulse+sub+tube.5",0,0,1,0,.3,.5,0,0,.5},{"saw+sub+doppler+tube",1,0,0,0,.5,.5,DETUNE_DOPPLER,.4,.4}};
	for (auto& v : vs) {
		SourceSpec s; s.fs = FS; s.f0 = 110; s.seconds = 8; s.seed = 3; s.noiseDb = -70; s.wSaw = v.saw; s.wTri = v.tri; s.wPulse = v.pulse; s.wSine = v.sine; s.pulseWidth = v.pw; s.sub = v.sub;
		s.detuneModel = v.det; s.detune = v.dk; s.tube = v.tube;
		auto x = scale(renderSource(s), 4.f);
		RunResult r = run(x, p);
		fc::Engine e; 
		printf("%-22s lock %.3fs  relocks %d  out/in %+5.2f dB  %s\n", v.n, r.lockTime, r.relocks, db(rms(r.l, 3 * 48000) / rms(x, 3 * 48000)), r.bad ? "BAD" : "ok");
	}
	printf("\n=== 3. pitch step 110 -> 165 Hz at t=2.0 s (N=8) ===\n");
	{
		SourceSpec s; s.fs = FS; s.f0 = 110; s.seconds = 5; s.seed = 4; s.noiseDb = -80;
		s.f0Track.resize((size_t) (5 * FS)); for (size_t i = 0; i < s.f0Track.size(); i++) s.f0Track[i] = i < 2 * FS ? 110.0 : 165.0;
		auto x = scale(renderSource(s), 4.f);
		RunResult r = run(x, p);
		auto ein = rmsEnvelope(x, FS, 5.0, 5.0), eout = rmsEnvelope(r.l, FS, 5.0, 5.0);
		printf("relocks after step: %d ; level (dB out/in) in 5 ms slices around the step:\n  ", r.relocks);
		for (int k = 380; k < 440; k += 3) printf("%+.1f ", db(eout[k] / ein[k]));
		printf("\n  max jump ratio %.2f\n", r.maxJump / 0.4);
	}
	printf("\n=== 4. glide 110 -> 220 Hz over 1 s ===\n");
	{
		SourceSpec s; s.fs = FS; s.f0 = 110; s.seconds = 5; s.seed = 4; s.noiseDb = -80;
		s.f0Track.resize((size_t) (5 * FS)); for (size_t i = 0; i < s.f0Track.size(); i++) { double t = i / FS; double u = std::min(1.0, std::max(0.0, t - 1.5)); s.f0Track[i] = 110.0 * std::pow(2.0, u); }
		auto x = scale(renderSource(s), 4.f);
		fc::Engine e; e.prepare(FS); e.setParams(p);
		std::vector<float> l(x.size()), rr(x.size()); int unl = 0; double maxErr = 0, maxErrClone = 0;
		for (size_t i = 0; i < x.size(); i++) { e.process(x[i], l[i], rr[i]); double t = i / FS; if (t > 1.6 && t < 2.4) { if (!e.lockedNow()) unl++; else { double f = e.analyzer().omega() * FS; double tr = s.f0Track[i]; maxErr = std::max(maxErr, std::fabs(1200 * std::log2(f / tr))); maxErrClone = std::max(maxErrClone, std::fabs(1200 * std::log2(e.cloneUnitFreq() / tr))); } } }
		printf("unlocked samples during glide: %d of %d; max pitch error while locked: tracker %.2f cents, clones (with slope lead) %.2f cents\n", unl, (int) (0.8 * FS), maxErr, maxErrClone);
	}
	printf("\n=== 5. hostile inputs (N=16): must stay finite, bounded, no lock on garbage ===\n");
	{
		fc::EngineParams q = p; q.voices = 16;
		struct H { const char* n; std::vector<float> x; } hs[5];
		size_t n = (size_t) (4 * FS);
		hs[0].n = "silence"; hs[0].x.assign(n, 0.f);
		hs[1].n = "DC 0.5"; hs[1].x.assign(n, 0.5f);
		hs[2].n = "white noise"; { fc::Rng r(1); hs[2].x.resize(n); for (auto& v : hs[2].x) v = 0.5f * r.bipolar(); }
		hs[3].n = "hard clipped saw 5x"; { SourceSpec s; s.fs = FS; s.f0 = 220; s.seconds = 4; auto y = renderSource(s); for (auto& v : y) v = std::max(-1.f, std::min(1.f, v * 20.f)); hs[3].x = y; }
		hs[4].n = "chord (3 saws)"; { SourceSpec s; s.fs = FS; s.seconds = 4; s.f0 = 110; auto a = renderSource(s); s.f0 = 138.6; auto b = renderSource(s); s.f0 = 164.8; auto c = renderSource(s); hs[4].x.resize(n); for (size_t i = 0; i < n; i++) hs[4].x[i] = 2.f * (a[i] + b[i] + c[i]); }
		for (auto& h : hs) { RunResult r = run(h.x, q); double pk = 0; for (float v : r.l) pk = std::max(pk, (double) std::fabs(v)); double pin = 0; for (float v : h.x) pin = std::max(pin, (double) std::fabs(v));
			printf("%-22s peak in %.2f out %.2f  lock at %.2fs relocks %d  %s\n", h.n, pin, pk, r.lockTime, r.relocks, r.bad ? "BAD(NaN/huge)" : "finite"); }
	}
	return 0;
}
