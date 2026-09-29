// Time-domain / frequency-shift candidates and the engine adapter (see candidates.hpp for the list).
#pragma once
#include "candidates.hpp"

namespace research {

/** Simple period estimator for the TD candidates: coarse YIN lanes (the production tracker) + full-rate refinement. Returns the
    repeating-unit period (samples) sampled every `hop` samples; 0 where unknown. */
inline std::vector<double> estimatePeriodTrack(const std::vector<float>& x, double fs, int hop) {
	fc::PitchTracker pt;
	pt.prepare(fs);
	std::vector<double> track;
	double last = 0;
	for (size_t i = 0; i < x.size(); i++) {
		if (pt.push(x[i] * 2.5f)) {
			const auto& e = pt.estimate();
			if (e.valid && e.conf > 0.8f) {
				// full-rate refinement around the coarse estimate
				double P0 = e.period;
				int W = std::min(6144, (int) std::ceil(1.5 * P0)), span = std::max(3, (int) std::ceil(0.035 * P0));
				int lag0 = (int) std::floor(P0) - span, lag1 = (int) std::ceil(P0) + span;
				if (lag0 > 2 && i > (size_t) (lag1 + W + 8)) {
					std::vector<double> d(lag1 - lag0 + 1);
					double best = 1e300; int bl = lag0;
					for (int lag = lag0; lag <= lag1; lag++) {
						double acc = 0;
						for (int j = 0; j < W; j++) { double df = x[i - j] - x[i - j - lag]; acc += df * df; }
						d[lag - lag0] = acc;
						if (acc < best) { best = acc; bl = lag; }
					}
					if (bl > lag0 && bl < lag1) {
						double a = d[bl - 1 - lag0], b = d[bl - lag0], c = d[bl + 1 - lag0], den = a - 2 * b + c;
						double off = std::fabs(den) > 1e-20 ? 0.5 * (a - c) / den : 0.0;
						last = bl + std::max(-1.0, std::min(1.0, off));
					}
				}
			}
		}
		if ((int) (i % hop) == 0) track.push_back(last);
	}
	return track;
}

// ------------------------------------------------------------------------------------------------------------------------------
/** F1: classic rotating-head pitch shifter (two delay taps 180 deg apart, Hann cross-fade). */
class RotatingDelay : public Candidate {
public:
	explicit RotatingDelay(double windowMs = 40) : winMs_(windowMs) { name_ = "F1 rotating 2-tap delay (chorus-style), " + std::to_string((int) windowMs) + " ms"; }
	const char* name() const override { return name_.c_str(); }
	std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) override {
		const double W = winMs_ * 1e-3 * c.fs;
		fc::MirrorRing ring;
		ring.alloc(fc::nextPow2((int) (W * 2 + 200)));
		fc::Rng r(c.seed);
		double d1 = c.randomPhase ? r.uniform() * W : 0.0;
		std::vector<float> y(x.size());
		const double dmin = 24.0;
		for (size_t n = 0; n < x.size(); n++) {
			ring.push(x[n]);
			d1 += (1.0 - ratio);
			d1 -= W * std::floor(d1 / W);
			double d2 = d1 + 0.5 * W;
			d2 -= W * std::floor(d2 / W);
			double g1 = 0.5 * (1 - std::cos(fc::kTwoPi * d1 / W)), g2 = 0.5 * (1 - std::cos(fc::kTwoPi * d2 / W));
			uint64_t newest = ring.count() - 1;
			double p1 = (double) newest - dmin - d1, p2 = (double) newest - dmin - d2;
			float a = p1 > 20 ? ring.readSinc<16>(p1) : 0.f, b = p2 > 20 ? ring.readSinc<16>(p2) : 0.f;
			y[n] = (float) (g1 * a + g2 * b);
		}
		return y;
	}
private:
	double winMs_;
	std::string name_;
};

// ------------------------------------------------------------------------------------------------------------------------------
/** F2: period-synchronous resampler. A read head runs at `ratio` behind the write head; whenever the lag leaves one repeating-unit
    wide window it jumps by exactly one period with a short cross-fade (seamless for truly periodic input). */
class PeriodJump : public Candidate {
public:
	const char* name() const override { return "F2 period-synchronous resampler (YIN period)"; }
	bool supportsVarPitch() const override { return true; }
	std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) override {
		std::vector<double> track = estimatePeriodTrack(x, c.fs, 64);
		fc::MirrorRing ring;
		ring.alloc(fc::nextPow2((int) c.fs * 2));
		fc::Rng r(c.seed);
		std::vector<float> y(x.size(), 0.f);
		const double dmin = 24.0;
		double P = track.empty() || track[track.size() / 2] <= 0 ? c.fs / c.f0 : track[track.size() / 2];
		double lag = dmin + (c.randomPhase ? r.uniform() * P : 0.0), lagB = lag;
		int xf = 0, xfN = 1;
		for (size_t n = 0; n < x.size(); n++) {
			ring.push(x[n]);
			size_t ti = n / 64;
			if (ti < track.size() && track[ti] > 0) P = track[ti];
			lag += (1.0 - ratio);
			lagB += (1.0 - ratio);
			if (lag < dmin || lag > dmin + P) {
				lagB = lag;
				lag += lag < dmin ? P : -P;
				xfN = std::max(16, std::min(1440, (int) (0.2 * P)));
				xf = xfN;
			}
			uint64_t newest = ring.count() - 1;
			double pa = (double) newest - lag;
			float a = pa > 20 ? ring.readSinc<16>(pa) : 0.f;
			if (xf > 0) {
				double pb = (double) newest - lagB;
				float b = pb > 20 ? ring.readSinc<16>(pb) : 0.f;
				float w = 0.5f - 0.5f * std::cos((float) (fc::kPi * (xfN - xf) / xfN));
				a = a * w + b * (1.f - w);
				xf--;
			}
			y[n] = a;
		}
		return y;
	}
};

// ------------------------------------------------------------------------------------------------------------------------------
/** G: PSOLA — Hann-windowed 2-period grains centred on analysis marks, re-placed at spacing P/ratio. */
class Psola : public Candidate {
public:
	const char* name() const override { return "G  PSOLA (pitch-synchronous OLA)"; }
	std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) override {
		std::vector<double> track = estimatePeriodTrack(x, c.fs, 64);
		double P = track.empty() ? c.fs / c.f0 : track[track.size() / 2];
		if (P <= 0) P = c.fs / c.f0;
		fc::MirrorRing ring;
		ring.alloc(fc::nextPow2((int) x.size() + 256));
		for (float v : x) ring.push(v);
		for (int i = 0; i < 64; i++) ring.push(0.f);
		std::vector<float> y(x.size(), 0.f);
		fc::Rng r(c.seed);
		double a0 = 2.0 * P + (c.randomPhase ? r.uniform() * P : 0.0);
		double sp = P / ratio;
		for (double s = 2.0 * P; s < (double) x.size() - 2 * P; s += sp) {
			long i = std::lround((s - a0) / P);
			double a = a0 + i * P;
			int lo = (int) std::floor(s - P), hi = (int) std::ceil(s + P);
			for (int n = std::max(lo, 0); n <= hi && n < (int) x.size(); n++) {
				double u = (n - s + P) / (2 * P); // 0..1
				if (u < 0 || u > 1) continue;
				double w = 0.5 - 0.5 * std::cos(fc::kTwoPi * u);
				double pos = a + (n - s);
				if (pos < 20) continue;
				y[n] += (float) (w * ring.readSinc<16>(pos) / ratio);
			}
		}
		return y;
	}
};

// ------------------------------------------------------------------------------------------------------------------------------
/** E: additive frequency shift (Bode-style), Delta = f0*(ratio-1) so the fundamental is shifted exactly like a pitch shift would. */
class FreqShifter : public Candidate {
public:
	const char* name() const override { return "E  SSB frequency shift (additive Hz), fundamental-matched"; }
	std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) override {
		const int H = 4095;
		std::vector<float> hk(H, 0.f);
		for (int k = 0; k < H; k++) {
			int m = k - H / 2;
			if (m & 1) hk[k] = (float) (2.0 / (fc::kPi * m) * fc::kaiser((double) k, (double) (H - 1), 8.0));
		}
		std::vector<float> hil = fftConvolveSame(x, hk);
		double delta = c.f0 * (ratio - 1.0);
		fc::Rng r(c.seed);
		double ph0 = c.randomPhase ? r.uniform() : 0.0;
		std::vector<float> y(x.size());
		for (size_t i = 0; i < x.size(); i++) {
			double a = fc::kTwoPi * (delta * (double) i / c.fs + ph0);
			y[i] = (float) (x[i] * std::cos(a) - hil[i] * std::sin(a));
		}
		return y;
	}
};

// ------------------------------------------------------------------------------------------------------------------------------
/** J: the production engine. Not a per-clone candidate: the driver calls renderEngine() for the whole bank. */
inline std::vector<float> renderEngine(const std::vector<float>& x, double fs, const std::vector<double>& centsOffsets, int quality, uint32_t seed,
                                       double phase = 1.0, double harmonic = 0.0, double character = 0.0, double drift = 0.0) {
	fc::Engine e;
	e.prepare(fs);
	fc::EngineParams p;
	p.voices = (int) centsOffsets.size() + 1;
	p.spread = 1.f; p.drift = (float) drift; p.character = (float) character; p.phase = (float) phase; p.harmonic = (float) harmonic; p.summing = 0.f;
	p.quality = quality; p.seed = seed; p.levelLaw = 0.f; // equal power so RMS is comparable
	e.setParams(p);
	float cents[fc::kMaxVoices];
	for (int i = 0; i < fc::kMaxVoices; i++) cents[i] = 0.f;
	for (size_t i = 0; i < centsOffsets.size() && i + 1 < (size_t) fc::kMaxVoices; i++) cents[i + 1] = (float) centsOffsets[i];
	e.setDetuneOverrideCents(cents, (int) centsOffsets.size() + 1);
	std::vector<float> y(x.size()), yr(x.size());
	for (size_t i = 0; i < x.size(); i++) e.process(x[i], y[i], yr[i]);
	return y;
}

} // namespace research
