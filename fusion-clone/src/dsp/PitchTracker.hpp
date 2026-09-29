// FusionClone DSP core — coarse period tracker.
//
// Purpose: robust *acquisition and verification* of the period of a monophonic oscillator signal, including
// period doubling caused by a sub oscillator (main + sub repeats every 2 main cycles). Precision is refined later by
// the cycle analyser's phase-slope frequency measurement and Kalman tracker, so ~1 % accuracy here is enough.
//
// Frequencies below are *repeating-unit* frequencies (a sub oscillator halves them).
//
// Design: five overlapping "lanes". Each lane low-passes/decimates the input to a sample rate of ~16 x its highest
// fundamental so that the YIN difference function only has to cover ~16..71 samples, which keeps the per-lane
// cost tiny (a few 10k MACs per run) while the effective analysis window scales with the period being tracked
// (long windows only where the fundamental is low).
//
//   lane 0:  7 ..   62 Hz      lane 3:  640 .. 2800 Hz
//   lane 1: 50 ..  220 Hz      lane 4: 2200 .. 6500 Hz
//   lane 2: 180 ..  800 Hz
//
// Algorithm per lane run: YIN cumulative-mean-normalised difference (de Cheveigné & Kawahara 2002), first dip below an
// absolute threshold, parabolic interpolation; then a *multiplicity* test: if 2x (or 3x, 4x) the found period gives a
// significantly lower normalised difference, the composite waveform really repeats at that longer period (sub
// oscillator) and the longer period is reported (mult = 2..4).
//
// Significance gate: the normalised difference is scale free, so a lane whose short window only sees a flat stretch of a sparse
// waveform (narrow pulse at a low pitch) would happily "find" a period in a tiny residual ripple. Every lane run therefore also
// requires the window to carry a minimum share of the input's long-term AC power (kSignificance); otherwise it reports "no estimate".
#pragma once
#include "Common.hpp"
#include "Filters.hpp"

namespace fc {

class PitchTracker {
public:
	struct Estimate {
		double period = 0.0; // full-rate samples (of the *repeating unit*, i.e. mult * fundamental period)
		int mult = 1;        // number of fundamental cycles per repeating unit
		float conf = 0.f;    // 1 - CMND at the chosen lag
		bool valid = false;
		uint32_t seq = 0;    // increments whenever a new estimate is produced
		int lane = -1;
	};

	static const int kLanes = 5;

	void prepare(double fs) {
		fs_ = fs;
		static const double fLo[kLanes] = {7.0, 50.0, 180.0, 640.0, 2200.0};
		static const double fHi[kLanes] = {62.0, 220.0, 800.0, 2800.0, 6500.0};
		for (int i = 0; i < kLanes; i++) {
			Lane& L = lane_[i];
			L.fLo = fLo[i];
			L.fHi = fHi[i];
			double targetRate = std::min(fs, 16.0 * fHi[i]);
			L.decim = std::max(1, (int) std::floor(fs / targetRate));
			L.rate = fs / L.decim;
			L.tauLo = std::max(4.0, L.rate / fHi[i]);
			L.tauHi = L.rate / fLo[i];
			L.tauMax = (int) std::ceil(2.0 * L.tauHi) + 2;              // room to evaluate the 2x multiple
			L.win = std::max(16, (int) std::ceil(L.tauHi)) + 2;
			int need = L.tauMax + L.win + 8;
			L.bufN = nextPow2(need * 2);
			L.buf.assign(L.bufN, 0.f);
			L.seg.assign(L.tauMax + L.win + 8, 0.f);
			L.d.assign(L.tauMax + 3, 0.f);
			L.cm.assign(L.tauMax + 3, 0.f);
			L.w = 0;
			L.w0 = 0;
			L.hop = (i >= 3) ? 24 : 16;
			L.sinceRun = 0;
			if (L.decim > 1) {
				int taps = std::min(2047, 8 * L.decim + 1);
				if (!(taps & 1))
					taps++;
				L.dec.init(L.decim, taps, 0.42 / L.decim, 8.0);
			} else {
				L.dec.init(1, 1, 0.5);
			}
			L.est = Estimate();
		}
		est_ = Estimate();
		activeMask_ = (1 << kLanes) - 1;
		silenceThresh_ = 1e-6f; // energy per sample
		dcCoef_ = 1.f - (float) std::exp(-1.0 / (0.2 * fs));
		lvlRise_ = 1.f - (float) std::exp(-1.0 / (0.005 * fs));
		lvlFall_ = 1.f - (float) std::exp(-1.0 / (0.1 * fs));
		dc_ = level2_ = 0.f;
	}

	void reset() {
		for (int i = 0; i < kLanes; i++) {
			Lane& L = lane_[i];
			std::fill(L.buf.begin(), L.buf.end(), 0.f);
			L.dec.reset();
			L.w = 0;
			L.w0 = 0;
			L.sinceRun = 0;
			L.est = Estimate();
		}
		est_ = Estimate();
		dc_ = level2_ = 0.f;
	}

	/** Forget the history older than `backSamples` full-rate samples (call after a detected pitch step / onset so that the
	    period search only sees clean post-event data). */
	void forget(int backSamples) {
		for (int i = 0; i < kLanes; i++) {
			Lane& L = lane_[i];
			uint64_t back = (uint64_t) std::max(0, backSamples) / (uint64_t) L.decim;
			L.w0 = L.w > back ? L.w - back : 0;
			L.est = Estimate();
		}
		est_ = Estimate();
	}

	/** Restrict which lanes run (bit i = lane i). Used once locked, to save CPU. */
	void setActiveMask(int mask) { activeMask_ = mask; }
	int laneForPeriod(double periodSamples) const {
		double f = fs_ / std::max(periodSamples, 1.0);
		int best = kLanes - 1;
		for (int i = 0; i < kLanes; i++)
			if (f <= lane_[i].fHi * 1.05) {
				best = i;
				break;
			}
		return best;
	}

	double laneHopSeconds(int i) const { return lane_[i].hop * lane_[i].decim / fs_; }
	const Estimate& estimate() const { return est_; }
	const Estimate& laneEstimate(int i) const { return lane_[i].est; }

	/** Feed one full-rate sample. Returns true when a fresh (valid or invalid) estimate has been published. */
	bool push(float x) {
		// long-term AC power reference for the significance gate (DC removed, fast attack / 100 ms release)
		dc_ += (x - dc_) * dcCoef_;
		const float ac2 = (x - dc_) * (x - dc_);
		level2_ += (ac2 - level2_) * (ac2 > level2_ ? lvlRise_ : lvlFall_);
		bool fresh = false;
		for (int i = 0; i < kLanes; i++) {
			Lane& L = lane_[i];
			if (!(activeMask_ & (1 << i)))
				continue;
			bool out = (L.decim == 1) ? (L.dec.push(x), true) : L.dec.push(x);
			if (!out)
				continue;
			float v = (L.decim == 1) ? x : L.dec.out();
			L.buf[L.w & (L.bufN - 1)] = v;
			L.w++;
			if (++L.sinceRun >= L.hop && L.w - L.w0 >= (uint64_t) minData(L)) {
				L.sinceRun = 0;
				runLane(i);
				fresh = true;
			}
		}
		if (fresh)
			choose();
		return fresh;
	}

private:
	struct Lane {
		double fLo = 0, fHi = 0, rate = 0, tauLo = 0, tauHi = 0;
		int decim = 1, tauMax = 0, win = 0, bufN = 0, hop = 16, sinceRun = 0;
		uint64_t w = 0, w0 = 0; // w0: first lane sample that may be used (history before it is forgotten)
		std::vector<float> buf, seg, d, cm;
		FirDecimator dec;
		Estimate est;
	};

	static void parabolic(float a, float b, float c, double& off, double& val) {
		double den = (double) a - 2.0 * b + c;
		if (std::fabs(den) < 1e-12) {
			off = 0;
			val = b;
			return;
		}
		off = 0.5 * ((double) a - c) / den;
		off = clampT(off, -1.0, 1.0);
		val = b - 0.25 * ((double) a - c) * off;
	}

	static int minData(const Lane& L) {
		// enough for lags up to ~2x the smallest period plus an equally long window
		return std::max(32, (int) std::ceil(4.0 * L.tauLo));
	}

	void runLane(int li) {
		Lane& L = lane_[li];
		// Progressive analysis: use only the data that is actually available. Lags up to avail/2 can be tested with a window at
		// least as long as the lag, so a 20 Hz note is detected after ~2.5 periods instead of waiting for a worst-case (7 Hz) window.
		const int fullTotal = L.tauMax + L.win;
		const int avail = (int) std::min<uint64_t>(L.w - L.w0, (uint64_t) fullTotal);
		const int tauMax = std::min(L.tauMax, avail / 2 - 1);
		const int W = std::max(8, std::min(L.win, avail - tauMax - 1));
		const int total = tauMax + W;
		// gather the most recent `total` samples, oldest first
		uint64_t start = L.w - total;
		double energy = 0, sum = 0;
		for (int i = 0; i < total; i++) {
			float v = L.buf[(start + i) & (L.bufN - 1)];
			L.seg[i] = v;
			if (i >= tauMax) {
				energy += (double) v * v;
				sum += v;
			}
		}
		energy /= W;
		const double mean = sum / W;
		const double windowAc = std::max(0.0, energy - mean * mean);
		const double kSignificance = 0.05; // minimum window AC power as a share of the long-term AC power
		Estimate e;
		e.lane = li;
		e.seq = L.est.seq + 1;
		if (energy < silenceThresh_ || windowAc < kSignificance * (double) level2_) {
			L.est = e; // invalid: silence, or a window that only sees an insignificant stretch of the signal
			return;
		}
		const float* s = L.seg.data();
		// difference function
		L.d[0] = 0.f;
		for (int tau = 1; tau <= tauMax; tau++) {
			const float* a = s + tauMax;
			const float* b = s + tauMax - tau;
			float acc = 0.f;
			for (int j = 0; j < W; j++) {
				float df = a[j] - b[j];
				acc += df * df;
			}
			L.d[tau] = acc;
		}
		// cumulative mean normalised difference
		L.cm[0] = 1.f;
		double run = 0;
		for (int tau = 1; tau <= tauMax; tau++) {
			run += L.d[tau];
			L.cm[tau] = run > 1e-20 ? (float) (L.d[tau] * tau / run) : 1.f;
		}
		const float thr = 0.15f;
		int lo = std::max(2, (int) std::floor(L.tauLo)), hi = std::min(tauMax - 1, (int) std::ceil(L.tauHi));
		if (hi <= lo + 2) {
			L.est = e;
			return;
		}
		int found = -1;
		for (int tau = lo; tau <= hi; tau++) {
			if (L.cm[tau] < thr) {
				while (tau + 1 <= hi && L.cm[tau + 1] < L.cm[tau])
					tau++;
				found = tau;
				break;
			}
		}
		if (found < 0) {
			// no dip below threshold: take the global minimum if reasonably periodic
			float best = 1e9f;
			for (int tau = lo; tau <= hi; tau++)
				if (L.cm[tau] < best) {
					best = L.cm[tau];
					found = tau;
				}
			if (best > 0.4f)
				found = -1;
		}
		if (found < 0) {
			L.est = e;
			return;
		}
		double off, val;
		parabolic(L.cm[found - 1], L.cm[found], L.cm[found + 1], off, val);
		double tau1 = found + off;
		// multiplicity test (sub oscillator => the true repeating unit is longer)
		int mult = 1;
		float c1 = std::max(0.f, (float) val);
		double bestScore = c1;
		double bestTau = tau1;
		for (int m = 2; m <= 2; m++) {
			double t = tau1 * m;
			int ti = (int) std::round(t);
			if (ti + 1 > tauMax)
				break;
			// local minimum near m*tau1 (allow +-1.5 samples for period estimation error)
			int bi = ti;
			float bv = 1e9f;
			for (int q = std::max(2, ti - 2); q <= std::min(tauMax - 1, ti + 2); q++)
				if (L.cm[q] < bv) {
					bv = L.cm[q];
					bi = q;
				}
			double o2, v2;
			parabolic(L.cm[bi - 1], L.cm[bi], L.cm[bi + 1], o2, v2);
			// the longer period must be *substantially* better than the shorter one and the shorter one must be imperfect
			if (c1 > 0.02f && v2 < 0.5 * c1 && v2 < bestScore) {
				mult = m;
				bestScore = v2;
				bestTau = (bi + o2);
			}
		}
		e.valid = true;
		e.mult = mult;
		e.conf = clampT(1.f - (float) bestScore, 0.f, 1.f);
		e.period = bestTau * L.decim;
		L.est = e;
	}

	void choose() {
		// Among the lanes with a valid, in-range estimate keep those whose confidence is within 0.12 of the best and pick the
		// one with the *shortest* unit period: a lane that cannot see the true period (because it is above its range) can only
		// answer with an integer multiple of it, so the shortest consistent period is the fundamental (classic YIN rule).
		float bestConf = -1.f;
		for (int i = 0; i < kLanes; i++) {
			const Estimate& e = lane_[i].est;
			if (inRange(i, e))
				bestConf = std::max(bestConf, e.conf);
		}
		int best = -1;
		double bestPeriod = 1e30;
		for (int i = 0; i < kLanes; i++) {
			const Estimate& e = lane_[i].est;
			if (!inRange(i, e) || e.conf < bestConf - 0.12f)
				continue;
			if (e.period < bestPeriod) {
				bestPeriod = e.period;
				best = i;
			}
		}
		Estimate out;
		out.seq = est_.seq + 1;
		if (best >= 0) {
			out = lane_[best].est;
			out.seq = est_.seq + 1;
			out.lane = best;
		}
		est_ = out;
	}

	bool inRange(int i, const Estimate& e) const {
		if (!e.valid || !(activeMask_ & (1 << i)))
			return false;
		double f = fs_ / e.period; // unit frequency
		return f >= lane_[i].fLo * 0.9 && f <= lane_[i].fHi * 1.1;
	}

	double fs_ = 48000.0;
	Lane lane_[kLanes];
	Estimate est_;
	int activeMask_ = (1 << kLanes) - 1;
	float silenceThresh_ = 1e-6f;
	float dc_ = 0.f, level2_ = 0.f, dcCoef_ = 0.f, lvlRise_ = 0.f, lvlFall_ = 0.f;
};

} // namespace fc
