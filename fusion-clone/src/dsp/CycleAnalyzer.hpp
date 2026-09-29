// FusionClone DSP core — period-synchronous harmonic analysis with an open-loop frequency measurement and a Kalman tracker.
//
// The source is assumed to be a (quasi-)periodic monophonic oscillator. Given a period estimate we:
//   1. resample the last M table periods of the input onto a uniform *angle* grid (Nc points per period) with a
//      windowed-sinc reader (this is "computed order tracking": the signal becomes exactly periodic in angle even if
//      the pitch wobbles or glides — the grid follows the tracked frequency *and its slope*),
//   2. Hann-window and FFT: with an integer number of periods in the window every harmonic j falls on bin M*j and the
//      Hann window has exact zeros at +-2 bins, so neighbouring harmonics do not leak into each other,
//   3. read the complex harmonic coefficients c_j directly from those bins (no peak picking, no bin quantisation),
//   4. measure the time shift between this window and the previous one from the magnitude-weighted phase *slope* across
//      harmonics (phase of c_j / c_j_prev = 2*pi*j*delta) — a sub-sample, all-harmonics time-shift estimator,
//   5. turn that shift into a *frequency measurement*. A symmetric (Hann) window reports the source phase at its centre, so the phase
//      difference between two windows is the phase the source advanced between the two window centres:
//          nu_bar = (delta + dTheta - dA) / (t_c,k - t_c,k-1)
//      with dTheta the advance of the reference phase and t_c the (grid dependent) window centre times. This is an exact, memory-less
//      inversion — no feedback loop, hence no loop bandwidth / dead-time trade-off (the previous PLL had ~2 Hz bandwidth and could not
//      follow a 5 Hz vibrato at low pitch),
//      The grid is warped with a smoothed copy of the tracked frequency slope so that a glide stays sharp inside the window. A wrong warp
//      biases the window-centre phase by 1/2 (kappa_true - kappa_grid) <b^2>_w (b = cycles from the window centre, kappa = slope / freq^2), which
//      would close a feedback loop from the tracker's own slope estimate back into its measurements (unstable for large M*Q). The bias that
//      the *applied* warp contributes is known exactly and is removed from the measurement, which makes the measurement independent of the
//      warp; only the (small, feedback-free) bias of the true chirp remains.
//      A grid frequency that is off by a relative error e moves harmonic j by M*j*e bins away from its bin centre; beyond ~0.15 bin the Hann
//      leakage of the neighbouring harmonics corrupts the phase of that bin. The phase-slope estimate therefore only uses harmonics up to
//      j_max = 0.15 / (M e), with e the recently observed prediction error of the tracker (start: 2 % -> 4 harmonics, steady: all 96).
//      This makes the measurement self-consistent at every accuracy level, so the tracker bootstraps from a coarse start value.
//      The measurement still carries a small error that repeats with the waveform (leakage of the neighbouring harmonics into a bin is first
//      order in the grid error and depends on where in the cycle the window sits). Because hops are placed at exact multiples of 1/Q of a
//      cycle, the mean of the last Q measurements — exactly one waveform period — cancels that ripple, and the slope between successive
//      such means is the change over one period, which cancels it too. The filter is fed these period means.
//   6. feed the measurement (with its own noise estimate from the phase-slope residual) to a two-state Kalman filter (frequency and
//      frequency slope). The filter predicts the frequency at any instant, in particular at the window centre of the next analysis (which
//      keeps the angular grid sharp) and at "now" (which the engine uses to drive the clone oscillators).
//
// "Table period" = the repeating unit reported by the PitchTracker (one main cycle, or two when a sub-oscillator makes the
// composite repeat every 2 cycles). Harmonic j of the table has frequency j / period.
#pragma once
#include <memory>
#include "Common.hpp"
#include "FFT.hpp"
#include "SincInterp.hpp"

namespace fc {

struct HarmonicSet {
	int J = 0;                       // number of valid harmonics (1..J)
	std::vector<float> re, im;       // complex coefficients c_j (real signal amplitude of harmonic j = 2|c_j|)
	float c0 = 0.f;                  // DC term
	double thetaEnd = 0.0;           // content-anchored source phase (table cycles) at the window end
	double refTime = 0.0;            // absolute sample index (fractional) of the window end
	float rms = 0.f;                 // RMS of the periodic model
	float coherence = 0.f;           // cross-hop coherence 0..1
	float periodicity = 0.f;         // fraction of window energy explained by the harmonic model (0..1)
	double period = 0.0;             // table period in samples used for this set (at the window centre)
	int Nc = 0;
	uint32_t seq = 0;
};

class CycleAnalyzer {
public:
	struct Config {
		int M = 2;            // periods per analysis window (2, 4 or 8)
		int minNc = 64;
		int maxNc = 4096;
		int taps = 16;        // 16 or 32 sinc taps
		double minHopSec = 0.0015;
		double maxHopSec = 0.004;
		/** Agility of the frequency tracker: standard deviation of the frequency-slope change over one second, relative to the frequency
		    (1/s^2). A 5 Hz +-15 cent vibrato needs ~9, 8 Hz +-30 cent ~45; the tracker never trusts a noisier measurement than its own
		    residual-based noise estimate says, so a large value costs nothing on clean signals. */
		double accelRel = 25.0;
	};

	/** Allocate for the largest configuration that will ever be used (call from a non-realtime context). */
	void prepare(double fs, const Config& maxCfg) {
		fs_ = fs;
		maxCfg_ = maxCfg;
		fft_.clear();
		fftSizes_.clear();
		// plans for every window length L = M*Nc that any configuration with M in {2,4,8} can need
		int maxL = 8 * maxCfg.maxNc;
		maxL = std::min(maxL, 4 * maxCfg.maxNc > 32768 ? 32768 : 4 * maxCfg.maxNc);
		for (int L = 2 * maxCfg.minNc; L <= maxL; L <<= 1) {
			fft_.push_back(std::unique_ptr<RealFFT>(new RealFFT(L)));
			fftSizes_.push_back(L);
		}
		grid_.alloc(maxL);
		spec_.alloc(maxL);
		win_.assign(maxL, 0.f);
		curWinN_ = 0;
		int maxJ = maxCfg.maxNc / 2;
		cur_.re.assign(maxJ + 2, 0.f);
		cur_.im.assign(maxJ + 2, 0.f);
		prev_.re.assign(maxJ + 2, 0.f);
		prev_.im.assign(maxJ + 2, 0.f);
		tmpRe_.assign(maxJ + 2, 0.f);
		tmpIm_.assign(maxJ + 2, 0.f);
		configure(maxCfg);
	}

	/** Change the runtime configuration (window periods, kernel, hop rate, max table size) without allocating. */
	void configure(const Config& cfg) {
		cfg_ = cfg;
		cfg_.maxNc = std::min(cfg.maxNc, maxCfg_.maxNc);
		M_ = cfg.M;
		lag_ = cfg.taps / 2 + 2;
		curWinN_ = 0;
		reset();
	}

	void reset() {
		active_ = false;
		nu_ = 0.0;
		nuDot_ = 0.0;
		theta_ = 0.0;
		delta_cum_ = 0.0;
		havePrev_ = false;
		hopCount_ = 0;
		cur_.seq = 0;
		coh_ = 0.f;
		goodHops_ = 0;
	}

	/** Start (or restart) tracking with a table period in samples. `nowSample` is the ring's count() at the call. */
	void start(double periodSamples, uint64_t nowSample) {
		nu_ = nuGood_ = 1.0 / periodSamples;
		rejected_ = 0;
		nuDot_ = nuDotWarp_ = 0.0;
		prevBias_ = 0.0;
		histPos_ = histCount_ = 0;
		tState_ = (double) nowSample - 1.0;
		nowT_ = tState_;
		// the start value comes from a coarse search: allow 2 % frequency error and a slope of up to 50 %/s
		p00_ = sq(0.02 * nu_);
		p01_ = 0.0;
		p11_ = sq(0.5 * nu_ / fs_);
		epsRel_ = 0.02;
		theta_ = 0.0;
		delta_cum_ = 0.0;
		lastSample_ = nowSample;
		havePrev_ = false;
		hopCount_ = 0;
		goodHops_ = 0;
		coh_ = 0.f;
		active_ = true;
		int Q = hopsPerPeriod(periodSamples);
		Q_ = Q;
		nextHop_ = std::ceil(theta_ * Q) / Q; // hop boundaries are at multiples of 1/Q in theta
	}

	void stop() { active_ = false; }
	bool active() const { return active_; }

	/** Advance the tracker by one sample. Must be called after `ring.push()` for that sample. Returns true when a new
	    harmonic set (available via set()) was produced. */
	bool step(const MirrorRing& ring) {
		if (!active_)
			return false;
		nowT_ = (double) ring.count() - 1.0;
		const double om = omegaAt(nowT_);
		theta_ += om; // the reference phase follows the predicted frequency, so successive windows differ by only a tiny shift
		lastSample_ = ring.count();
		// the reference instant lags the newest sample by lag_ samples (the sinc reader needs future samples)
		const double thetaRef = theta_ - lag_ * om;
		if (thetaRef < nextHop_)
			return false;
		bool ok = analyse(ring, thetaRef);
		int Q = hopsPerPeriod(1.0 / omega());
		Q_ = Q;
		nextHop_ = (std::floor(thetaRef * Q + 1e-9) + 1.0) / Q;
		return ok;
	}

	const HarmonicSet& set() const { return cur_; }
	/** Frequency (table cycles per sample) predicted for absolute sample time t. The extrapolation from the last measurement is bounded to
	    +-3 % so a stalled measurement stream cannot run away. */
	double omegaAt(double t) const {
		double d = nuDot_ * (t - tState_);
		const double lim = 0.03 * nu_;
		d = d > lim ? lim : (d < -lim ? -lim : d);
		return nu_ + d;
	}
	/** Frequency the analysis grid is built on: the tracker state extrapolated with the *smoothed* slope only. The fast slope amplifies the
	    small window-position dependent bias of the measurement (leakage of neighbouring harmonics into a bin, first order in the grid error) and
	    would close an unstable loop at M = 2 and low pitch; the smoothed slope keeps that loop gain well below one. */
	double omegaGridAt(double t) const {
		double d = nuDotWarp_ * (t - tState_);
		const double lim = 0.03 * nu_;
		d = d > lim ? lim : (d < -lim ? -lim : d);
		return nu_ + d;
	}
	/** Predicted frequency at the newest sample. */
	double omega() const { return omegaAt(nowT_); }
	double period() const { const double o = omega(); return o > 0 ? 1.0 / o : 0.0; }
	/** Frequency slope (cycles per sample^2). */
	double slope() const { return nuDot_; }
	/** Content-anchored source phase (table cycles) at the newest sample. */
	double thetaNow() const { return theta_ + delta_cum_; }
	float coherence() const { return coh_; }
	int goodHops() const { return goodHops_; }
	int hopsPerPeriod(double period) const {
		double periodSec = period / fs_;
		int Q = (int) std::floor(periodSec / cfg_.maxHopSec);
		return clampT(Q, 1, 8);
	}
	int lagSamples() const { return lag_; }
	int windowPeriods() const { return M_; }
	/** Last per-hop time shift between successive windows in table cycles (diagnostic). */
	double lastDelta() const { return lastDelta_; }
	/** Last frequency measurement (cycles/sample) and its standard deviation (diagnostics). */
	double lastMeasurement() const { return lastNuBar_; }
	double lastMeasurementSigma() const { return lastNuSigma_; }
	double lastMeasurementTime() const { return lastNuTime_; }
	/** Number of discarded measurements / numerical resets since start() (diagnostic). */
	int rejectedMeasurements() const { return rejected_; }
	double dbgDtc = 0, dbgDTheta = 0, dbgTc = 0, dbgNuC = 0, dbgAc = 0; int dbgJMax = 0; bool dbgFreeze = false;
	/** Test hook: overwrite the tracker state (frequency in cycles/sample, slope in cycles/sample^2) valid from absolute sample time t. */
	void dbgSetState(double nu, double nuDot, double t) { nu_ = nu; nuDot_ = nuDot; nuDotWarp_ = nuDot; tState_ = t; }
	int currentNc() const { return cur_.Nc; }

private:
	static inline double sq(double x) { return x * x; }

	/** Multiply harmonic j by exp(j*2*pi*j*cycles). */
	static void rotateSet(float* re, float* im, int J, double cycles) {
		if (cycles == 0.0)
			return;
		for (int j = 1; j <= J; j++) {
			double a = kTwoPi * frac((double) j * cycles);
			float cr = (float) std::cos(a), ci = (float) std::sin(a);
			float r = re[j] * cr - im[j] * ci, i = re[j] * ci + im[j] * cr;
			re[j] = r;
			im[j] = i;
		}
	}

	/** Fraction of the windowed energy that the harmonic model explains. Non-harmonic bins are cleaned of the Hann leakage that
	    the neighbouring harmonic bins predict (bin +-1 gets -0.5 * X[M*j] for a periodic signal), so what remains is genuinely
	    non-periodic energy: noise, detune sidebands, modulation. Result is 1 for a perfectly periodic input, ~0.5 for noise. */
	float residualPeriodicity(int J) const {
		const int M = M_;
		double eH = 0.0, eR = 0.0;
		for (int j = 1; j <= J; j++) {
			int b = M * j;
			eH += 1.5 * ((double) spec_[2 * b] * spec_[2 * b] + (double) spec_[2 * b + 1] * spec_[2 * b + 1]);
		}
		int bins = 0;
		for (int j = 0; j <= J; j++) {
			for (int m = 1; m < M; m++) {
				int b = M * j + m;
				if (b >= curWinN_ / 2)
					break;
				double pr = 0, pi = 0;
				if (m == 1 && j >= 1) {
					pr += -0.5 * spec_[2 * (M * j)];
					pi += -0.5 * spec_[2 * (M * j) + 1];
				} else if (m == 1 && j == 0) {
					pr += -0.5 * spec_[0]; // DC (packed layout: spec_[0] = Re F(0), imaginary part 0) leaks into bin 1 like any harmonic
				}
				if (m == M - 1 && j + 1 <= J) {
					pr += -0.5 * spec_[2 * (M * (j + 1))];
					pi += -0.5 * spec_[2 * (M * (j + 1)) + 1];
				}
				double rr = spec_[2 * b] - pr, ri = spec_[2 * b + 1] - pi;
				eR += rr * rr + ri * ri;
				bins++;
			}
		}
		if (bins == 0)
			return 1.f;
		double eNon = eR * (double) M / (double) (M - 1); // spread the measured off-harmonic energy over all bins
		return (float) (eH / (eH + eNon + 1e-30));
	}

	RealFFT* fftFor(int L) {
		for (size_t i = 0; i < fftSizes_.size(); i++)
			if (fftSizes_[i] == L)
				return fft_[i].get();
		return nullptr;
	}

	/** Kalman update of (nu, nuDot) with a frequency measurement `z` (cycles/sample, variance R) that refers to time tMeas. */
	void kalmanUpdate(double z, double R, double tMeas) {
		const double dt = tMeas - tState_;
		if (dt < 0.0)
			return;
		// sanity: a non-finite measurement, or one 25 % away from the tracked frequency (impossible within a hop for a physical oscillator; a
		// pitch step is handled by the engine's transient detector, which restarts the analysis), is discarded
		if (!(z > 0.0) || !(R >= 0.0) || std::fabs(z - nu_) > 0.25 * nu_) {
			rejected_++;
			return;
		}
		nu_ += nuDot_ * dt; // predict the state to the measurement time
		// white-acceleration process noise; sigma_a (cycles/sample^2 per sqrt(s)) scales with the frequency
		const double sa = cfg_.accelRel * nu_ / fs_;
		const double qs = sa * sa / fs_;
		const double p00 = p00_ + 2.0 * dt * p01_ + dt * dt * p11_ + qs * dt * dt * dt / 3.0;
		const double p01 = p01_ + dt * p11_ + qs * dt * dt / 2.0;
		const double p11 = p11_ + qs * dt;
		const double rho = z - nu_;
		epsRel_ = std::max(1.5 * std::fabs(rho) / nu_, 0.9 * epsRel_); // observed prediction error, peak-held (sets the usable harmonic range)
		double S = p00 + R;
		// robustness: never trust a single measurement further than 5 sigma from the prediction (inflate its variance instead)
		if (rho * rho > 25.0 * S)
			S = rho * rho / 25.0;
		const double k0 = p00 / S, k1 = p01 / S;
		nu_ += k0 * rho;
		nuDot_ += k1 * rho;
		p00_ = (1.0 - k0) * p00;
		p01_ = (1.0 - k0) * p01;
		p11_ = p11 - k1 * p01;
		if (p00_ < 1e-30) p00_ = 1e-30; // keep the covariance positive definite against round-off
		if (p11_ < 1e-40) p11_ = 1e-40;
		tState_ = tMeas;
		if (!(nu_ > 0.0) || !(nuDot_ == nuDot_)) { // numerical failure: fall back to the last good frequency with a wide covariance
			nu_ = nuGood_;
			nuDot_ = nuDotWarp_ = 0.0;
			p00_ = sq(0.02 * nu_);
			p01_ = 0.0;
			p11_ = sq(0.5 * nu_ / fs_);
			rejected_++;
		} else {
			nuGood_ = nu_;
		}
	}

	bool analyse(const MirrorRing& ring, double thetaRef) {
		// ---- window geometry: Nc, centre time and centre frequency -------------------------------------------------------
		const double T = (double) (ring.count() - 1) - lag_; // window end: `lag_` samples before the newest sample
		const double nuEnd = omegaGridAt(T);
		if (!(nuEnd > 0.0))
			return false;
		// slope used for the grid warp: the tracker's slope averaged over about one analysis window (see the header note on feedback)
		{
			// about one analysis window of hops, but never faster than 4 hops: the measurement error has a small component proportional to the
			// hop-to-hop *change* of the grid frequency (worst at unlucky waveform alignments, e.g. a narrow pulse), and the fast slope amplifies
			// that alternating component (period 2 hops) into a slowly growing oscillation unless it is attenuated here
			const double kw = std::min(0.25, 2.0 / ((double) M_ * (double) std::max(1, Q_)));
			nuDotWarp_ += (nuDot_ - nuDotWarp_) * kw;
		}
		const double nuApprox = omegaGridAt(T - 0.5 * M_ / nuEnd);
		const double P0 = 1.0 / nuApprox;
		int Nc = nextPow2((int) std::ceil(1.15 * P0));
		Nc = clampT(Nc, cfg_.minNc, cfg_.maxNc);
		const int L = M_ * Nc;
		RealFFT* fft = fftFor(L);
		if (!fft)
			return false;
		// The Hann window is periodic (centre at grid index L/2). Grid index i sits b_i = (L-1-i)/Nc - aC table cycles before the window
		// centre, so aC = M/2 - 1/Nc cycles separate the window end from its centre.
		const double aC = 0.5 * M_ - 1.0 / Nc;
		// self-consistent centre: with a linear frequency ramp the end lies s_end = 2 aC / (nu_c + sqrt(nu_c^2 + 2 nuDot aC)) after the centre
		double tc = T - aC / nuEnd;
		double nuC = omegaGridAt(tc);
		const double bMax = 0.5 * M_; // largest |b| (window start)
		// the warp is only applied when it moves grid points by more than 0.02 samples; otherwise the grid is uniform (slope 0)
		const bool warp = std::fabs(nuDotWarp_) * bMax * bMax / (2.0 * nuEnd * nuEnd * nuEnd) > 0.02;
		const double chirp = warp ? nuDotWarp_ : 0.0;
		for (int it = 0; it < 3; it++) {
			nuC = omegaGridAt(tc);
			double disc = nuC * nuC + 2.0 * chirp * aC;
			if (disc < 0.25 * nuC * nuC)
				disc = 0.25 * nuC * nuC;
			tc = T - 2.0 * aC / (nuC + std::sqrt(disc));
		}
		nuC = omegaGridAt(tc);
		const double P = 1.0 / nuC; // table period at the window centre (samples)
		// grid position of index i: t = tc + s(b), s(b) = -2 b / (nu_c + sqrt(nu_c^2 - 2 chirp b)); reduces to -b/nu_c for chirp = 0
		double disc0 = nuC * nuC - 2.0 * chirp * bMax;
		if (disc0 < 0.25 * nuC * nuC)
			disc0 = 0.25 * nuC * nuC;
		const double tFirst = tc - 2.0 * bMax / (nuC + std::sqrt(disc0));
		const double oldest = (double) ring.count() - (double) ring.size() + 64.0;
		if (tFirst - lag_ - 2 < oldest)
			return false; // not enough history yet
		if (warp) {
			for (int i = 0; i < L; i++) {
				const double b = (double) (L - 1 - i) / Nc - aC;
				double d = nuC * nuC - 2.0 * chirp * b;
				if (d < 0.25 * nuC * nuC)
					d = 0.25 * nuC * nuC;
				const double pos = tc - 2.0 * b / (nuC + std::sqrt(d));
				grid_[i] = (cfg_.taps == 32) ? ring.readSinc<32>(pos) : ring.readSinc<16>(pos);
			}
		} else {
			const double step = P / Nc; // samples per grid point
			const double t0 = tc - ((double) (L - 1) / Nc - aC) * P;
			if (cfg_.taps == 32) {
				for (int i = 0; i < L; i++)
					grid_[i] = ring.readSinc<32>(t0 + i * step);
			} else {
				for (int i = 0; i < L; i++)
					grid_[i] = ring.readSinc<16>(t0 + i * step);
			}
		}
		if (curWinN_ != L) {
			for (int i = 0; i < L; i++)
				win_[i] = hannPeriodic(i, L);
			curWinN_ = L;
		}
		double energy = 0.0;
		for (int i = 0; i < L; i++) {
			energy += (double) grid_[i] * grid_[i];
			grid_[i] *= win_[i];
		}
		fft->forward(grid_.data(), spec_.data());

		const int J = std::min(Nc / 2 - 1, std::max(1, (int) std::floor(0.49 * P)));
		const float periodicity = residualPeriodicity(J);
		const float W0 = 0.5f * L;
		// harmonic coefficients in the raw frame: the phase of c_j is 2*pi*j*tau with tau = (source phase at the window centre) - (reference
		// phase there), the reference phase being the tracker phase theta at the window end minus aC cycles
		double thetaFrac = frac(thetaRef);
		for (int j = 1; j <= J; j++) {
			int b = M_ * j;
			float xr = spec_[2 * b], xi = spec_[2 * b + 1];
			double ang = -kTwoPi * frac((double) j * (thetaFrac + 1.0 / Nc));
			float cr = (float) std::cos(ang), ci = (float) std::sin(ang);
			tmpRe_[j] = (xr * cr - xi * ci) / W0;
			tmpIm_[j] = (xr * ci + xi * cr) / W0;
		}
		double c0 = spec_[0] / (double) (W0 * 2.0);

		// ---- alignment against the previous hop ---------------------------------------------------------------------
		// Bring the raw set into the content-anchored frame using the offset accumulated so far, then measure only the
		// *increment* against the previous (already anchored) set.
		rotateSet(tmpRe_.data(), tmpIm_.data(), J, -delta_cum_);
		double delta = 0.0, sigmaDelta = 1.0;
		float coh = 0.f;
		if (havePrev_ && prevJ_ > 0) {
			int Jc = std::min(J, prevJ_);
			// magnitude-weighted phase-slope estimate with incremental unwrapping (low harmonics first), limited to the harmonics that are
			// still on their bins given the tracker's recent prediction error
			const int jMax = clampT((int) (0.15 / (M_ * std::max(epsRel_, 1e-9))), 4, 96);
			dbgJMax = jMax;
			double sxy = 0.0, sxx = 0.0;
			double est = 0.0;
			int used = 0;
			double maxMag = 0.0;
			for (int j = 1; j <= Jc; j++)
				maxMag = std::max(maxMag, (double) tmpRe_[j] * tmpRe_[j] + (double) tmpIm_[j] * tmpIm_[j]);
			const double magFloor = maxMag * 1e-5;
			for (int j = 1; j <= Jc && j <= jMax; j++) {
				double ar = tmpRe_[j], ai = tmpIm_[j], br = prev_.re[j], bi = prev_.im[j];
				double mag2a = ar * ar + ai * ai, mag2b = br * br + bi * bi;
				if (mag2a < magFloor || mag2b < magFloor)
					continue;
				// c_now * conj(c_prev): phase = 2*pi*j*delta for a pure time shift
				double pr = ar * br + ai * bi, pi = ai * br - ar * bi;
				double meas = std::atan2(pi, pr);
				double pred = kTwoPi * j * est;
				double resid = meas - pred;
				resid -= kTwoPi * std::floor(resid / kTwoPi + 0.5);
				double unwrapped = pred + resid;
				double w = std::sqrt(mag2a * mag2b);
				double x = kTwoPi * j;
				sxy += w * x * unwrapped;
				sxx += w * x * x;
				est = sxy / sxx;
				used++;
			}
			delta = est;
			// noise of the slope estimate: weighted least squares residual (weights ~ 1/phase variance) and coherence after removing the shift
			double swr = 0.0, sw = 0.0;
			double nr = 0, ni = 0, ea = 0, eb = 0;
			int n = 0;
			for (int j = 1; j <= Jc; j++) {
				double ar = tmpRe_[j], ai = tmpIm_[j], br = prev_.re[j], bi = prev_.im[j];
				double a = -kTwoPi * j * delta;
				double cr = std::cos(a), ci = std::sin(a);
				double rr = ar * cr - ai * ci, ri = ar * ci + ai * cr;
				double pr = rr * br + ri * bi, pi = ri * br - rr * bi; // c_now(shifted) * conj(c_prev): its phase is the residual
				double mag2a = ar * ar + ai * ai, mag2b = br * br + bi * bi;
				if (j <= jMax && mag2a >= magFloor && mag2b >= magFloor) { // same harmonics as the slope estimate above
					double w = std::sqrt(mag2a * mag2b);
					double r = std::atan2(pi, pr);
					swr += w * r * r;
					sw += w;
					n++;
				}
				nr += pr;
				ni += pi;
				ea += ar * ar + ai * ai;
				eb += br * br + bi * bi;
			}
			coh = (float) (std::sqrt(nr * nr + ni * ni) / (std::sqrt(ea * eb) + 1e-30));
			if (n >= 2 && sxx > 0.0) {
				// var(est) = N0 / sum(w x^2) with N0 = sum(w r^2)/(n-1); w in the same units on both sides
				sigmaDelta = std::sqrt((swr / (double) (n - 1)) / sxx);
			} else {
				sigmaDelta = 1.0 / (kTwoPi + 1e-9); // one harmonic or none: little information
			}
			sigmaDelta = std::max(sigmaDelta, 3e-7); // float32 FFT precision floor
			// hop-to-hop deltas beyond a fraction of a period are not physical -> treat as incoherent
			if (std::fabs(delta) > 0.25)
				coh = 0.f;
		}
		lastDelta_ = delta;

		// ---- frequency measurement + tracker update -------------------------------------------------------------------
		bool good = havePrev_ && coh > 0.5f;
		const double b2 = (double) M_ * M_ * (1.0 / 12.0 - 1.0 / (2.0 * kPi * kPi)); // <b^2> of the periodic Hann window
		const double biasNow = 0.5 * b2 * (chirp / (nuC * nuC));
		if (good) {
			const double dtc = tc - prevTc_;   // samples between the two window centres
			const double dTheta = thetaRef - prevThetaRef_; // reference phase advance between the two window ends
			const double dA = aC - prevAc_;    // window-centre offset change (Nc changes)
			if (dtc > 1.0 && dTheta > 1e-6) {
				// source phase advance between the centres = delta + (reference advance) - (offset change), all in table cycles; the centre phase
				// carries 1/2 (kappa_true - kappa_grid) <b^2>_w of chirp bias, of which the applied-warp part is known and removed here
				const double nuBar = (delta + (biasNow - prevBias_) + dTheta - dA) / dtc;
				// two independent windows enter the difference; energy the harmonic model cannot explain (a sub oscillator that has just
				// appeared, heavy modulation) corrupts the bins, so trust the measurement less the lower the periodicity index is
				const double pd = 1.0 - (double) periodicity;
				const double sig = sigmaDelta * 1.4142 * std::sqrt(1.0 + 400.0 * pd * pd) / dtc;
				lastNuBar_ = nuBar;
				lastNuSigma_ = sig;
				lastNuTime_ = 0.5 * (tc + prevTc_);
				dbgDtc = dtc; dbgDTheta = dTheta; dbgTc = tc; dbgNuC = nuC; dbgAc = aC;
				// sliding mean over one waveform period (the last Q hops)
				histNu_[histPos_] = nuBar;
				histT_[histPos_] = 0.5 * (tc + prevTc_);
				histVar_[histPos_] = sig * sig;
				histPos_ = (histPos_ + 1) % kHist;
				if (histCount_ < kHist) histCount_++;
				const int q = std::min(std::max(Q_, 1), histCount_);
				double mz = 0.0, mt = 0.0, mv = 0.0;
				for (int h = 1; h <= q; h++) {
					const int idx = (histPos_ - h + kHist) % kHist;
					mz += histNu_[idx];
					mt += histT_[idx];
					mv += histVar_[idx];
				}
				mz /= q; mt /= q; mv /= q;
				if (!dbgFreeze && (histCount_ >= Q_ || Q_ <= 1))
					kalmanUpdate(mz, mv, mt);
			}
			delta_cum_ += delta; // the content-anchored frame absorbs the measured shift
			goodHops_++;
			rotateSet(tmpRe_.data(), tmpIm_.data(), J, -delta);
		} else if (havePrev_) {
			goodHops_ = 0;
		}
		coh_ = coh;
		prevTc_ = tc;
		prevAc_ = aC;
		prevBias_ = biasNow;

		// publish (tmp now holds the aligned set)
		cur_.J = J;
		cur_.Nc = Nc;
		cur_.period = P;
		for (int j = 1; j <= J; j++) {
			cur_.re[j] = tmpRe_[j];
			cur_.im[j] = tmpIm_[j];
		}
		cur_.c0 = (float) c0;
		cur_.thetaEnd = thetaRef + delta_cum_;
		cur_.refTime = T;
		double e2 = 0;
		for (int j = 1; j <= J; j++)
			e2 += 2.0 * ((double) cur_.re[j] * cur_.re[j] + (double) cur_.im[j] * cur_.im[j]);
		cur_.rms = (float) std::sqrt(e2);
		cur_.coherence = coh;
		cur_.periodicity = periodicity;
		cur_.seq++;

		// keep the aligned set as the reference for the next hop
		prev_.re = cur_.re;
		prev_.im = cur_.im;
		prevJ_ = J;
		prevThetaRef_ = thetaRef;
		havePrev_ = true;
		hopCount_++;
		(void) energy;
		return true;
	}

	double fs_ = 48000.0;
	Config cfg_, maxCfg_;
	int M_ = 2;
	std::vector<std::unique_ptr<RealFFT> > fft_;
	std::vector<int> fftSizes_;
	AlignedBuffer<float> grid_, spec_;
	std::vector<float> win_, tmpRe_, tmpIm_;
	int curWinN_ = 0;
	int lag_ = 10;

	bool active_ = false;
	// frequency tracker state: nu (cycles/sample) and nuDot (cycles/sample^2) valid at absolute sample time tState_, covariance p00/p01/p11
	double nu_ = 0.0, nuDot_ = 0.0, tState_ = 0.0, p00_ = 0.0, p01_ = 0.0, p11_ = 0.0;
	double nowT_ = 0.0;
	double theta_ = 0.0, delta_cum_ = 0.0, nextHop_ = 0.0, lastDelta_ = 0.0;
	double lastNuBar_ = 0.0, lastNuSigma_ = 0.0, lastNuTime_ = 0.0;
	double prevTc_ = 0.0, prevAc_ = 0.0, prevBias_ = 0.0, nuDotWarp_ = 0.0, epsRel_ = 0.02;
	double nuGood_ = 0.0;
	int rejected_ = 0;
	static const int kHist = 16;
	double histNu_[kHist], histT_[kHist], histVar_[kHist];
	int histPos_ = 0, histCount_ = 0;
	uint64_t lastSample_ = 0;
	int Q_ = 1;
	bool havePrev_ = false;
	int prevJ_ = 0;
	double prevThetaRef_ = 0.0;
	int hopCount_ = 0, goodHops_ = 0;
	float coh_ = 0.f;
	HarmonicSet cur_, prev_;
};

} // namespace fc
