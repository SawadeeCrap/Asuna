// FusionClone DSP core — the cloning engine.
//
//   input ──┬─────────────────────────────────────────────► original path (direct, zero latency) ─┐
//           │                                                                                    ├─► sum ─► summing ─► mix ─► out
//           ├─► PitchTracker (5 decimated YIN lanes) ─► period acquisition                        │
//           │        │                                                                             │
//           └─► CycleAnalyzer (angular resampling, Hann-FFT, harmonic coefficients, phase-slope PLL)
//                    │  aligned complex harmonic set c_j, tracked frequency omega
//                    ▼
//              per-voice period tables (IFFT of c_j * per-voice divergence, anti-aliased for the voice's own ratio)
//                    ▼
//              N-1 free-running wavetable oscillators: phase += omega * ratio_v(t)   ─────────────► clones
//
// The clones "bloom in" once the periodic structure is confirmed (a few periods after an onset or pitch step); until
// then the original is boosted so the total power stays constant. Everything is deterministic given the seed and nothing
// allocates after prepare().
#pragma once
#include <memory>

#include "CycleAnalyzer.hpp"
#include "Filters.hpp"
#include "Params.hpp"
#include "PitchTracker.hpp"
#include "Voice.hpp"

namespace fc {

class Engine {
public:
	Engine() {}

	/** (Re)initialise for a sample rate. Allocates everything for the largest configuration; call from a non-realtime context. */
	void prepare(double sampleRate) {
		fs_ = sampleRate;
		tracker_.prepare(fs_);
		CycleAnalyzer::Config maxCfg;
		maxCfg.M = 4;
		maxCfg.taps = 32;
		maxCfg.maxNc = 8192;
		maxCfg.minNc = 64;
		analyzer_.prepare(fs_, maxCfg);
		ifft_.clear();
		ifftN_.clear();
		for (int n = 64; n <= 8192; n <<= 1) {
			ifft_.push_back(std::unique_ptr<RealFFT>(new RealFFT(n)));
			ifftN_.push_back(n);
		}
		specBuf_.alloc(8192);
		tabBuf_.alloc(8192);
		avgRe_.assign(8192 / 2 + 2, 0.f);
		avgIm_.assign(8192 / 2 + 2, 0.f);
		pubRe_.assign(8192 / 2 + 2, 0.f);
		pubIm_.assign(8192 / 2 + 2, 0.f);
		for (int v = 1; v < kMaxVoices; v++) {
			vs_[v].tab[0].alloc(8192);
			vs_[v].tab[1].alloc(8192);
		}
		int ringSize = nextPow2((int) (fs_ * 1.0)) * 2;
		ring_.alloc(std::max(ringSize, 131072));
		sq_.assign(65536, 0.f);
		xr_.assign(65536, 0.f);
		sqMask_ = 65535;
		dcCoef_ = 1.f - (float) std::exp(-1.0 / (0.1 * fs_));
		refineBuf_.assign(4096, 0.f);
		outGain_.setTimeConstant(0.02f, (float) fs_);
		norm_.setTimeConstant(0.02f, (float) fs_);
		applyQuality(params_.quality);
		reseed(params_.seed);
		paramsSeen_ = false;
		resetState();
	}

	/** Realtime-safe. Parameters take effect smoothly; a quality change drops the current lock (no allocation). */
	void setParams(const EngineParams& p) {
		bool qChanged = p.quality != params_.quality;
		bool seedChanged = p.seed != params_.seed;
		params_ = p;
		params_.voices = clampT(params_.voices, 1, kMaxVoices);
		if (!paramsSeen_) {
			// first parameter set after prepare(): start from the requested values instead of fading in from the defaults
			// (a patch saved with VOICES = 1 must be the untouched original from the very first sample)
			paramsSeen_ = true;
			normTarget_ = computeNorm();
			norm_.reset(normTarget_);
			outGain_.reset(dbToLin(params_.outputDb));
		}
		if (qChanged)
			applyQuality(params_.quality);
		if (seedChanged)
			reseed(params_.seed);
	}

	const EngineParams& params() const { return params_; }

	/** Test/tuning hook: force the static detune of voices 1..n-1 to explicit cents (bypasses SPREAD and the personality). */
	void setDetuneOverrideCents(const float* cents, int n) {
		for (int v = 0; v < kMaxVoices; v++)
			overrideCents_[v] = (v < n) ? cents[v] : 0.f;
		overrideActive_ = true;
	}
	void clearDetuneOverride() { overrideActive_ = false; }

	void reseed(uint32_t seed) {
		seed_ = seed;
		for (int v = 1; v < kMaxVoices; v++) {
			VoiceState& s = vs_[v];
			s.vp.generate(seed, v);
			s.drift.seed(hashCombine(s.vp.rngSeed, 1));
			s.levelDrift.seed(hashCombine(s.vp.rngSeed, 2));
			s.tiltDrift.seed(hashCombine(s.vp.rngSeed, 3));
			s.noiseRng.reseed(hashCombine(s.vp.rngSeed, 4));
		}
		common_.seed(hashCombine(seed, 0x51ED));
		lockCounter_ = 0;
	}

	void resetState() {
		dropSample_ = 0;
		ring_.reset();
		tracker_.reset();
		analyzer_.reset();
		std::fill(sq_.begin(), sq_.end(), 0.f);
		std::fill(xr_.begin(), xr_.end(), 0.f);
		sqSum_ = xSum_ = 0.0;
		dcSlow_ = 0.f;
		sqW_ = 64;
		state_ = ST_IDLE;
		lockW_ = 0.f;
		lockTarget_ = 0.f;
		envSm_ = envInst_ = 0.f;
		envGain_ = 0.f;
		noveltyShort_ = noveltyLong_ = 0.f;
		avgValid_ = false;
		pubPending_ = 0;
		samplesSincePub_ = 1 << 20;
		omegaS_ = 0.0;
		stepMismatch_ = 0;
		badHops_ = 0;
		acqCount_ = 0;
		doublingCount_ = 0;
		cooldown_ = 0;
		estStreak_ = 0;
		ctrlCounter_ = kCtrl; // force a control tick on the first sample
		triedDouble_ = false;
		tracker_.setActiveMask((1 << PitchTracker::kLanes) - 1);
		for (int v = 1; v < kMaxVoices; v++) {
			VoiceState& s = vs_[v];
			s.haveTable = false;
			s.gain = 0.f;
			s.alpha = 1.f;
			s.cur = 0;
			s.ratio = s.ratioTarget = 1.0;
			s.ratioInc = 0.0;
			s.phase = 0.0;
			s.drift.x = s.levelDrift.x = s.tiltDrift.x = 0.f;
		}
		outGain_.reset(dbToLin(params_.outputDb));
		normTarget_ = computeNorm();
		norm_.reset(normTarget_);
		cloneWeightSm_ = 0.f;
	}

	/** Process one sample. `in` is normalised so that 1.0 = the nominal 5 V audio level. */
	inline void process(float in, float& outL, float& outR) {
		if (!(in == in) || in > 1e6f || in < -1e6f)
			in = 0.f;
		ring_.push(in);
		samplesSincePub_ += 1.0;

		updateEnvelope(in);

		if (tracker_.push(in))
			onTrackerEstimate();
		if (analyzer_.active() && analyzer_.step(ring_))
			onAnalyzerSet();
		else if (state_ == ST_ACQ && ring_.count() - acqStartSample_ > (uint64_t) (0.6 * fs_)) {
			// watchdog: an acquisition that produced no usable analysis set for 0.6 s (history never sufficient, degenerate input) is abandoned
			analyzer_.stop();
			state_ = ST_IDLE;
			cooldown_ = 8;
			tracker_.setActiveMask((1 << PitchTracker::kLanes) - 1);
		}
		updateNovelty(in);

		if (++ctrlCounter_ >= kCtrl) {
			ctrlCounter_ = 0;
			controlTick();
		}
		if (pubPending_)
			buildNextTable();

		lockW_ += (lockTarget_ - lockW_) * (lockTarget_ > lockW_ ? lockRise_ : lockFall_);
		const int N = params_.voices;

		// ---- voices -------------------------------------------------------------------------------------------------------
		float wetMid = 0.f, wetL = 0.f, wetR = 0.f;
		const bool clonesRunning = !params_.originalOnly && N > 1 && lockW_ > 1e-4f && omegaS_ > 0.0;
		if (clonesRunning) {
			// the tracker's prediction for "now" (plus the slew's own lag) drives the clone oscillators; it already contains the frequency
			// slope, so glides and vibrato are followed without the (M/2 + 1)-period lag of the analysis window
			omegaLead_ = analyzer_.omegaAt((double) ring_.count() - 1.0 + omegaLeadSamples_);
			omegaS_ += (omegaLead_ - omegaS_) * omegaSlew_;
			const float eg = envGain();
			for (int v = 1; v < kMaxVoices; v++) {
				VoiceState& s = vs_[v];
				if (s.gain < 1e-5f && v >= N)
					continue;
				s.ratio += s.ratioInc;
				s.phase += omegaS_ * s.ratio;
				s.phase -= std::floor(s.phase);
				if (!s.haveTable)
					continue;
				float y = (taps_ == 32) ? renderVoice<32>(s, omegaS_) : renderVoice<16>(s, omegaS_);
				y *= eg * s.gain * s.levelLin;
				wetMid += y;
				wetL += y * s.panL;
				wetR += y * s.panR;
			}
		}
		for (int v = 1; v < kMaxVoices; v++) {
			VoiceState& s = vs_[v];
			s.gain += ((v < N && !params_.originalOnly ? 1.f : 0.f) - s.gain) * gainSlew_;
		}

		// ---- mix ------------------------------------------------------------------------------------------------------------
		const float normG = norm_.process(normTarget_);
		const float lw = lockW_;
		// power-complementary original gain: total power stays at the steady-state value while clones fade in
		const int Ne = params_.originalOnly ? 1 : N;
		const float go = normG * std::sqrt(std::max(0.f, (float) Ne - (float) (Ne - 1) * lw * lw));
		// the level law scales the AC content only: the input's DC offset passes at unity for any voice count (no thump when VOICES changes)
		const float orig = go * in + (1.f - go) * dcSlow_;
		const float wl = orig + normG * lw * wetL;
		const float wr = orig + normG * lw * wetR;
		const float wm = orig + normG * lw * wetMid;
		float l = width_ > 1e-4f ? wl : wm;
		float r = width_ > 1e-4f ? wr : wm;
		const float sumAmt = params_.summing * std::min(1.f, cloneWeightSm_);
		if (sumAmt > 1e-4f) {
			l = summingSat(l, sumAmt);
			r = summingSat(r, sumAmt);
		}
		const float mx = params_.mix;
		const float g = outGain_.process(dbToLin(params_.outputDb));
		outL = (in * (1.f - mx) + l * mx) * g;
		outR = (in * (1.f - mx) + r * mx) * g;
	}

	const EngineStatus& status() {
		status_.locked = state_ == ST_LOCKED;
		status_.acquiring = state_ == ST_ACQ;
		status_.lockWeight = lockW_;
		status_.activeVoices = params_.voices;
		status_.quality = params_.quality;
		status_.coherence = analyzer_.active() ? analyzer_.coherence() : 0.f;
		status_.periodicity = analyzer_.active() ? analyzer_.set().periodicity : 0.f;
		status_.unitFreqHz = analyzer_.active() ? analyzer_.omega() * fs_ : 0.0;
		status_.inputLevelDb = 20.f * std::log10(std::max(envSm_, 1e-6f));
		for (int v = 1; v < kMaxVoices; v++)
			status_.voiceCents[v] = (float) (1200.0 * std::log2(vs_[v].ratio));
		double P = analyzer_.active() ? analyzer_.period() : 0.0;
		status_.bloomMs = (float) (P > 0 ? 1000.0 * P * (analyzer_.windowPeriods() + 3.0) / fs_ : 0.0);
		status_.latencySamples = 0.f;
		return status_;
	}

	double sampleRate() const { return fs_; }
	const CycleAnalyzer& analyzer() const { return analyzer_; }
	const PitchTracker& tracker() const { return tracker_; }
	double voiceCents(int v) const { return 1200.0 * std::log2(vs_[v].ratio); }
	double voiceRatioTarget(int v) const { return vs_[v].ratioTarget; }
	bool lockedNow() const { return state_ == ST_LOCKED; }
	/** Diagnostics: why the last lock was dropped (1 novelty, 2 tracker mismatch, 3 coherence, 4 re-lock at a doubled period) and how often each
	    reason occurred. */
	int lastDropReason() const { return lastDropReason_; }
	int dropCount(int reason) const { return dropCount_[reason]; }
	/** Diagnostics/tests: the period table voice v is currently playing (returns false if it has none yet). */
	bool debugVoiceTable(int v, const float*& data, int& Nc) const {
		if (v < 1 || v >= kMaxVoices || !vs_[v].haveTable)
			return false;
		const PeriodTable& t = vs_[v].tab[vs_[v].cur];
		data = t.d.data() + PeriodTable::G;
		Nc = t.Nc;
		return true;
	}
	/** Repeating-unit frequency (Hz) the clones are currently running at (before their individual detune). */
	double cloneUnitFreq() const { return omegaS_ * fs_; }
	void noveltyDebug(float& shortV, float& longV, float& env2) const { shortV = noveltyShort_; longV = noveltyLong_; env2 = envInst_ * envInst_; }

private:
	enum State { ST_IDLE = 0, ST_ACQ = 1, ST_LOCKED = 2 };
	static const int kCtrl = 64; // control-rate tick in samples
	static constexpr double kOmegaSlewSec = 0.0015; // time constant of the clone-frequency slew

	// ------------------------------------------------------------------------------------------------------------------
	void applyQuality(int q) {
		CycleAnalyzer::Config c;
		c.minNc = 64;
		switch (q) {
		case QUALITY_ECO:
			c.M = 2; c.taps = 16; c.maxNc = 2048; c.maxHopSec = 0.006;
			pubMinSec_ = 0.012;
			break;
		case QUALITY_HIGH:
			c.M = 4; c.taps = 32; c.maxNc = 8192; c.maxHopSec = 0.004;
			pubMinSec_ = 0.005;
			break;
		case QUALITY_ULTRA:
			c.M = 4; c.taps = 32; c.maxNc = 8192; c.maxHopSec = 0.003;
			pubMinSec_ = 0.004;
			break;
		default:
			c.M = 2; c.taps = 16; c.maxNc = 4096; c.maxHopSec = 0.004;
			pubMinSec_ = 0.007;
			break;
		}
		cfg_ = c;
		taps_ = c.taps;
		analyzer_.configure(cfg_);
		state_ = ST_IDLE;
		lockW_ = lockTarget_ = 0.f;
		avgValid_ = false;
		pubPending_ = 0;
		for (int v = 1; v < kMaxVoices; v++)
			vs_[v].haveTable = false;
		tracker_.setActiveMask((1 << PitchTracker::kLanes) - 1);
	}

	RealFFT* ifftFor(int n) {
		for (size_t i = 0; i < ifftN_.size(); i++)
			if (ifftN_[i] == n)
				return ifft_[i].get();
		return nullptr;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// level tracking: AC RMS over ~one table period (ripple free for periodic input). The mean over the same window is removed: a DC offset
	// (asymmetric tube stage, unipolar pulse) is not part of the oscillator's tone and must not inflate the clone level.
	void updateEnvelope(float x) {
		const float x2 = x * x;
		const uint64_t n = ring_.count();
		const size_t iOld = (size_t) ((n - 1 - (uint64_t) sqW_) & (uint64_t) sqMask_), iNew = (size_t) ((n - 1) & (uint64_t) sqMask_);
		const float old = sq_[iOld], oldX = xr_[iOld];
		sq_[iNew] = x2;
		xr_[iNew] = x;
		sqSum_ += (double) x2 - (double) old;
		xSum_ += (double) x - (double) oldX;
		if ((n & 4095) == 0) { // exact recompute against drift
			double s2 = 0, s1 = 0;
			for (int i = 0; i < sqW_; i++) {
				const size_t k = (size_t) ((n - 1 - (uint64_t) i) & (uint64_t) sqMask_);
				s2 += sq_[k];
				s1 += xr_[k];
			}
			sqSum_ = s2;
			xSum_ = s1;
		}
		const double mean = xSum_ / sqW_;
		const float rms = (float) std::sqrt(std::max(0.0, sqSum_ / sqW_ - mean * mean));
		envSm_ += (rms - envSm_) * (rms > envSm_ ? 0.05f : 0.005f);
		envInst_ = rms;
		// slow DC estimate (tau ~ 100 ms) for the output stage
		dcSlow_ += (x - dcSlow_) * dcCoef_;
	}

	void setEnvWindow(double periodSamples) {
		int k = std::max(1, (int) std::ceil(0.0007 * fs_ / std::max(periodSamples, 1.0)));
		int W = clampT((int) std::lround(periodSamples * k), 16, 60000);
		if (W == sqW_)
			return;
		sqW_ = W;
		const uint64_t n = ring_.count();
		double s2 = 0, s1 = 0;
		for (int i = 0; i < sqW_; i++) {
			const size_t k = (size_t) ((n - 1 - (uint64_t) i) & (uint64_t) sqMask_);
			s2 += sq_[k];
			s1 += xr_[k];
		}
		sqSum_ = s2;
		xSum_ = s1;
	}

	inline float envGain() {
		const float t = tabRms_ > 1e-9f ? clampT(envInst_ / tabRms_, 0.f, 4.f) : 0.f;
		envGain_ += (t - envGain_) * envSlew_;
		return envGain_;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// transient / discontinuity detector: residual of x(n) against x(n - P) (one period ago)
	void updateNovelty(float x) {
		if (state_ != ST_LOCKED || !analyzer_.active()) {
			noveltyShort_ = noveltyLong_ = 0.f;
			return;
		}
		const uint64_t n = ring_.count();
		// mean period over the last cycle (the tracked frequency can move during it): evaluate the prediction half a cycle back
		const double P = 1.0 / analyzer_.omegaAt((double) n - 1.0 - 0.5 * analyzer_.period());
		if (P + 40.0 > (double) ring_.size() * 0.5)
			return;
		const float past = ring_.readSinc<16>((double) (n - 1) - P);
		const float e = x - past;
		const float e2 = e * e;
		// Fast vs slow estimate of the same quantity. A pitch step / onset / dropout raises the residual within a few ms; natural
		// modulation (detune-sideband beating, glides, PWM) moves it smoothly over >= 100 ms, so it never opens a large gap
		// between the two estimates.
		const float tauFast = std::max(8.f, (float) (0.5 * P));
		noveltyShort_ += (e2 - noveltyShort_) * (1.f - std::exp(-1.f / tauFast));
		noveltyLong_ += (e2 - noveltyLong_) * (1.f - std::exp(-1.f / (10.f * tauFast)));
		const float floorV = 2e-3f * envInst_ * envInst_;
		// Absolute floor: a glide's onset raises the one-period residual by a few % of the power, a real step/onset by tens of %.
		if (lockAge_ > 6 && noveltyShort_ > 6.f * noveltyLong_ + floorV && noveltyShort_ > 0.10f * envInst_ * envInst_)
			dropLock(1);
	}

	void dropLock(int reason = 0) {
		lastDropReason_ = reason;
		dropCount_[reason < 0 ? 0 : (reason > 4 ? 4 : reason)]++;
		// the event happened a little before we noticed it: forget the tracker history before it
		const int back = (int) (0.004 * fs_);
		tracker_.forget(back);
		dropSample_ = ring_.count() - (uint64_t) back;
		state_ = ST_IDLE;
		lockTarget_ = 0.f;
		analyzer_.stop();
		tracker_.setActiveMask((1 << PitchTracker::kLanes) - 1);
		cooldown_ = 1;
		noveltyShort_ = noveltyLong_ = 0.f;
		avgValid_ = false;
		estStreak_ = 0;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// acquisition
	void onTrackerEstimate() {
		const PitchTracker::Estimate& e = tracker_.estimate();
		if (cooldown_ > 0)
			cooldown_--;
		const bool loud = envInst_ > 3e-4f; // about -70 dB re 5 V
		if (state_ == ST_IDLE) {
			if (!e.valid || e.conf < 0.85f || !loud || cooldown_ > 0) {
				estStreak_ = 0;
				return;
			}
			if (estStreak_ > 0 && std::fabs(e.period / lastEstPeriod_ - 1.0) < 0.03)
				estStreak_++;
			else
				estStreak_ = 1;
			lastEstPeriod_ = e.period;
			// the analysis window of the cycle analyser must not straddle the last event
			const double need = (cfg_.M + 1.0) * e.period;
			if (estStreak_ >= 2 && (double) (ring_.count() - dropSample_) >= need)
				beginAcquisition(refinePeriod(e.period));
		} else if (state_ == ST_LOCKED) {
			// verification: repeated confident estimates that match neither the PLL period nor a simple multiple of it
			if (e.valid && e.conf > 0.9f) {
				const double ratio = e.period / analyzer_.period();
				bool consistent = std::fabs(ratio - 1.0) < 0.05 || std::fabs(ratio - 0.5) < 0.03 || std::fabs(ratio - 2.0) < 0.08;
				if (!consistent) {
					if (++stepMismatch_ >= 3) {
						stepMismatch_ = 0;
						dropLock(2);
					}
				} else
					stepMismatch_ = 0;
				// Period doubling while locked (the SUB knob was turned up): the tracker now reports twice the locked period and the harmonic
				// model of the single period stops explaining the signal (its half-integer harmonics are non-periodic energy). Re-acquire at
				// the doubled period so the sub is cloned too. The periodicity condition keeps decimation artefacts of narrow pulses (which
				// the tracker also reports as period doubling, but which are perfectly periodic) from triggering it.
				if (std::fabs(ratio - 2.0) < 0.05 && analyzer_.set().periodicity < 0.985f) {
					if (++doublingCount_ >= 8 && analyzer_.period() * 2.0 < (double) ring_.size() * 0.2) {
						doublingCount_ = 0;
						lastDropReason_ = 4;
						dropCount_[4]++;
						lockTarget_ = 0.f;
						beginAcquisition(analyzer_.period() * 2.0);
					}
				} else
					doublingCount_ = 0;
			}
		}
	}

	/** Full-rate refinement of the period around a coarse estimate (integer lag scan + parabolic peak). */
	double refinePeriod(double P0) {
		const uint64_t n = ring_.count();
		const int W = clampT((int) std::ceil(1.5 * P0), 128, 6144);
		int span = std::max(3, (int) std::ceil(0.035 * P0));
		span = std::min(span, 1000);
		const int lag0 = (int) std::floor(P0) - span, lag1 = (int) std::ceil(P0) + span;
		if (lag0 < 2 || (uint64_t) (lag1 + W + 8) >= n || lag1 + W + 8 > ring_.size() - 64 || (size_t) (lag1 - lag0 + 1) > refineBuf_.size())
			return P0;
		std::vector<float>& d = refineBuf_;
		float best = 1e30f;
		int bestLag = lag0;
		for (int lag = lag0; lag <= lag1; lag++) {
			float acc = 0.f;
			for (int j = 0; j < W; j++) {
				const float df = ring_.at(n - 1 - (uint64_t) j) - ring_.at(n - 1 - (uint64_t) j - (uint64_t) lag);
				acc += df * df;
			}
			d[lag - lag0] = acc;
			if (acc < best) {
				best = acc;
				bestLag = lag;
			}
		}
		if (bestLag <= lag0 || bestLag >= lag1)
			return P0;
		const double a = d[bestLag - 1 - lag0], b = d[bestLag - lag0], c = d[bestLag + 1 - lag0];
		const double den = a - 2.0 * b + c;
		double off = std::fabs(den) > 1e-20 ? 0.5 * (a - c) / den : 0.0;
		off = clampT(off, -1.0, 1.0);
		return bestLag + off;
	}

	void beginAcquisition(double P) {
		analyzer_.start(P, ring_.count());
		setEnvWindow(P);
		state_ = ST_ACQ;
		acqStartSample_ = ring_.count();
		acqCount_ = 0;
		badHops_ = 0;
		lockAge_ = 0;
		avgValid_ = false;
		stepMismatch_ = 0;
		// once we have a period only the lanes around it keep running
		const int lane = tracker_.laneForPeriod(P);
		int mask = 1 << lane;
		if (lane > 0)
			mask |= 1 << (lane - 1);
		if (lane < PitchTracker::kLanes - 1)
			mask |= 1 << (lane + 1);
		tracker_.setActiveMask(mask);
	}

	// ------------------------------------------------------------------------------------------------------------------
	void onAnalyzerSet() {
		const HarmonicSet& s = analyzer_.set();
		lockAge_++;
		if (state_ == ST_ACQ) {
			acqCount_++;
			// coherence is the noise discriminator (noise/chaos ~ 0); the periodicity index only rejects signals that are mostly inharmonic
			const bool good = s.coherence > 0.92f && analyzer_.goodHops() >= 2 && s.periodicity > 0.30f;
			if (good) {
				if (unitIsDoubled(s)) {
					beginAcquisition(analyzer_.period() * 0.5);
					return;
				}
				lockAcquired();
			} else if (acqCount_ > 14) {
				// could not lock: maybe a wrong octave -> try the double period once, else back off
				const double P = analyzer_.period();
				if (!triedDouble_ && P * 2.0 < (double) ring_.size() * 0.2) {
					triedDouble_ = true;
					beginAcquisition(P * 2.0);
				} else {
					triedDouble_ = false;
					analyzer_.stop();
					state_ = ST_IDLE;
					cooldown_ = 8;
					tracker_.setActiveMask((1 << PitchTracker::kLanes) - 1);
				}
				return;
			}
		} else if (state_ == ST_LOCKED) {
			if (s.coherence < 0.6f || s.periodicity < 0.12f) {
				if (++badHops_ >= 2) {
					dropLock(3);
					dropSample_ -= std::min<uint64_t>(dropSample_, (uint64_t) (0.012 * fs_));
					tracker_.forget((int) (0.016 * fs_));
					return;
				}
			} else
				badHops_ = 0;
		}
		if (state_ == ST_IDLE)
			return;
		// smooth the harmonic set over hops (a complex average is valid because successive sets are aligned)
		const int J = s.J;
		float beta = s.coherence > 0.95f ? 0.5f : 1.f;
		if (!avgValid_ || avgJ_ != J || avgNc_ != s.Nc)
			beta = 1.f;
		for (int j = 1; j <= J; j++) {
			avgRe_[j] += (s.re[j] - avgRe_[j]) * beta;
			avgIm_[j] += (s.im[j] - avgIm_[j]) * beta;
		}
		avgJ_ = J;
		avgNc_ = s.Nc;
		avgPeriod_ = s.period;
		avgValid_ = true;
		if (state_ == ST_LOCKED)
			maybePublish(false);
	}

	/** The tracker's multiplicity test can call a genuinely single-cycle waveform "period doubled" (e.g. a narrow pulse whose position alternates
	    between two sub-sample phases of a decimated lane looks like "every second cycle differs"). A real sub oscillator puts energy on the odd
	    harmonics of the doubled unit; if those carry (almost) nothing, the true period is half the unit. Odd bins are free of Hann leakage from
	    the even ones (exact zeros at +-2 bins), so the test is clean down to about -37 dB. Also catches a plain octave-down tracker error. */
	bool unitIsDoubled(const HarmonicSet& s) const {
		if (s.J < 4 || analyzer_.period() * 0.5 < 6.0)
			return false;
		double eOdd = 0.0, eAll = 0.0;
		for (int j = 1; j <= s.J; j++) {
			const double e2 = (double) s.re[j] * s.re[j] + (double) s.im[j] * s.im[j];
			eAll += e2;
			if (j & 1)
				eOdd += e2;
		}
		return eAll > 1e-12 && eOdd < 2e-4 * eAll;
	}

	void lockAcquired() {
		triedDouble_ = false;
		state_ = ST_LOCKED;
		lockAge_ = 5;
		omegaS_ = analyzer_.omega();
		omegaLead_ = omegaS_;
		lockCounter_++;
		Rng r(hashCombine(seed_, lockCounter_));
		const double theta = analyzer_.thetaNow();
		for (int v = 1; v < kMaxVoices; v++) {
			VoiceState& s = vs_[v];
			// PHASE = 0 -> phase-aligned with the source; PHASE = 1 -> fully random start phase (fresh draw per lock event)
			const double off = (double) params_.phase * (double) r.uniform();
			s.phase = frac(theta + off);
			s.xPhase[0] = frac(s.phase + 0.23 + 0.2 * (double) r.uniform());
			s.xPhase[1] = frac(s.phase + 0.61 + 0.2 * (double) r.uniform());
			s.lfoPhase = (double) s.vp.lfoPhase;
			s.carrier[0].setPhase((double) r.uniform());
			s.carrier[1].setPhase((double) r.uniform());
			s.hil.reset();
			s.haveTable = false;
			s.alpha = 1.f;
		}
		// the set that made us lock becomes the first published table set
		const HarmonicSet& hs = analyzer_.set();
		for (int j = 1; j <= hs.J; j++) {
			avgRe_[j] = hs.re[j];
			avgIm_[j] = hs.im[j];
		}
		avgJ_ = hs.J;
		avgNc_ = hs.Nc;
		avgPeriod_ = hs.period;
		avgValid_ = true;
		maybePublish(true);
		lockTarget_ = 1.f;
	}

	// ------------------------------------------------------------------------------------------------------------------
	// table publication (time sliced: one voice per sample)
	void maybePublish(bool force) {
		if (!avgValid_)
			return;
		if (!force && (samplesSincePub_ < pubMinSec_ * fs_ || pubPending_))
			return;
		const int J = avgJ_;
		for (int j = 1; j <= J; j++) {
			pubRe_[j] = avgRe_[j];
			pubIm_[j] = avgIm_[j];
		}
		pubJ_ = J;
		pubNc_ = avgNc_;
		pubPeriod_ = avgPeriod_;
		double e2 = 0;
		for (int j = 1; j <= J; j++)
			e2 += 2.0 * ((double) pubRe_[j] * pubRe_[j] + (double) pubIm_[j] * pubIm_[j]);
		tabRms_ = (float) std::sqrt(e2);
		samplesSincePub_ = 0;
		pubPending_ = 0;
		for (int v = 1; v < params_.voices; v++)
			pubPending_ |= (1u << v);
		// GUI spectrum snapshot (seqlock: odd while writing)
		status_.specSeq++;
		float mx = 1e-14f;
		for (int j = 1; j <= std::min(J, EngineStatus::kSpecBins); j++)
			mx = std::max(mx, pubRe_[j] * pubRe_[j] + pubIm_[j] * pubIm_[j]);
		for (int j = 0; j < EngineStatus::kSpecBins; j++) {
			const float m2 = (j + 1 <= J) ? pubRe_[j + 1] * pubRe_[j + 1] + pubIm_[j + 1] * pubIm_[j + 1] : 1e-14f;
			status_.spectrumDb[j] = 10.f * std::log10(std::max(m2, 1e-14f) / mx);
		}
		status_.J = J;
		status_.Nc = pubNc_;
		status_.specSeq++;
	}

	void buildNextTable() {
		unsigned m = pubPending_;
		int v = 1;
		while (v < kMaxVoices && !((m >> v) & 1u))
			v++;
		if (v >= kMaxVoices) {
			pubPending_ = 0;
			return;
		}
		pubPending_ &= ~(1u << v);
		buildVoiceTable(v);
	}

	void buildVoiceTable(int v) {
		VoiceState& s = vs_[v];
		const int Nc = pubNc_;
		RealFFT* fft = ifftFor(Nc);
		if (!fft || pubJ_ < 1)
			return;
		const double P = pubPeriod_;
		const float h = params_.harmonic, ph = params_.phase;
		s.curve.build(s.vp, h, ph, s.tiltDrift.x, fs_ / P);
		// anti-aliasing limit for this voice: harmonic j sits at j*ratio/P cycles/sample and must stay below 0.49
		const double rmax = s.ratioTarget * 1.004;
		const int Jaa = (int) std::floor(0.49 * P / rmax);
		const int J = std::min(std::min(pubJ_, Jaa), Nc / 2 - 1);
		std::fill(specBuf_.data(), specBuf_.data() + Nc, 0.f);
		// noise gate: drop harmonics 90 dB below the strongest
		float mx2 = 1e-20f;
		for (int j = 1; j <= J; j++)
			mx2 = std::max(mx2, pubRe_[j] * pubRe_[j] + pubIm_[j] * pubIm_[j]);
		const float floor2 = mx2 * 1e-9f;
		const Log2Lut& L = log2Lut();
		const float gscale = (float) DivergenceCurve::kGrid / (float) DivergenceCurve::kMaxOct;
		const int taperStart = (int) (0.90 * J);
		for (int j = 1; j <= J; j++) {
			const float re = pubRe_[j], im = pubIm_[j];
			if (re * re + im * im < floor2)
				continue;
			const float x = L.v[j] * gscale;
			const int gi = std::min((int) x, DivergenceCurve::kGrid - 1);
			const float fr = x - gi;
			float g = s.curve.gain[gi] + (s.curve.gain[gi + 1] - s.curve.gain[gi]) * fr;
			const float th = s.curve.phase[gi] + (s.curve.phase[gi + 1] - s.curve.phase[gi]) * fr;
			g *= 1.f + ((j & 1) ? -s.curve.oddEven : s.curve.oddEven);
			if (j > taperStart) {
				const float t = (float) (j - taperStart) / (float) std::max(1, J - taperStart);
				const float c = std::cos(0.5f * (float) kPi * t);
				g *= c * c;
			}
			// small-angle rotation e^{i th} ~ (1 - th^2/2) + i th   (|th| < ~0.3 rad)
			const float cr = 1.f - 0.5f * th * th, ci = th;
			specBuf_[2 * j] = (re * cr - im * ci) * g;
			specBuf_[2 * j + 1] = (re * ci + im * cr) * g;
		}
		fft->inverse(specBuf_.data(), tabBuf_.data());
		if (params_.character > 1e-3f && params_.quality >= QUALITY_HIGH)
			applyCharacter(s, Nc, J, fft);
		PeriodTable& dst = s.haveTable ? s.tab[s.cur ^ 1] : s.tab[s.cur];
		float* d = dst.d.data() + PeriodTable::G;
		for (int i = 0; i < Nc; i++)
			d[i] = tabBuf_[i];
		dst.finalize(Nc);
		if (!s.haveTable) {
			s.haveTable = true;
			s.alpha = 1.f;
		} else {
			s.alpha = 0.f;
			s.alphaInc = 1.f / std::max(32.f, (float) (pubMinSec_ * fs_ * 0.85));
		}
	}

	/** CHARACTER: static, tiny, asymmetric waveshaping of the period table. The shaped table is re-band-limited in the
	    frequency domain (harmonics above J removed), so it cannot alias — no oversampling is needed at run time. */
	void applyCharacter(VoiceState& s, int Nc, int J, RealFFT* fft) {
		float peak = 1e-9f;
		for (int i = 0; i < Nc; i++)
			peak = std::max(peak, std::fabs(tabBuf_[i]));
		const float c = params_.character;
		const float a = 0.08f * c * (0.4f + 0.6f * s.vp.drive) * 4.f; // <= 0.32: ~1% THD at full scale
		const float b = 0.06f * c * s.vp.bias;
		const float inv = 1.f / peak;
		const float t0 = fastTanh(a * b);
		for (int i = 0; i < Nc; i++) {
			const float u = tabBuf_[i] * inv;
			tabBuf_[i] = (fastTanh(a * (u + b)) - t0) / a * peak;
		}
		fft->forward(tabBuf_.data(), specBuf_.data());
		specBuf_[0] = 0.f; // no DC
		specBuf_[1] = 0.f;
		for (int k = J + 1; k < Nc / 2; k++)
			specBuf_[2 * k] = specBuf_[2 * k + 1] = 0.f;
		fft->inverse(specBuf_.data(), tabBuf_.data());
		const float sc = 1.f / (float) Nc;
		for (int i = 0; i < Nc; i++)
			tabBuf_[i] *= sc;
	}

	template <int TAPS>
	inline float readMix(const VoiceState& s, double phase) const {
		const float a = s.tab[s.cur].template read<TAPS>(phase);
		if (s.alpha >= 1.f)
			return a;
		const float b = s.tab[s.cur ^ 1].template read<TAPS>(phase);
		return a + s.alpha * (b - a);
	}

	/** One output sample of a clone: the main oscillator plus (FUSION algorithm) the two detune-cluster lines. */
	template <int TAPS>
	inline float renderVoice(VoiceState& s, double omega) {
		float y = readMix<TAPS>(s, s.phase);
		if (fusionOn_) {
			if (params_.shiftMode == SHIFT_RATIO) {
				// two extra oscillators whose ratio is pushed apart by a soft-square LFO (the Doppler / time-varying-delay reading of the
				// BBD "frequency shifter": multiplicative pitch offsets that swap sign every half LFO cycle)
				s.lfoPhase += s.lfoInc;
				s.lfoPhase -= std::floor(s.lfoPhase);
				const float tri = 1.f - 4.f * (float) std::fabs(s.lfoPhase - 0.5);
				const double sh = (double) s.shiftFrac * (double) clampT(3.f * tri, -1.f, 1.f);
				s.xPhase[0] += omega * s.ratio * (1.0 + sh);
				s.xPhase[1] += omega * s.ratio * (1.0 - sh);
				s.xPhase[0] -= std::floor(s.xPhase[0]);
				s.xPhase[1] -= std::floor(s.xPhase[1]);
				const float y0 = readMix<TAPS>(s, s.xPhase[0]), y1 = readMix<TAPS>(s, s.xPhase[1]);
				y = (y + s.shiftGain * (y0 + y1)) * fusionNorm_;
			} else {
				// literal single-sideband reading: constant additive offsets (+D, -D') in Hz on all partials
				float i, q;
				s.hil.process(y, i, q);
				s.carrier[0].step();
				s.carrier[1].step();
				const float up = i * s.carrier[0].c + q * s.carrier[0].s; // sign convention verified in tests/test_hilbert.cpp
				const float dn = i * s.carrier[1].c - q * s.carrier[1].s;
				y = (i + s.shiftGain * (up + dn)) * fusionNorm_;
			}
		}
		if (s.alpha < 1.f) {
			s.alpha += s.alphaInc;
			if (s.alpha >= 1.f) {
				s.alpha = 1.f;
				s.cur ^= 1;
			}
		}
		return y;
	}

	// ------------------------------------------------------------------------------------------------------------------
	void controlTick() {
		const float dt = (float) kCtrl / (float) fs_;
		const EngineParams& p = params_;
		const int N = p.voices;
		const float spreadCurved = std::pow(clampT(p.spread, 0.f, 1.f), p.spreadCurve);
		const double sigmaCents = p.detuneRangeCents / 2.4 * spreadCurved; // one-sigma of the tolerance distribution
		const float tau = 30.f * std::pow(0.07f, clampT(p.driftRate, 0.f, 1.f));
		const float driftSigma = 1.2f * p.drift; // cents
		const float common = common_.step(dt, tau, 1.f);
		const float rho = clampT(p.driftCorrelation, 0.f, 1.f);
		width_ = p.width;
		fusionOn_ = p.algorithm == ALGO_FUSION && p.fusionShift > 0.002f;
		const float fs_amt = clampT(p.fusionShift, 0.f, 1.f);
		const float shGain = 0.55f * std::sqrt(fs_amt);
		fusionNorm_ = 1.f / std::sqrt(1.f + 2.f * shGain * shGain);
		cloneWeightSm_ += ((N > 1 ? 1.f : 0.f) - cloneWeightSm_) * 0.02f;
		normTarget_ = computeNorm();
		gainSlew_ = 1.f - std::exp(-1.f / (float) (0.02 * fs_));
		omegaSlew_ = 1.f - std::exp(-1.f / (float) (kOmegaSlewSec * fs_));
		omegaLeadSamples_ = kOmegaSlewSec * fs_;
		envSlew_ = 1.f - std::exp(-1.f / (float) (0.001 * fs_));
		const double P = analyzer_.active() ? analyzer_.period() : 0.0;
		const double riseSec = clampT(1.5 * P / fs_, 0.003, 0.03);
		lockRise_ = 1.f - std::exp(-1.f / (float) (riseSec * fs_));
		lockFall_ = 1.f - std::exp(-1.f / (float) (0.0012 * fs_));
		for (int v = 1; v < kMaxVoices; v++) {
			VoiceState& s = vs_[v];
			// pitch: static tolerance + shared/independent drift + tiny jitter
			const float indep = s.drift.step(dt, tau, driftSigma);
			const float dcents = std::sqrt(rho) * common * driftSigma + std::sqrt(1.f - rho) * indep;
			const float jit = s.noiseRng.gauss() * 0.01f * p.character; // cents, white
			const double cents = (overrideActive_ ? (double) overrideCents_[v] : s.vp.centsUnit * sigmaCents) + dcents + jit;
			const double r1 = centsToRatio(cents);
			s.ratioTarget = r1;
			if (!(s.ratio > 0.5 && s.ratio < 2.0))
				s.ratio = r1;
			s.ratioInc = (r1 - s.ratio) / (double) kCtrl;
			const float lvlDb = s.vp.levelDb * 0.3f * p.character + s.levelDrift.step(dt, tau * 0.5f, 0.12f * p.drift);
			s.levelLin = dbToLin(lvlDb);
			s.tiltDrift.step(dt, tau * 0.7f, 0.25f * p.drift);
			// Fusion detune-cluster layer parameters (hypothesis model, see docs/RESEARCH.md): LFO rate and depth grow together with the knob,
			// exactly as the product description says for the hardware DETUNE control; each voice has its own LFO phase and rate tolerance.
			s.shiftGain = shGain;
			s.lfoInc = (0.12 + 2.0 * fs_amt) * s.vp.lfoRate / fs_;
			s.shiftFrac = (float) (centsToRatio(3.0 + 22.0 * fs_amt) - 1.0);
			{
				const double dHz = (0.35 + 5.5 * fs_amt) * s.vp.lfoRate;
				s.carrier[0].setFreq(dHz / fs_);
				s.carrier[1].setFreq(1.13 * dHz / fs_);
			}
			// stereo: linear pan so that L+R equals the mono sum for any WIDTH
			const float a = 0.25f * p.width * s.vp.pan;
			s.panL = 1.f - a;
			s.panR = 1.f + a;
		}
	}

	float computeNorm() const {
		const int Ne = params_.originalOnly ? 1 : params_.voices;
		return std::pow((float) Ne, -(0.5f - params_.levelLaw));
	}

	static inline float fastTanh(float x) {
		// Pade approximation, |error| < 2e-3 on [-3,3], saturating outside
		x = clampT(x, -4.97f, 4.97f);
		const float x2 = x * x;
		return x * (27.f + x2) / (27.f + 9.f * x2);
	}
	static inline float summingSat(float x, float amt) {
		// y = k*tanh(x/k) with k = 4 (= 20 V); blended by `amt`. Symmetric on purpose: the source's tube colour must dominate.
		const float k = 4.f;
		const float y = k * fastTanh(x / k);
		return x + (y - x) * amt;
	}

	// ------------------------------------------------------------------------------------------------------------------
	double fs_ = 48000.0;
	EngineParams params_;
	uint32_t seed_ = 1;
	CycleAnalyzer::Config cfg_;
	int taps_ = 16;
	double pubMinSec_ = 0.007;

	MirrorRing ring_;
	PitchTracker tracker_;
	CycleAnalyzer analyzer_;

	std::vector<std::unique_ptr<RealFFT> > ifft_;
	std::vector<int> ifftN_;
	AlignedBuffer<float> specBuf_, tabBuf_;
	std::vector<float> avgRe_, avgIm_, pubRe_, pubIm_;
	int avgJ_ = 0, avgNc_ = 0, pubJ_ = 0, pubNc_ = 0;
	double avgPeriod_ = 0, pubPeriod_ = 0;
	bool avgValid_ = false;
	float tabRms_ = 0.f;
	unsigned pubPending_ = 0;
	double samplesSincePub_ = 0;

	VoiceState vs_[kMaxVoices];
	OuProcess common_;
	uint32_t lockCounter_ = 0;

	// envelope
	std::vector<float> sq_, xr_;
	int sqMask_ = 65535, sqW_ = 64;
	double sqSum_ = 0, xSum_ = 0;
	float dcSlow_ = 0.f, dcCoef_ = 0.f;
	float envSm_ = 0.f, envInst_ = 0.f, envGain_ = 0.f, envSlew_ = 0.02f;
	float noveltyShort_ = 0.f, noveltyLong_ = 0.f;

	// state machine
	State state_ = ST_IDLE;
	int estStreak_ = 0;
	uint64_t dropSample_ = 0;
	double lastEstPeriod_ = 0;
	int acqCount_ = 0, badHops_ = 0, stepMismatch_ = 0, doublingCount_ = 0, lockAge_ = 0, cooldown_ = 0;
	bool triedDouble_ = false;
	float lockW_ = 0.f, lockTarget_ = 0.f, lockRise_ = 0.01f, lockFall_ = 0.02f;
	double omegaS_ = 0.0;
	double omegaLead_ = 0.0, omegaLeadSamples_ = 0.0;
	float omegaSlew_ = 0.005f, gainSlew_ = 0.001f, normTarget_ = 1.f;
	float width_ = 0.f, cloneWeightSm_ = 0.f;
	int ctrlCounter_ = 0;
	std::vector<float> refineBuf_;

	uint64_t acqStartSample_ = 0;
	int lastDropReason_ = 0;
	int dropCount_[5] = {0, 0, 0, 0, 0};
	bool paramsSeen_ = false;
	bool fusionOn_ = false;
	float fusionNorm_ = 1.f;
	float overrideCents_[kMaxVoices] = {0};
	bool overrideActive_ = false;
	OnePole outGain_, norm_;
	EngineStatus status_;
};

} // namespace fc
