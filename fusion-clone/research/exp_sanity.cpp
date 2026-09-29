// Sanity checks for the synthetic source model and the metrics (does the harness measure what we think it measures?).
#include "common/fusion_source.hpp"
#include "common/metrics.hpp"
#include "common/wav.hpp"
#include <cstdio>

using namespace research;

static std::vector<double> stratified(int n, double sigmaCents, uint32_t seed) {
	std::vector<double> c(n);
	fc::Rng r(seed);
	std::vector<int> perm(n);
	for (int i = 0; i < n; i++) perm[i] = i;
	for (int i = n - 1; i > 0; i--) std::swap(perm[i], perm[r.nextU32() % (i + 1)]);
	for (int i = 0; i < n; i++) {
		double p = (perm[i] + 0.5 + 0.4 * r.bipolar()) / n;
		c[i] = sigmaCents * fc::normalQuantile(p);
	}
	return c;
}

int main() {
	const double fs = 48000;
	// 1. clean saw
	{
		SourceSpec s; s.f0 = 110; s.seconds = 8; s.seed = 3;
		auto x = renderSource(s);
		Psd p = welch(x, fs, 1 << 16);
		Clusters c = harmonicClusters(p, 110, 8);
		printf("[1] saw 110 Hz: harmonic energies rel. k=1 (ideal -6.02*log2(k) dB):\n   ");
		for (int k = 0; k < 8; k++) printf("%5.1f ", dbPow(c.energy[k] / c.energy[0]) );
		printf("\n   centroid offsets (cents): ");
		for (int k = 0; k < 8; k++) printf("%5.2f ", c.centroid[k]);
		printf("\n   inter-harmonic ratio: %.1f dB\n", interHarmonicDb(p, 110, 16));
	}
	// 2. sub oscillator: energy at f0/2
	{
		SourceSpec s; s.f0 = 110; s.seconds = 8; s.sub = 0.5; s.seed = 3;
		auto x = renderSource(s);
		Psd p = welch(x, fs, 1 << 16);
		double e0 = bandEnergy(p, 110 * 0.98, 110 * 1.02), es = bandEnergy(p, 55 * 0.98, 55 * 1.02);
		printf("[2] sub=0.5: level at f0/2 relative to f0: %.1f dB (expect ~ +something; sub square fundamental 4/pi*0.5 vs saw 2/pi)\n", dbPow(es / e0));
	}
	// 3. Detune hypotheses: sideband spacing vs harmonic number
	for (int model : {DETUNE_DOPPLER, DETUNE_SSB}) {
		SourceSpec s; s.f0 = 110; s.seconds = 20; s.detune = 0.3; s.detuneModel = model; s.seed = 5;
		auto x = renderSource(s);
		Psd p = welch(x, fs, 1 << 18);
		Clusters c = harmonicClusters(p, 110, 12, 0.03);
		printf("[3] detune model=%s: cluster spread (cents) vs k:\n   ", model == DETUNE_DOPPLER ? "Doppler(mult.)" : "SSB(additive)");
		for (int k = 0; k < 12; k++) printf("%5.1f ", c.spread[k]);
		printf("\n   -> spread in Hz = spread*k*110*ln2/1200: ");
		for (int k = 0; k < 12; k++) printf("%5.2f ", c.spread[k] * (k + 1) * 110 * 0.000577623);
		printf("\n");
	}
	// 4. independent bank: envelope statistics vs N
	for (int N : {1, 2, 4, 16}) {
		SourceSpec s; s.f0 = 110; s.seconds = 20; s.seed = 11; s.wSaw = 0; s.wSine = 1;  // sine so the k=1 band holds only the fundamental
		auto offs = stratified(std::max(0, N - 1), 6.0, 42);
		auto x = renderBank(s, N, offs, 0.0, 99);
		EnvStats e = bandEnvelopeStats(x, fs, 110 * 0.9, 110 * 1.1, 48000, x.size() - 48000);
		printf("[4] bank N=%2d: envelope CV=%.3f  modulation peakiness=%.1f dB @ %.2f Hz  (Rayleigh N->inf: CV=0.523)\n", N, e.cv, e.peakinessDb, e.peakHz);
	}
	return 0;
}
