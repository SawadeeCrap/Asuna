#!/usr/bin/env python3
"""Decide whether a DETUNE section shifts pitch multiplicatively (Doppler / variable delay) or additively (single-sideband shift).

Background (docs/RESEARCH.md §2.3): the Fusion VCO2 manufacturer text says DETUNE is made of "two BBD delay lines that make a frequency
shifter mixed back to the principal oscillator". Two hypotheses fit that text:

  H-DOPPLER  the shifted copies have pitch ratio r = 1 +- rho: the sideband offset around harmonic n is  n * f0 * rho   (grows with n)
  H-SSB      the copies are shifted by a fixed frequency +- D: the offset is the same D Hz around every harmonic

The probe measures the RMS offset of the sideband cluster around each harmonic n of a recording of a steady, DETUNE-only note and fits
offset(n) = a + b n (weighted least squares). frac = b n_ref / (a + b n_ref) is ~1 for H-DOPPLER and ~0 for H-SSB.

    python3 tools/fusion_detune_probe.py rec.wav --f0 65.406 [--nmax 30]
    python3 tools/fusion_detune_probe.py --selftest
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_reference as ar  # noqa: E402


def classify(rows, n_ref=None):
    """rows: cluster_stats() output. Returns a dict with the fit and the verdict."""
    pts = [(r["n"], r["rms_offset_hz"], r["sideband_db"]) for r in rows if r["sideband_db"] is not None and r["rms_offset_hz"] > 0]
    if len(pts) < 4:
        return {"verdict": "INSUFFICIENT", "reason": "fewer than 4 harmonics with a measurable sideband cluster (is DETUNE on? is the file long enough?)"}
    n = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts])
    wgt = np.array([10 ** (p[2] / 20.0) for p in pts])  # weight by the sideband amplitude relative to the carrier (louder = better measured)
    wgt = wgt / np.max(wgt)
    A = np.vstack([np.ones_like(n), n]).T
    W = np.diag(wgt)
    coef, *_ = np.linalg.lstsq(W @ A, W @ y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    nr = float(np.median(n)) if n_ref is None else n_ref
    tot = a + b * nr
    frac = (b * nr) / tot if tot > 1e-12 else 0.0
    if frac > 0.7:
        verdict = "H-DOPPLER (multiplicative ratio; use Shift model = RATIO)"
    elif frac < 0.3:
        verdict = "H-SSB (additive Hz shift; use Shift model = HZ)"
    else:
        verdict = "MIXED (pick the dominant term; frac = %.2f)" % frac
    return {"a_hz": a, "b_hz_per_harmonic": b, "n_ref": nr, "multiplicative_fraction": frac, "verdict": verdict, "points": len(pts)}


def probe(path, f0_hint, nmax):
    fs, xs = ar.read_wav(path)
    x = xs[:, 0]
    f0 = ar.estimate_f0(x, fs, f0_hint)
    rows = ar.cluster_stats(x, fs, f0, nmax=nmax)
    res = classify(rows)
    res["f0_hz"] = f0
    res["envelope"] = ar.envelope_stats(x, fs, f0)
    # symmetry of the cluster (two detuned VCOs = symmetric +-D around the main line)
    asym = [r["asymmetry"] for r in rows if r["sideband_db"] is not None]
    res["mean_asymmetry"] = float(np.mean(asym)) if asym else 0.0
    res["rows"] = rows
    return res


def report(res):
    print("f0 = %.4f Hz" % res["f0_hz"])
    if "a_hz" in res:
        print("offset(n) = %.4f Hz + %.4f Hz * n   (n_ref = %.0f, %d harmonics used)" % (res["a_hz"], res["b_hz_per_harmonic"], res["n_ref"], res["points"]))
        print("multiplicative fraction = %.2f" % res["multiplicative_fraction"])
    print("verdict: %s" % res["verdict"])
    if "reason" in res:
        print("  " + res["reason"])
    e = res["envelope"]
    print("cluster asymmetry (0 = symmetric +-D, +1 = only upper side): %+.2f" % res["mean_asymmetry"])
    print("fundamental-band envelope: CV %.3f, dominant modulation %.3f Hz (beating / LFO rate)" % (e["envelope_cv"], e["dominant_modulation_hz"]))
    print("per harmonic (n: sideband dB re main, RMS offset Hz, cents):")
    for r in res["rows"][:16]:
        if r["sideband_db"] is None:
            print("  n=%2d: clean" % r["n"])
        else:
            print("  n=%2d: %6.1f dB  %8.3f Hz  %7.3f cents" % (r["n"], r["sideband_db"], r["rms_offset_hz"], r["rms_offset_cents"]))


# ------------------------------------------------------------------------------------------------------------------------------
# self test on synthetic H-DOPPLER / H-SSB signals
# ------------------------------------------------------------------------------------------------------------------------------
def synth(kind, fs=48000, secs=40.0, f0=65.406, depth_cents=6.0, lfo_hz=0.35, gain=0.5, seed=1):
    """Saw (1/n harmonics) plus two shifted copies. kind = 'doppler' (ratio 1 +- rho(t)) or 'ssb' (+- D(t) Hz on every harmonic)."""
    rng = np.random.default_rng(seed)
    n = int(fs * secs)
    t = np.arange(n) / fs
    nh = int((0.45 * fs) // (f0 * (1 + 0.02)))
    mod = 1.0 + 0.4 * np.sin(2 * np.pi * lfo_hz * t + rng.uniform(0, 6.28))
    rho0 = 2.0 ** (depth_cents / 1200.0) - 1.0
    d0 = rho0 * f0 * 4.0  # SSB shift chosen equal to the Doppler offset at harmonic 4
    y = np.zeros(n)
    for h in range(1, nh + 1):
        amp = 1.0 / h
        y += amp * np.sin(2 * np.pi * h * f0 * t)
        for sgn in (+1.0, -1.0):
            if kind == "doppler":
                ratio = 1.0 + sgn * rho0 * mod
                ph = 2 * np.pi * h * f0 * np.cumsum(ratio) / fs
            else:
                shift = sgn * d0 * mod
                ph = 2 * np.pi * (h * f0 * t + np.cumsum(shift) / fs)
            y += gain * 0.5 * amp * np.sin(ph + rng.uniform(0, 6.28))
    y /= np.max(np.abs(y)) * 1.1
    y += rng.normal(0, 1e-5, n)
    return fs, y


def selftest():
    ok = True
    for kind, expect in (("doppler", "H-DOPPLER"), ("ssb", "H-SSB")):
        fs, y = synth(kind)
        f0 = ar.estimate_f0(y, fs, 65.406)
        rows = ar.cluster_stats(y, fs, f0, nmax=24)
        res = classify(rows)
        good = res["verdict"].startswith(expect)
        print("synthetic %-8s -> %s   (frac %.2f)  %s" % (kind, res["verdict"], res.get("multiplicative_fraction", float("nan")), "OK" if good else "FAIL"))
        ok = ok and good
    print("selftest %s" % ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?")
    ap.add_argument("--f0", type=float, default=None)
    ap.add_argument("--nmax", type=int, default=30, help="highest harmonic to analyse")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.wav:
        ap.error("wav file required (or --selftest)")
    report(probe(a.wav, a.f0, a.nmax))
    return 0


if __name__ == "__main__":
    sys.exit(main())
