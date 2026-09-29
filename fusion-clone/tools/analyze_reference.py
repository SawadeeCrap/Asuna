#!/usr/bin/env python3
"""Spectrum / sideband / drift report for a recording of a (quasi-)periodic oscillator.

Meant for recordings of a real Erica Synths Fusion VCO2 (see docs/REFERENCE_PROTOCOL.md) but works on any monophonic oscillator recording,
including the output of the Fusion Clone module itself.

    python3 tools/analyze_reference.py rec.wav --f0 65.406 [--out report.json] [--plot report.png] [--compare other.wav]

Reports: measured f0 and its drift, harmonic levels (dB re the fundamental), sub-harmonic lines (period doubling), THD and even/odd ratio,
DC and noise floor, and the detune cluster around each harmonic (sideband level, RMS offset in Hz and in cents, symmetry).
Only numpy is required; matplotlib is used for --plot if installed.
"""
import argparse
import json
import math
import struct
import sys

import numpy as np


# ------------------------------------------------------------------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------------------------------------------------------------------
def read_wav(path):
    """Minimal RIFF/WAVE reader (16/24/32-bit PCM, 32-bit float). Returns (sample_rate, float64 array of shape (frames, channels))."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("%s: not a RIFF/WAVE file" % path)
    pos = 12
    fmt = None
    raw = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and size >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            raw = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise ValueError("%s: missing fmt or data chunk" % path)
    tag, ch, sr, bits = fmt
    if tag == 3 and bits == 32:
        x = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(raw[: len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - 0x1000000, v)
        x = v.astype(np.float64) / 8388608.0
    elif tag == 1 and bits == 32:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError("%s: unsupported WAV format tag=%d bits=%d" % (path, tag, bits))
    n = len(x) // ch
    return sr, x[: n * ch].reshape(n, ch)


def hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


# ------------------------------------------------------------------------------------------------------------------------------
# pitch
# ------------------------------------------------------------------------------------------------------------------------------
def _spectrum(x, fs, pad=4, nmax=1 << 22):
    """Hann-windowed, zero-padded magnitude spectrum of the central part of x (at most nmax samples)."""
    if len(x) > nmax:
        s = (len(x) - nmax) // 2
        x = x[s:s + nmax]
    n = len(x)
    nfft = 1
    while nfft < n * pad:
        nfft <<= 1
    w = hann(n)
    X = np.fft.rfft((x - np.mean(x)) * w, nfft)
    return X, fs / nfft, np.sum(w)


def estimate_f0(x, fs, hint=None, nharm=12):
    """Fundamental by maximising the harmonic sum of the magnitude spectrum around `hint` (or around the strongest low peak)."""
    X, df, wsum = _spectrum(x, fs)
    mag = np.abs(X)
    if hint is None:
        lo, hi = int(15.0 / df), int(4000.0 / df)
        hint = (lo + int(np.argmax(mag[lo:hi]))) * df
    grid = hint * np.exp(np.linspace(-0.06, 0.06, 2401))
    best_f, best_s = hint, -1.0
    for f in grid:
        idx = np.arange(1, nharm + 1) * f / df
        idx = idx[idx < len(mag) - 2]
        s = float(np.sum(np.interp(idx, np.arange(len(mag)), mag)))
        if s > best_s:
            best_s, best_f = s, f
    # parabolic refinement on the log-grid
    g = best_f * np.exp(np.linspace(-1e-3, 1e-3, 201))
    sc = []
    for f in g:
        idx = np.arange(1, nharm + 1) * f / df
        idx = idx[idx < len(mag) - 2]
        sc.append(float(np.sum(np.interp(idx, np.arange(len(mag)), mag))))
    return float(g[int(np.argmax(sc))])


def line_amplitude(x, fs, f, w=None):
    """Complex amplitude of the component at frequency f (Hann-windowed projection)."""
    n = len(x)
    if w is None:
        w = hann(n)
    t = np.arange(n) / fs
    return 2.0 * np.sum(w * x * np.exp(-2j * np.pi * f * t)) / np.sum(w)


def f0_track(x, fs, f0, frame_s=1.0, hop_s=0.5):
    """f0 (Hz) per frame from the strongest peak near f0 (parabolic interpolation on a zero-padded FFT)."""
    n = int(frame_s * fs)
    hop = int(hop_s * fs)
    out = []
    for s in range(0, len(x) - n + 1, hop):
        X, df, _ = _spectrum(x[s:s + n], fs, pad=8)
        mag = np.abs(X)
        lo, hi = int(f0 * 0.9 / df), int(f0 * 1.1 / df)
        k = lo + int(np.argmax(mag[lo:hi]))
        a, b, c = np.log(mag[k - 1] + 1e-30), np.log(mag[k] + 1e-30), np.log(mag[k + 1] + 1e-30)
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        out.append((k + d) * df)
    return np.array(out)


# ------------------------------------------------------------------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------------------------------------------------------------------
def harmonic_report(x, fs, f0, nmax=60):
    n = len(x)
    w = hann(n)
    a1 = line_amplitude(x, fs, f0, w)
    nh = int(min(nmax, (0.48 * fs) // f0))
    harm = []
    for k in range(1, nh + 1):
        a = line_amplitude(x, fs, k * f0, w)
        harm.append(a)
    harm = np.array(harm)
    lv = 20 * np.log10(np.abs(harm) / (abs(a1) + 1e-30) + 1e-30)
    subs = {}
    for m in range(0, 4):
        f = (2 * m + 1) * f0 / 2.0
        if f < 0.48 * fs:
            subs["%g" % ((2 * m + 1) / 2.0)] = float(20 * np.log10(abs(line_amplitude(x, fs, f, w)) / (abs(a1) + 1e-30) + 1e-30))
    p = np.abs(harm) ** 2
    thd = float(math.sqrt(np.sum(p[1:]) / (p[0] + 1e-30)))
    even = float(np.sum(p[1::2]))
    odd = float(np.sum(p[0::2]))
    return {
        "fundamental_dbfs": float(20 * np.log10(abs(a1) + 1e-30)),
        "harmonic_levels_db": [float(v) for v in lv],
        "harmonic_phases_deg": [float(np.degrees(np.angle(h * np.exp(-1j * np.angle(a1) * (i + 1))))) for i, h in enumerate(harm)],
        "subharmonic_levels_db": subs,
        "thd_ratio": thd,
        "even_over_odd_energy_ratio": float(even / (odd + 1e-30)),
    }


def noise_report(x, fs, f0):
    X, df, wsum = _spectrum(x, fs, pad=1)
    p = np.abs(X) ** 2
    mask = np.ones(len(p), dtype=bool)
    k = 1
    while k * f0 / 2.0 < 0.5 * fs:
        c = int(round(k * f0 / 2.0 / df))
        half = max(6, int(0.01 * c))
        mask[max(0, c - half):c + half + 1] = False
        k += 1
    mask[:int(10.0 / df) + 1] = False
    fund_bin = int(round(f0 / df))
    fund_pow = np.max(p[fund_bin - 4:fund_bin + 5]) + 1e-30
    floor = np.median(p[mask]) if np.any(mask) else 0.0
    return {"dc": float(np.mean(x)), "noise_floor_db_re_fundamental_line": float(10 * np.log10(floor / fund_pow + 1e-30))}


def cluster_stats(x, fs, f0, nmax=40, span=0.4, guard_hz=None):
    """Detune cluster around each harmonic: sideband level, RMS offset from the main line (Hz and cents), symmetry.

    The full-length Hann spectrum has a resolution of ~1.5 / T. Sidebands closer than `guard_hz` to the main line are treated as part of it.
    """
    X, df, _ = _spectrum(x, fs, pad=2)
    p = np.abs(X) ** 2
    T = len(x) / fs
    if guard_hz is None:
        guard_hz = 6.0 / T
    nh = int(min(nmax, (0.45 * fs) // f0))
    rows = []
    for n in range(1, nh + 1):
        fc = n * f0
        lo, hi = int((fc - span * f0) / df), int((fc + span * f0) / df)
        seg = p[lo:hi]
        freqs = (np.arange(lo, hi)) * df
        km = int(np.argmax(seg))
        fm = freqs[km]
        pm = seg[km]
        off = freqs - fm
        # local noise floor: median of the segment; only bins clearly above it count as sidebands
        floor = np.median(seg)
        side = (np.abs(off) > guard_hz) & (seg > 4.0 * floor)
        if not np.any(side):
            rows.append({"n": n, "main_hz": float(fm), "sideband_db": None, "rms_offset_hz": 0.0, "rms_offset_cents": 0.0, "asymmetry": 0.0})
            continue
        ps = seg[side]
        offs = off[side]
        tot = float(np.sum(ps))
        rms = math.sqrt(float(np.sum(ps * offs ** 2)) / tot)
        rows.append({
            "n": n,
            "main_hz": float(fm),
            "sideband_db": float(10 * np.log10(tot / (pm + 1e-30) + 1e-30)),
            "rms_offset_hz": rms,
            "rms_offset_cents": float(1200 * math.log2(1 + rms / fm)),
            "asymmetry": float((np.sum(ps[offs > 0]) - np.sum(ps[offs < 0])) / tot),
        })
    return rows


def envelope_stats(x, fs, f0):
    """Envelope of the fundamental band: coefficient of variation and the dominant modulation frequency (beating / LFO)."""
    n = len(x)
    X = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f > 0.85 * f0) & (f < 1.15 * f0)
    z = np.fft.irfft(np.where(band, X, 0.0), n)  # band-limited signal around the fundamental
    # analytic signal via the FFT (Hilbert transformer)
    H = np.zeros(n)
    if n % 2 == 0:
        H[0] = H[n // 2] = 1
        H[1:n // 2] = 2
    else:
        H[0] = 1
        H[1:(n + 1) // 2] = 2
    zc = np.fft.ifft(np.fft.fft(z) * H)
    env = np.abs(zc)
    env = env[int(0.05 * n):int(0.95 * n)]
    cv = float(np.std(env) / (np.mean(env) + 1e-30))
    e = env - np.mean(env)
    E = np.abs(np.fft.rfft(e * hann(len(e))))
    fe = np.fft.rfftfreq(len(e), 1.0 / fs)
    m = (fe > 0.02) & (fe < 20.0)
    dom = float(fe[m][int(np.argmax(E[m]))]) if np.any(m) else 0.0
    return {"envelope_cv": cv, "dominant_modulation_hz": dom}


def analyse(path, f0_hint):
    fs, xs = read_wav(path)
    x = xs[:, 0]
    f0 = estimate_f0(x, fs, f0_hint)
    tr = f0_track(x, fs, f0)
    rep = {"file": path, "sample_rate": fs, "seconds": len(x) / fs, "f0_hz": f0}
    if len(tr) > 1:
        c = 1200 * np.log2(tr / np.median(tr))
        rep["f0_drift_cents"] = {"std": float(np.std(c)), "peak_to_peak": float(np.ptp(c)), "median_hz": float(np.median(tr))}
    rep["harmonics"] = harmonic_report(x, fs, f0)
    rep.update(noise_report(x, fs, f0))
    rep["cluster"] = cluster_stats(x, fs, f0)
    rep["envelope"] = envelope_stats(x, fs, f0)
    return rep, x, fs


def print_report(rep):
    print("%s: %.1f s @ %d Hz, f0 = %.4f Hz" % (rep["file"], rep["seconds"], rep["sample_rate"], rep["f0_hz"]))
    if "f0_drift_cents" in rep:
        d = rep["f0_drift_cents"]
        print("  f0 drift: std %.3f cents, peak-to-peak %.3f cents" % (d["std"], d["peak_to_peak"]))
    h = rep["harmonics"]
    print("  fundamental %.1f dBFS, THD %.3f, even/odd energy %.3f, DC %.2e, noise floor %.1f dB re fundamental line"
          % (h["fundamental_dbfs"], h["thd_ratio"], h["even_over_odd_energy_ratio"], rep["dc"], rep["noise_floor_db_re_fundamental_line"]))
    print("  sub-harmonic lines (dB re f0): " + ", ".join("%s f0: %.1f" % (k, v) for k, v in h["subharmonic_levels_db"].items()))
    print("  harmonic levels (dB re f0), n=1..%d: %s" % (len(h["harmonic_levels_db"]), " ".join("%.0f" % v for v in h["harmonic_levels_db"][:24])))
    e = rep["envelope"]
    print("  fundamental-band envelope: CV %.3f, dominant modulation %.3f Hz" % (e["envelope_cv"], e["dominant_modulation_hz"]))
    print("  detune cluster (n: sideband dB re main / rms offset Hz / cents / asymmetry):")
    for r in rep["cluster"][:12]:
        if r["sideband_db"] is None:
            print("    n=%2d: clean line" % r["n"])
        else:
            print("    n=%2d: %6.1f dB  %8.3f Hz  %7.3f cents  asym %+.2f" % (r["n"], r["sideband_db"], r["rms_offset_hz"], r["rms_offset_cents"], r["asymmetry"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--f0", type=float, default=None, help="expected fundamental in Hz (recommended)")
    ap.add_argument("--out", help="write the report as JSON")
    ap.add_argument("--plot", help="write a PNG plot (needs matplotlib)")
    ap.add_argument("--compare", help="second recording to compare side by side (cluster spread, envelope statistics)")
    a = ap.parse_args()

    rep, x, fs = analyse(a.wav, a.f0)
    print_report(rep)
    if a.compare:
        rep2, _, _ = analyse(a.compare, a.f0)
        print()
        print_report(rep2)
        print("\ncluster RMS offset (cents), n = 1, 2, 4, 8:")
        for name, r in (("A", rep), ("B", rep2)):
            byn = {c["n"]: c for c in r["cluster"]}
            print("  %s: " % name + "  ".join("%s" % (("%.3f" % byn[n]["rms_offset_cents"]) if n in byn else "-") for n in (1, 2, 4, 8)))
        rep["compare"] = rep2
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
        print("wrote", a.out)
    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed: --plot skipped", file=sys.stderr)
            return 0
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        lv = rep["harmonics"]["harmonic_levels_db"]
        ax[0].bar(range(1, len(lv) + 1), lv)
        ax[0].set_title("harmonic levels (dB re f0)")
        X, df, _ = _spectrum(x, fs, pad=2)
        f0 = rep["f0_hz"]
        for n, c in ((1, "C0"), (4, "C1")):
            lo, hi = int((n * f0 - 0.4 * f0) / df), int((n * f0 + 0.4 * f0) / df)
            ax[1].plot((np.arange(lo, hi) * df - n * f0), 20 * np.log10(np.abs(X[lo:hi]) / np.max(np.abs(X[lo:hi])) + 1e-9), c, label="n=%d" % n)
        ax[1].set_title("spectrum around harmonics (Hz offset)")
        ax[1].legend()
        tr = f0_track(x, fs, f0)
        ax[2].plot(1200 * np.log2(tr / np.median(tr)))
        ax[2].set_title("f0 drift (cents)")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=110)
        print("wrote", a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
