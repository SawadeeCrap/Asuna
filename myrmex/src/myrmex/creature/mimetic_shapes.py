"""Forms of the Mimetic line (creatures v9-v13): configurations of the same finite material, not models.

Every form is a map from each node's fixed random coordinates ``U`` (n x 3 in [0, 1]) to a place in the
body frame (x forward, y left, z up); the organism moves between forms through its latent weights and
spring dynamics, so a form is never switched on - material flows into it.

    STREAM  a dense head and a dozen long wavy filaments streaming behind (distributed fluid flight)
    LANCE   a long needle core with swept blades along it (elongated, high-speed, directional)
    SHARDS  a porous drifting cloud of clumps around a thin core (dispersion, camouflage)
    SWEEP   flame-like curved blades fanning back from a point (high-velocity cutting)
    CRAWL   a segmented body on 4-8 multi-joint legs whose clawed feet grip the terrain
"""
from __future__ import annotations

import math

import numpy as np

from .polyalloy import attractor_shape

MIMETIC_SHAPES = ("STREAM", "LANCE", "SHARDS", "SWEEP", "CRAWL")


def _fib_dirs(k: int) -> np.ndarray:
    i = np.arange(k) + 0.5
    phi = np.arccos(1 - 2 * i / k)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_CLUMPS = _fib_dirs(14)


def mimetic_shape(name: str, U: np.ndarray, s: float, elong: float, ph: dict, ground: float) -> np.ndarray:
    u, v, w = U[:, 0], U[:, 1], U[:, 2]
    t, mech, pulse = ph["t"], ph["mech"], ph["pulse"]
    if name == "STREAM":
        head = u < 0.28
        P = attractor_shape("CORE", U, s * 0.5, 1.4) + np.array([0.55 * s, 0.0, 0.0])
        F = 11
        k = np.floor(v * F)
        a = np.clip((u - 0.28) / 0.72, 0.0, 1.0)                      # 0 at the head .. 1 at the far end
        L = 3.2 * s * elong
        r = 0.08 * s + 0.62 * s * a ** 1.25 * (1.0 + 0.25 * pulse)
        th = 2 * np.pi * k / F + 1.3 * a + 0.35 * np.sin(0.4 * t + k)
        wave = 0.22 * s * a * np.sin(2 * np.pi * (1.5 * a - (0.7 + 0.8 * mech) * t) + 1.7 * k)
        cy, cz = np.cos(th), np.sin(th)
        y = cy * r - cz * wave + (w - 0.5) * 0.03 * s
        z = 0.55 * (cz * r + cy * wave) + (w - 0.5) * 0.03 * s
        tail = np.stack([0.45 * s - L * a, y, z], 1)
        return np.where(head[:, None], P, tail)
    if name == "LANCE":
        spine = u < 0.34
        a = u / 0.34
        x = -1.3 * s * elong + 3.0 * s * elong * a                     # the needle, tip forward
        r = 0.14 * s * np.clip(1.0 - a, 0.0, 1.0) ** 0.7
        ang = 2 * np.pi * v
        core = np.stack([x, r * np.cos(ang), r * np.sin(ang)], 1)
        B = 6
        k = np.floor(v * B)
        b = (v * B) % 1.0                                              # along the blade, root .. tip
        pair = np.floor(k / 2)
        side = np.where(k % 2 == 0, 1.0, -1.0)
        flare = 0.55 + 0.25 * pulse + 0.12 * mech * math.sin(math.pi * ph["beat"] / 2.0)
        rx = (0.75 - 0.55 * pair) * s * elong
        Lb = (1.15 - 0.18 * pair) * s
        up = np.where(pair == 1, 0.35, -0.12)
        dx, dy, dz = -1.0, side * flare, up * flare
        nrm = np.sqrt(dx * dx + dy * dy + dz * dz)
        blade = np.stack([rx + dx / nrm * Lb * b - 0.35 * s * b * b,
                          dy / nrm * Lb * b + (w - 0.5) * 0.02 * s,
                          dz / nrm * Lb * b + (w - 0.5) * 0.09 * s * (1.0 - b)], 1)
        return np.where(spine[:, None], core, blade)
    if name == "SHARDS":
        core = u < 0.18
        P = attractor_shape("CORE", U, s * 0.35, 1.0)
        j = np.floor(v * len(_CLUMPS)).astype(int)
        rad = s * (0.95 + 0.6 * ((j * 0.618) % 1.0)) * (1.0 + 0.35 * pulse)
        c = _CLUMPS[j] * rad[:, None] * np.array([1.4, 1.1, 0.8])
        spin = 0.15 * t + j
        c = np.stack([c[:, 0] * np.cos(0.3 * spin) - c[:, 1] * np.sin(0.3 * spin),
                      c[:, 0] * np.sin(0.3 * spin) + c[:, 1] * np.cos(0.3 * spin), c[:, 2]], 1)
        d = np.stack([np.cos(2 * np.pi * w), np.sin(2 * np.pi * w), np.cos(2 * np.pi * u) * 0.6], 1)
        flake = c + d * (0.12 * s * ((u * 7.0) % 1.0))[:, None]
        return np.where(core[:, None], P, flake)
    if name == "SWEEP":
        root = u < 0.14
        P = attractor_shape("CORE", U, s * 0.3, 1.3) + np.array([0.85 * s, 0.0, 0.0])
        K = 7
        k = np.floor(v * K)
        b = np.clip((u - 0.14) / 0.86, 0.0, 1.0)
        phi = -1.0 + 2.0 * k / (K - 1)                                 # fan position -1 .. 1
        L = (2.5 - 0.9 * np.abs(phi)) * s * elong
        spread = (0.2 * s + 1.15 * s * b ** 1.35) * (1.0 + 0.3 * pulse)
        curl = 0.18 * s * b * np.sin(np.pi * 1.3 * b + 1.2 * t * (0.5 + mech) + k)
        sheet = (w - 0.5) * 0.14 * s * (1.0 - b) ** 0.6
        y = np.sin(phi * 1.2) * spread + sheet * np.cos(phi)
        z = 0.42 * np.cos(phi * 1.7) * spread * (0.5 + 0.5 * b) - 0.25 * s * b + curl + sheet * np.sin(phi) * 0.5
        blades = np.stack([0.8 * s - L * b, y, z], 1)
        return np.where(root[:, None], P, blades)
    # CRAWL: segmented body + legs (ph["legs"] 4..8) gripping the ground; ph["ground_fn"](x, y) -> local ground z
    n_legs = int(ph.get("legs", 6))
    half = max(2, n_legs // 2)
    gfn = ph.get("ground_fn")
    body = u < 0.45
    seg = np.floor(v * 3.0)
    cx = (0.6 - 0.6 * seg) * s * elong
    rr = np.array([0.34, 0.28, 0.22])[seg.astype(int)] * s
    th, pz = 2 * np.pi * w, np.arccos(1 - 2 * ((u / 0.45 * 5.0) % 1.0))
    lobe = np.stack([cx + 0.9 * rr * np.sin(pz) * np.cos(th), 0.8 * rr * np.sin(pz) * np.sin(th), 0.6 * rr * np.cos(pz)], 1)
    k = np.floor(v * n_legs)
    side = np.where(k < half, 1.0, -1.0)
    i = k % half
    ax = (0.55 - 1.2 * i / max(half - 1, 1)) * s * elong
    phase = ph["gait"] + np.where((i + (side > 0)) % 2 == 0, 0.0, np.pi)      # alternating tripods
    lift = np.maximum(0.0, np.sin(phase)) * 0.2 * s
    swing = np.cos(phase) * 0.22 * s
    f = np.clip((u - 0.45) / 0.55, 0.0, 1.0)
    spread = 1.0 + 0.25 * pulse
    hip = np.stack([ax, side * 0.25 * s, np.zeros_like(u)], 1)
    knee = np.stack([ax + 0.4 * swing, side * 0.85 * s * spread, 0.42 * s + 0.6 * lift], 1)
    fx, fy = 1.25 * ax + swing, side * 1.25 * s * spread
    gz = gfn(fx, fy) if gfn is not None else np.full_like(u, ground)
    foot = np.stack([fx, fy, gz + lift], 1)
    claw = foot + np.stack([0.12 * s * np.ones_like(u), -side * 0.1 * s, -0.05 * s * np.ones_like(u)], 1)
    t1 = np.clip(f / 0.45, 0, 1)[:, None]
    t2 = np.clip((f - 0.45) / 0.4, 0, 1)[:, None]
    t3 = np.clip((f - 0.85) / 0.15, 0, 1)[:, None]
    leg = np.where((f < 0.45)[:, None], hip + (knee - hip) * t1,
                   np.where((f < 0.85)[:, None], knee + (foot - knee) * t2, foot + (claw - foot) * t3))
    return np.where(body[:, None], lobe, leg)


__all__ = ["MIMETIC_SHAPES", "mimetic_shape"]
