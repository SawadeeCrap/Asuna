"""Fit a humanoid deform skeleton to a biped :class:`Morphology`.

Geometry decides wherever it is unambiguous (ankle = narrowest ring above the
foot, wrist = narrowest ring behind the hand, elbow/knee = real bend points,
shoulder = where the arm tube meets the torso).  Anatomical proportions
(Winter / Drillis–Contini) fill in what generated meshes hide: hips inside a
pencil skirt, the spine inside a coat, a neck under hair.

Bone names follow the Unreal mannequin convention (``pelvis``,
``spine_01``, ``thigh_l`` ...), which keeps the rig familiar and retargetable.
"""
from __future__ import annotations

import numpy as np

from .morphology import Limb, Morphology
from .rigdesc import ChainDesc, RigDescription

UP = np.array([0.0, 0.0, 1.0])


class _Ctx:
    def __init__(self, m: Morphology):
        self.m = m
        self.V = m.verts
        self.H = m.height
        self.fwd = np.asarray(m.forward, dtype=float)
        self.left = np.cross(UP, self.fwd)
        self.notes: list[str] = []
        arm_idx = [np.concatenate([m.arcs[a].verts for a in l.arcs]) for l in m.limbs_by_role("arm")]
        self.arm_mask = np.zeros(len(self.V), bool)
        for idx in arm_idx:
            self.arm_mask[idx] = True

    def lat(self, p) -> float:
        return float(np.dot(p, self.left))

    def dep(self, p) -> float:
        return float(np.dot(p, self.fwd))

    def compose(self, lat: float, dep: float, z: float) -> np.ndarray:
        return self.left * lat + self.fwd * dep + UP * z

    def torso_slice(self, z: float, half: float | None = None, lateral_limit: float | None = None):
        half = half if half is not None else 0.012 * self.H
        lim = lateral_limit if lateral_limit is not None else 0.16 * self.H
        m = (np.abs(self.V[:, 2] - z) < half) & (~self.arm_mask)
        P = self.V[m]
        if len(P) == 0:
            return None
        P = P[np.abs(P @ self.left) < lim]
        return P if len(P) >= 4 else None

    def torso_center(self, z: float, posterior: float = 0.0) -> np.ndarray:
        """Centroid of the torso ring at height z, shifted toward the back by ``posterior`` * half-depth."""
        P = self.torso_slice(z)
        if P is None:
            return self.compose(0.0, 0.0, z)
        d = P @ self.fwd
        l = P @ self.left
        lat = 0.5 * (l.max() + l.min())
        dep_c = 0.5 * (d.max() + d.min())
        half_depth = 0.5 * (d.max() - d.min())
        return self.compose(lat, dep_c - posterior * half_depth, z)


def _ring(points: np.ndarray, z: float, half: float) -> tuple[np.ndarray, float, float] | None:
    m = np.abs(points[:, 2] - z) < half
    if m.sum() < 5:
        return None
    P = points[m]
    c = P.mean(axis=0)
    c[2] = z
    r = float(np.linalg.norm((P - c)[:, :2], axis=1).mean())
    ext = float(np.ptp(P[:, :2], axis=0).max())
    return c, r, ext


def _limb_verts(m: Morphology, limb: Limb) -> np.ndarray:
    return np.concatenate([m.arcs[a].verts for a in limb.arcs])


def _fit_foot(c: _Ctx, leg: Limb) -> dict:
    V, H = c.V, c.H
    LV = V[_limb_verts(c.m, leg)]
    toe = max(leg.tips, key=lambda p: c.dep(p))
    heel = min(leg.tips, key=lambda p: c.dep(p))
    if len(leg.tips) == 1:
        # Only one ground tip: estimate heel from the lowest-rear foot vertices.
        low = LV[LV[:, 2] < 0.02 * H]
        heel = low[np.argmin(low @ c.fwd)] if len(low) else toe - c.fwd * 0.14 * H
    # Ankle: narrowest ring above the foot.
    best = None
    for z in np.arange(0.03 * H, 0.15 * H, 0.004 * H):
        res = _ring(LV, z, 0.006 * H)
        if res is None:
            continue
        cen, r, ext = res
        if ext > 0.075 * H:
            continue  # still cutting through the foot
        score = r + 0.15 * max(0.0, z - 0.09 * H)  # mild preference for lower candidates
        if best is None or score < best[0]:
            best = (score, cen)
    if best is None:
        ankle = 0.5 * (toe + heel)
        ankle[2] = 0.05 * H
        c.notes.append(f"{leg.name}: ankle from prior")
    else:
        ankle = best[1]
    foot_len = max(c.dep(toe) - c.dep(heel), 0.08 * H)
    # Ball of the foot ~72% from heel to toe; height = lower third of the sole slab.
    ball_dep = c.dep(heel) + 0.72 * foot_len
    low_all = LV[LV[:, 2] < float(ankle[2])]
    slab = low_all[np.abs(low_all @ c.fwd - ball_dep) < 0.01 * H]
    if len(slab):
        zlo, zhi = float(slab[:, 2].min()), float(slab[:, 2].max())
        zhi = min(zhi, zlo + 0.06 * H)
        ball = c.compose(float((slab @ c.left).mean()), ball_dep, zlo + 0.35 * (zhi - zlo))
    else:
        ball = c.compose(c.lat(toe), ball_dep, 0.012 * H)
    low = LV[LV[:, 2] < float(ankle[2]) * 0.6 + 0.01 * H]
    tslab = low[np.abs(low @ c.fwd - (c.dep(toe) - 0.015 * H)) < 0.01 * H]
    toe_end = toe.copy()
    if len(tslab):
        toe_end[2] = float(tslab[:, 2].min() + 0.4 * np.ptp(tslab[:, 2]))
        toe_end = c.compose(float((tslab @ c.left).mean()), c.dep(toe), toe_end[2])
    return {"toe": toe_end, "toe_tip": toe, "heel": heel, "ankle": ankle, "ball": ball,
            "foot_length": foot_len, "leg_verts": LV}


def _line_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    d = vt[0]
    if d[2] < 0:
        d = -d
    return c, d


def _point_on_line_at_z(c: np.ndarray, d: np.ndarray, z: float) -> np.ndarray:
    if abs(d[2]) < 1e-6:
        return np.array([c[0], c[1], z])
    t = (z - c[2]) / d[2]
    return c + d * t


def _bend_point(poly: np.ndarray, a: np.ndarray, b: np.ndarray, lo: float, hi: float):
    """Point of the polyline farthest from segment a-b within arc-length fraction [lo, hi]."""
    if len(poly) < 3:
        return None, 0.0, None
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= max(s[-1], 1e-9)
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-9:
        return None, 0.0, None
    u = ab / L
    best, bestd, bestf = None, 0.0, None
    for p, fr in zip(poly, s):
        if fr < lo or fr > hi:
            continue
        v = p - a
        d = np.linalg.norm(v - u * float(np.dot(v, u)))
        if d > bestd:
            best, bestd, bestf = p, d, fr
    return best, bestd, bestf


def _point_at_fraction(poly: np.ndarray, fr: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = fr * s[-1]
    k = int(np.searchsorted(s, target))
    k = min(max(k, 1), len(poly) - 1)
    t = (target - s[k - 1]) / max(s[k] - s[k - 1], 1e-9)
    return poly[k - 1] + (poly[k] - poly[k - 1]) * t


def fit_humanoid(m: Morphology) -> RigDescription:
    c = _Ctx(m)
    H, V = c.H, c.V
    legs = {l.side: l for l in m.limbs_by_role("leg")}
    arms = {l.side: l for l in m.limbs_by_role("arm")}
    heads = m.limbs_by_role("head")
    if set(legs) != {"L", "R"}:
        raise ValueError(f"humanoid fit needs a left and a right leg, found {sorted(legs)}")

    # ---------------------------------------------------------------- feet
    feet = {s: _fit_foot(c, legs[s]) for s in ("L", "R")}
    ankle_z = 0.5 * (feet["L"]["ankle"][2] + feet["R"]["ankle"][2])

    # ---------------------------------------------------------------- hips
    heel_rise = max(0.0, ankle_z - 0.045 * H)
    z_anat = ankle_z + 0.49 * (H - heel_rise)
    best_w, z_w = -1.0, None
    for z in np.arange(0.42 * H, 0.64 * H, 0.005 * H):
        P = c.torso_slice(z, lateral_limit=0.2 * H)
        if P is None:
            continue
        w = float(np.ptp(P @ c.left))
        if w > best_w:
            best_w, z_w = w, z
    if z_w is not None and abs(z_w - z_anat) < 0.08 * H:
        hip_z = 0.5 * (z_w + z_anat)
    else:
        hip_z = z_anat
        c.notes.append("hip height from anatomical prior (widest-hip cue disagreed)")
    hip_center = c.torso_center(hip_z, posterior=0.1)

    hips = {}
    for s, sign in (("L", 1.0), ("R", -1.0)):
        leg = legs[s]
        cl = leg.centerline
        sel = cl[(cl[:, 2] > ankle_z + 0.08 * H) & (cl[:, 2] < hip_z - 0.06 * H)]
        if len(sel) >= 3:
            lc, ld = _line_fit(sel)
            at = _point_on_line_at_z(lc, ld, hip_z)
            lat = abs(c.lat(at) - c.lat(hip_center))
        else:
            lat = 0.055 * H
            c.notes.append(f"{leg.name}: hip lateral offset from prior")
        lat = float(np.clip(lat, 0.035 * H, 0.075 * H))
        hips[s] = c.compose(c.lat(hip_center) + sign * lat, c.dep(hip_center), hip_z)

    # ---------------------------------------------------------------- knees
    knees = {}
    for s in ("L", "R"):
        a, h = feet[s]["ankle"], hips[s]
        guess = a + 0.5 * (h - a)
        ring = _ring(feet[s]["leg_verts"], float(guess[2]), 0.01 * H)
        knee = guess.copy()
        if ring is not None and ring[2] < 0.12 * H:
            knee[:2] = ring[0][:2]
        # Guarantee a small forward bend so IK has an unambiguous plane.
        axis = (h - a) / np.linalg.norm(h - a)
        rel = knee - a
        perp = rel - axis * float(np.dot(rel, axis))
        if float(np.dot(perp, c.fwd)) < 0.004 * H:
            knee = knee + c.fwd * (0.004 * H - float(np.dot(perp, c.fwd)))
        knees[s] = knee

    # ---------------------------------------------------------------- shoulders / arms
    sh_z_list, arm_fit = [], {}
    for s, sign in (("L", 1.0), ("R", -1.0)):
        if s not in arms:
            continue
        arm = arms[s]
        cl = arm.centerline                    # fingertip -> shoulder
        junction = V[arm.attach_vertex]
        sh_z = float(junction[2] + 0.03 * H)
        n_top = max(3, int(0.4 * len(cl)))
        uc, ud = _line_fit(cl[-n_top:])
        sh = _point_on_line_at_z(uc, ud, sh_z)
        lat = float(np.clip(abs(c.lat(sh)), 0.08 * H, 0.13 * H))
        tc = c.torso_center(sh_z, posterior=0.15)
        sh = c.compose(sign * lat, c.dep(tc), sh_z)
        # Wrist: narrowest slice 0.07..0.13 H behind the fingertips.
        seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
        sarc = np.concatenate([[0.0], np.cumsum(seg)])
        cand = [(arm.radii[k], k) for k in range(len(cl)) if 0.07 * H <= sarc[k] <= 0.13 * H]
        k_w = min(cand)[1] if cand else int(np.searchsorted(sarc, 0.1 * H))
        k_w = min(k_w, len(cl) - 2)
        wrist = cl[k_w].copy()
        upper = np.concatenate([cl[k_w:], [sh]], axis=0)       # wrist -> shoulder
        elbow = _point_at_fraction(upper, 0.44)
        bp, bd, bf = _bend_point(upper, wrist, sh, 0.28, 0.62)
        if bp is not None and bd > 0.012 * H:
            elbow = bp
        arm_fit[s] = {"shoulder": sh, "elbow": elbow, "wrist": wrist, "hand_tip": arm.tip.copy()}
        sh_z_list.append(sh_z)

    sh_z = float(np.mean(sh_z_list)) if sh_z_list else hip_z + 0.29 * H

    # ---------------------------------------------------------------- spine / neck / head
    neck_z = sh_z + 0.02 * H
    raw = [c.torso_center(hip_z + t * (neck_z - hip_z), posterior=0.3) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    # Smooth the depth/lateral profile (narrow waists and busts make raw centroids zig-zag).
    zs = np.array([p[2] for p in raw])
    lat = np.array([c.lat(p) for p in raw])
    dep = np.array([c.dep(p) for p in raw])
    cl_ = np.polyfit(zs, lat, 1)
    cd_ = np.polyfit(zs, dep, 2)
    smooth = [c.compose(float(np.polyval(cl_, z)), float(np.polyval(cd_, z)), float(z)) for z in zs]
    spine_pts = smooth[1:4]
    neck_base = smooth[4]
    head_tip = heads[0].tip.copy() if heads else neck_base + UP * 0.18 * H
    skull_z = neck_z + 0.075 * H
    if heads:
        hc = heads[0].centerline
        k = int(np.argmin(np.abs(hc[:, 2] - skull_z)))
        skull = hc[k].copy()
        skull[2] = skull_z
        # Pull the skull base toward the neck line (hair can drag the centreline back).
        skull[:2] = 0.5 * skull[:2] + 0.5 * neck_base[:2]
    else:
        skull = neck_base + UP * 0.075 * H
    head_top = head_tip if head_tip[2] > skull[2] + 0.05 * H else skull + UP * 0.15 * H

    # ---------------------------------------------------------------- assemble
    rd = RigDescription("biped", H, forward=[float(x) for x in c.fwd])
    fwd = c.fwd
    rd.add_bone("root", [0.0, 0.0, 0.0], (fwd * 0.12 * H).tolist(), None, UP, "root", deform=False)
    pelvis_head = c.compose(c.lat(hip_center), c.dep(hip_center), hip_z)
    rd.add_bone("pelvis", pelvis_head, spine_pts[0], "root", fwd, "pelvis")
    rd.add_bone("spine_01", spine_pts[0], spine_pts[1], "pelvis", fwd, "spine", connect=True)
    rd.add_bone("spine_02", spine_pts[1], spine_pts[2], "spine_01", fwd, "spine", connect=True)
    rd.add_bone("spine_03", spine_pts[2], neck_base, "spine_02", fwd, "chest", connect=True)
    rd.add_bone("neck_01", neck_base, skull, "spine_03", fwd, "neck", connect=True)
    rd.add_bone("head", skull, head_top, "neck_01", fwd, "head", connect=True)
    for s, sign in (("L", 1.0), ("R", -1.0)):
        sl = s.lower()
        f = feet[s]
        rd.add_bone(f"thigh_{sl}", hips[s], knees[s], "pelvis", fwd, "thigh", s, f"leg_{sl}")
        rd.add_bone(f"calf_{sl}", knees[s], f["ankle"], f"thigh_{sl}", fwd, "calf", s, f"leg_{sl}", connect=True)
        rd.add_bone(f"foot_{sl}", f["ankle"], f["ball"], f"calf_{sl}", UP, "foot", s, f"leg_{sl}", connect=True)
        rd.add_bone(f"ball_{sl}", f["ball"], f["toe"], f"foot_{sl}", UP, "toe", s, f"leg_{sl}", connect=True)
        rd.chains.append(ChainDesc(f"leg_{sl}", "leg", [f"thigh_{sl}", f"calf_{sl}", f"foot_{sl}", f"ball_{sl}"], s,
                                   {"foot_length": float(f["foot_length"])}))
        for key in ("heel", "toe_tip", "ankle", "ball"):
            rd.landmarks[f"{key}_{sl}"] = [float(x) for x in f[key]]
        rd.landmarks[f"hip_{sl}"] = [float(x) for x in hips[s]]
        rd.landmarks[f"knee_{sl}"] = [float(x) for x in knees[s]]
        if s in arm_fit:
            a = arm_fit[s]
            clav_head = neck_base + c.left * sign * 0.02 * H + fwd * 0.015 * H - UP * 0.01 * H
            rd.add_bone(f"clavicle_{sl}", clav_head, a["shoulder"], "spine_03", fwd, "clavicle", s, f"arm_{sl}")
            rd.add_bone(f"upperarm_{sl}", a["shoulder"], a["elbow"], f"clavicle_{sl}", fwd, "upperarm", s, f"arm_{sl}", connect=True)
            rd.add_bone(f"lowerarm_{sl}", a["elbow"], a["wrist"], f"upperarm_{sl}", fwd, "lowerarm", s, f"arm_{sl}", connect=True)
            rd.add_bone(f"hand_{sl}", a["wrist"], a["hand_tip"], f"lowerarm_{sl}", fwd, "hand", s, f"arm_{sl}", connect=True)
            rd.chains.append(ChainDesc(f"arm_{sl}", "arm", [f"clavicle_{sl}", f"upperarm_{sl}", f"lowerarm_{sl}", f"hand_{sl}"], s))
            for key in ("shoulder", "elbow", "wrist", "hand_tip"):
                rd.landmarks[f"{key}_{sl}"] = [float(x) for x in a[key]]
    rd.chains.insert(0, ChainDesc("spine", "spine", ["pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head"]))
    rd.landmarks["hip_center"] = [float(x) for x in pelvis_head]
    rd.landmarks["neck_base"] = [float(x) for x in neck_base]
    rd.landmarks["head_top"] = [float(x) for x in head_top]
    rd.params.update({"heel_rise": float(heel_rise), "hip_height": float(hip_z),
                      "shoulder_height": float(sh_z), "ankle_height": float(ankle_z)})
    rd.notes.extend(m.notes + c.notes)
    return rd
