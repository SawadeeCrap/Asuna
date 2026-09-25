"""CreatureEngine: behaviour -> forces -> dynamics -> morphology -> state (simulation only, no visuals).

Feedback loop: audio perturbs arousal and forces; arousal and state change morphology;
morphology moves material (mass field) which changes node masses, i.e. inertia; inertia
changes how the next forces move the body; motion leans and stretches the body again.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ..util.rng import RngStreams
from .behavior import BehaviorSystem
from .config import PARAMS, CreatureConfig
from .control import CreatureControlInput, ParameterSet
from .material import MassField
from .morphology import ARCHETYPES, AppendageSystem, Morphology
from .puppet import GloveControl, GloveDriver
from .nodes import APPENDAGE, CORE, PRIMARY, SECONDARY, NodeSystem

log = logging.getLogger("myrmex.creature")
EVENTS = ("MORPHOLOGY_SHIFT", "MASS_REBALANCE", "APPENDAGE_BURST", "COLLAPSE", "RECONSTRUCTION")


def _fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class CreatureState:
    t: float
    behavior: str
    morphology: str
    pos: np.ndarray
    radius: np.ndarray
    stretch: np.ndarray            # (N, 3) ellipsoid scale per node (fins, blades)
    kind: np.ndarray
    anchor: np.ndarray
    com: np.ndarray
    heading: float
    params: dict
    volumes: dict
    surface: float
    glow: float
    arousal: float
    instability: float
    events: list = field(default_factory=list)


class CreatureEngine:
    def __init__(self, cfg: CreatureConfig | None = None):
        self.cfg = cfg = cfg or CreatureConfig()
        mc = cfg.morphology
        self.rs = RngStreams(cfg.seed)
        self.params = ParameterSet(cfg.params)
        self.inp = CreatureControlInput()
        self.t = 0.0
        P, S, A, K = mc.n_primary, mc.n_secondary, mc.max_appendages, mc.segments
        self.n = n = 1 + P + S + A * K
        self.nodes = nodes = NodeSystem(n)
        self.i_primary = np.arange(1, 1 + P)
        self.i_secondary = np.arange(1 + P, 1 + P + S)
        pool = np.arange(1 + P + S, n).reshape(A, K)
        nodes.kind[:] = APPENDAGE
        nodes.kind[0] = CORE
        nodes.kind[self.i_primary] = PRIMARY
        nodes.kind[self.i_secondary] = SECONDARY
        rng = self.rs.stream("shape")
        # Primary directions: an even sphere, perturbed per seed (the creature's "genome").
        d = _fib_sphere(P) + np.array([[rng.normal(0, 0.18 * cfg.variation) for _ in range(3)] for _ in range(P)])
        self.dir_p = d / np.linalg.norm(d, axis=1, keepdims=True)
        nodes.anchor[self.i_primary] = 0
        self.sec_parent = np.array([self.i_primary[i % P] for i in range(S)])
        self.sec_mix = np.array([rng.uniform(0.35, 0.8) for _ in range(S)])
        self.sec_dir = _fib_sphere(S)[rng.permutation(S) if hasattr(rng, "permutation") else np.arange(S)]
        nodes.anchor[self.i_secondary] = self.sec_parent
        for a in range(A):
            ids = pool[a]
            nodes.anchor[ids[0]] = 0
            nodes.anchor[ids[1:]] = ids[:-1]
        self.pool = pool
        self.apps = AppendageSystem(pool, self.rs.stream("appendages"))
        self.morph = Morphology(mc.transition_time, self.rs.stream("morph"))
        self.mass = MassField(mc.total_material, mc.min_core_fraction, mc.size)
        self.beh = BehaviorSystem(self.rs.stream("behavior"), cfg.stage_radius)
        self.ev_rng = self.rs.stream("events")
        self.side_bias = np.sign(self.dir_p[:, 1]) + 0.0           # +1 left, -1 right (coherent asymmetry)
        self.asym_phase = rng.uniform(0, 100)
        self.heading = self.beh.heading
        self.yaw_v = 0.0
        self.lean = np.zeros(2)
        self.pressure = 0.0
        self.pressure_v = 0.0
        self.vol = {"core": mc.total_material}
        self.events: list[tuple[float, str, object]] = []
        self._pending: list[tuple] = []
        h = 0.55 * mc.size
        nodes.pos[:] = np.array([0.0, 0.0, h])
        nodes.pos[self.i_primary] += self.dir_p * 0.2 * mc.size
        nodes.pos[self.i_secondary] += self.sec_dir * 0.25 * mc.size
        nodes.target[:] = nodes.pos
        self.glove = GloveDriver()                         # a hand (Hand Glove) holding the creature
        self.finger_of = {int(n): i % 5 for i, n in enumerate(self.i_primary)}   # primary node -> finger
        self.surface = 0.2
        self.glow = 0.0

    # ------------------------------------------------------------------ API
    def set_glove(self, ctrl: GloveControl | None, shapes: tuple | None = None) -> None:
        """A hand holding the creature: yaw / tilt of the body, fingers = limbs, height, steering, size."""
        self.glove.set(ctrl)

    def set_input(self, inp: CreatureControlInput) -> None:
        self.inp = inp.sanitized()

    def set_parameter(self, name: str, value: float | None) -> bool:
        return self.params.set(name, value)

    def trigger_event(self, name: str, arg=None) -> bool:
        name = name.upper()
        if name not in EVENTS:
            log.warning("unknown creature event %s", name)
            return False
        self._pending.append((name, arg))
        return True

    # ------------------------------------------------------------------ events
    def _apply_event(self, name: str, arg) -> None:
        mc, t = self.cfg.morphology, self.t
        self.events.append((t, name, arg))
        self.events = self.events[-50:]
        if name == "MORPHOLOGY_SHIFT":
            self.morph.set_target(arg if isinstance(arg, str) else self.beh.profile("morph"), t)
        elif name == "APPENDAGE_BURST":
            arch = arg if arg in ARCHETYPES else self.beh.profile("archetype")
            n = 1 + int(self.params["tendril_activity"] * 2.5)
            for _ in range(n):
                i = int(self.ev_rng.randint(0, mc.n_primary - 1))
                self.apps.spawn(arch, int(self.i_primary[i]), self.dir_p[i] + self.ev_rng.normal(0, 0.3), t,
                                scale=self.ev_rng.uniform(0.7, 1.2))
        elif name == "COLLAPSE":
            self.apps.retract_all()
            self.morph.set_target("COLLAPSED", t, 0.9)
            self.nodes.impulse(slice(None), np.array([0.0, 0.0, -2.5]))
        elif name == "RECONSTRUCTION":
            self.morph.set_target("REORGANIZING", t, 1.6)
            self._pending.append(("MORPHOLOGY_SHIFT", self.beh.profile("morph")))
        elif name == "MASS_REBALANCE":
            r = self.ev_rng
            self.side_bias = np.array([r.uniform(-1, 1) for _ in range(len(self.side_bias))])

    # ------------------------------------------------------------------ simulation
    def update(self, dt: float) -> CreatureState:
        if not math.isfinite(dt) or dt <= 0:
            return self.state()
        h = 1.0 / self.cfg.sim_rate
        steps = max(1, min(12, int(round(dt / h))))
        for _ in range(steps):
            self._step(dt / steps)
        return self.state()

    def _step(self, dt: float) -> None:
        self.t += dt
        t, inp, cfg, mc = self.t, self.inp, self.cfg, self.cfg.morphology
        pr = self.params.values()
        # 1. behaviour (autonomous) and its events
        for ev in self.beh.update(dt, t, inp, pr):
            self._pending.append(ev)
        while self._pending:
            self._apply_event(*self._pending.pop(0))
        a = self.beh.mem.arousal
        self.params.auto.update(arousal=a, aggression=0.7 * a if self.beh.state in ("AGGRESSIVE", "HUNTING") else 0.25,
                                surface_activity=min(1.0, self.beh.profile("surface") + 0.4 * inp.amplitude),
                                tendril_activity=self.beh.profile("appendage"))
        pr = self.params.values()
        # 2. morphology + audio pressure (bass inflates, transients compress then rebound)
        mv = self.morph.update(t, pr["mutation"], pr["instability"] + 0.5 * self.beh.mem.agitation)
        spread, elong, flat, lift, lean_m, budget, spike, legs, fins, mnoise = mv
        au = cfg.audio
        w = 2 * math.pi * 2.2
        target_p = au.bass_expansion * 0.35 * inp.bass * (0.5 + pr["expansion"]) - 0.25 * pr["contraction"]
        self.pressure_v += dt * (w * w * (target_p - self.pressure) - 2 * 0.35 * w * self.pressure_v)
        self.pressure += dt * self.pressure_v
        if inp.transient > 0.3:
            k = au.transient_impulse * inp.transient * (0.4 + pr["reactivity"])
            self.pressure_v -= 6.0 * k * dt * 60.0 / cfg.sim_rate
            self.nodes.impulse(0, np.array([0.0, 0.0, -0.8 * k]))
        # 3. appendages: favoured archetype emerges with activity, the rest retracts
        want = budget * (0.4 + 1.2 * pr["tendril_activity"])
        act = self.apps.active
        if len(act) < int(want * mc.max_appendages + 0.5) and self.ev_rng.chance(dt * 0.8):
            i = int(self.ev_rng.randint(0, mc.n_primary - 1))
            arch = "SPIKE" if spike > 0.6 else ("FIN" if fins > 0.5 else ("LIMB" if legs > 0.5 else self.beh.profile("archetype")))
            self.apps.spawn(arch, int(self.i_primary[i]), self.dir_p[i], t, self.ev_rng.uniform(0.7, 1.2))
        elif len(act) > int(want * mc.max_appendages + 1.5) and self.ev_rng.chance(dt * 0.6):
            oldest = min((x for x in act if x.activation > 0), key=lambda x: x.born, default=None)
            if oldest is not None:
                self.apps.retract(oldest)
        self.apps.update(dt, pr["fluidity"])
        # 4. finite material: appendage volume is paid by the core
        va = self.apps.volumes(mc.segments, mc.size)
        asym = pr["asymmetry"]
        tw = math.sin(0.05 * t + self.asym_phase)
        wp = 1.0 + asym * 0.6 * self.side_bias * (0.7 + 0.3 * tw) + pr["mass_shift"] * 0.4 * np.sin(0.3 * t + np.arange(len(self.side_bias)))
        vc, vp, vs, va = self.mass.allocate(wp, np.ones(len(self.i_secondary)), va, 0.30 + 0.25 * min(spread, 1.5) / 1.5, 0.10)
        self.vol = {"core": float(vc), "primary": float(vp.sum()), "secondary": float(vs.sum()),
                    "appendages": float(va.sum()), "total": float(vc + vp.sum() + vs.sum() + va.sum())}
        nd = self.nodes
        dens = 0.5 + pr["density"]
        rad = np.zeros(self.n)
        rad[0] = self.mass.radius(vc) * (1.0 + 0.8 * self.pressure)
        rad[self.i_primary] = self.mass.radius(vp) * (1.0 + 0.5 * self.pressure)
        rad[self.i_secondary] = self.mass.radius(vs)
        rad[self.pool.ravel()] = self.mass.radius(va)
        nd.radius = np.maximum(rad, 0.0)
        vol_n = np.zeros(self.n)
        vol_n[0], vol_n[self.i_primary], vol_n[self.i_secondary], vol_n[self.pool.ravel()] = vc, vp, vs, va
        nd.mass = 0.05 + 12.0 * dens * vol_n                   # inertia follows the material
        # 5. stiffness per level (rigidity vs fluidity, coherent left/right asymmetry)
        rig = (0.6 + 0.9 * pr["rigidity"]) * (1.25 - 0.5 * pr["fluidity"]) * self.beh.profile("stiffness")
        ph = cfg.physics
        nd.freq[0], nd.zeta[0] = ph.core[0] * rig ** 0.5, ph.core[1]
        nd.freq[self.i_primary] = ph.primary[0] * rig * (1.0 - 0.25 * asym * self.side_bias)
        nd.zeta[self.i_primary] = ph.primary[1]
        nd.freq[self.i_secondary], nd.zeta[self.i_secondary] = ph.secondary[0] * rig, ph.secondary[1]
        gc = self.glove.ctrl
        if gc.active:                                      # held: the body follows the hand more tightly
            nd.freq[self.i_primary] *= 1.0 + 0.6 * gc.grip
            nd.freq[self.i_secondary] *= 1.0 + 0.6 * gc.grip
        # 6. locomotion: the core pursues a wandering goal; body yaw follows velocity with inertia
        goal, speed = self.beh.locomotion_goal(dt, nd.pos[0], pr)
        vxy = nd.vel[0, :2]
        if np.linalg.norm(vxy) > 0.08:
            want_yaw = math.atan2(vxy[1], vxy[0])
            dy = (want_yaw - self.heading + math.pi) % (2 * math.pi) - math.pi
            self.yaw_v += dt * (4.0 * dy - 2.5 * self.yaw_v)
        self.heading += dt * self.yaw_v
        R = _rot_z(self.heading)
        if gc.active:                                      # the hand steers the goal sideways
            goal = np.asarray(goal, float).copy()
            goal[:2] += R[:2, 1] * gc.offset[0] * 1.5 * mc.size
        dR = self.glove.begin(dt)
        self.glove.rigid(nd.pos, nd.vel, nd.pos[0].copy(), R @ dR @ R.T, slice(1, None))
        R = R @ self.glove.G
        size = mc.size
        breathe = 0.04 * math.sin(2 * math.pi * (inp.beat / 4.0 if inp.playing else 0.18 * t)) * au.tempo_oscillation
        body_h = size * (0.18 + 0.42 * lift) * (1 - 0.3 * max(0.0, -self.pressure))
        if gc.active:                                      # raise the hand: it rears up; lower it: it crouches
            body_h = max(0.08 * size, body_h + 0.35 * size * gc.offset[1])
        gait = inp.beat * 0.5 if inp.playing else t * speed
        nd.target[0] = np.array([goal[0], goal[1], body_h]) + nd.ext[0] * 0
        # 7. primary masses: spread / elongation / flattening in the body frame, lean from acceleration
        acc = (nd.vel[0] - getattr(self, "_v0", nd.vel[0])) / max(dt, 1e-4)
        self._v0 = nd.vel[0].copy()
        self.lean += (np.clip(acc[:2] * 0.02, -0.4, 0.4) - self.lean) * min(1.0, dt * 3.0)
        scale = np.array([elong, 1.0 / max(elong, 0.3) ** 0.5, flat]) * size * 0.32 * spread * (1 + self.pressure + breathe)
        local = self.dir_p * scale
        if gc.active:                                      # fingers = the primary masses (and their limbs)
            if gc.finger_mode == "limbs":
                ext = np.array([gc.fingers[self.finger_of[int(n)]] for n in self.i_primary])
                local = local * (0.55 + 0.9 * ext * min(1.0, gc.amount) + 0.45 * (1 - min(1.0, gc.amount)))[:, None]
            local = local * gc.scale
        local[:, 0] += lean_m * 0.3 * size * local[:, 2]
        noise_amp = (pr["noise"] * 0.6 + mnoise * 0.4 + self.beh.profile("noise") * 0.3) * 0.06 * size
        ph_i = np.arange(len(self.dir_p)) * 1.7
        local += noise_amp * np.stack([np.sin(0.7 * t + ph_i), np.sin(0.53 * t + 2 * ph_i), np.sin(0.61 * t + 3 * ph_i)], axis=1)
        tgt = nd.pos[0] + local @ R.T
        tgt[:, :2] += (self.lean * size * 0.15)[None, :] * (local[:, 2:3] / size + 0.5)
        if legs > 0.3:                                   # four-point locomotion: lowest masses become feet
            order = np.argsort(self.dir_p[:, 2])[:4]
            for j, i in enumerate(order):
                ph_g = (gait + j * 0.25 + (j % 2) * 0.5) % 1.0
                lift_f = max(0.0, math.sin(2 * math.pi * ph_g)) * 0.12 * size
                foot = tgt[i].copy()
                foot[2] = 0.3 * nd.radius[1 + i] + lift_f
                tgt[i] = tgt[i] + (foot - tgt[i]) * min(1.0, legs)
        nd.target[self.i_primary] = tgt
        # 8. secondary nodes: between a primary and the core, pushed outward, jittering with the surface
        par = nd.pos[self.sec_parent]
        sec = nd.pos[0] + (par - nd.pos[0]) * self.sec_mix[:, None] + self.sec_dir * size * 0.12 * spread
        hf = (0.02 + 0.05 * pr["surface_activity"] + 0.04 * inp.high * au.high_vibration) * size
        sec += hf * np.sin(9.0 * t + np.arange(len(sec))[:, None] * np.array([1.3, 2.1, 0.7]))
        nd.target[self.i_secondary] = sec
        # 9. appendages: follow-the-leader chains with wave, curl, gravity and high-frequency tremble
        self._appendage_targets(t, R, pr, inp)
        nd.step(dt, ph.gravity, ph.max_speed, self.i_primary, ph.repulsion * (0.5 + pr["density"]), ph.ground)
        # 10. surface / energy for presentation
        self.surface += (min(1.0, pr["surface_activity"] + 0.5 * inp.high + 0.4 * inp.transient) - self.surface) * min(1.0, dt * 4)
        self.glow = max(self.glow * math.exp(-dt / 0.35), 0.8 * inp.transient * a)

    def _appendage_targets(self, t: float, R: np.ndarray, pr: dict, inp) -> None:
        nd, K, size = self.nodes, self.cfg.morphology.segments, self.cfg.morphology.size
        for a in self.apps.items:
            ids = a.nodes
            if a.growth <= 0.01:
                nd.target[ids] = nd.pos[a.origin]
                nd.freq[ids], nd.zeta[ids] = 3.0, 0.9
                nd.gravity_w[ids] = 0.0
                continue
            L = a.spec("length") * a.scale * size * a.growth
            gc = self.glove.ctrl
            fext = gc.fingers[self.finger_of.get(int(a.origin), 0)] if gc.active and gc.finger_mode == "limbs" else None
            if fext is not None:                           # an extended finger stretches its limb
                L *= 0.45 + 1.1 * fext
            seg = L / K
            f = a.spec("freq") * (0.7 + 0.6 * pr["rigidity"]) * (1.2 - 0.4 * pr["fluidity"])
            nd.freq[ids], nd.zeta[ids] = f, a.spec("zeta")
            nd.gravity_w[ids] = a.spec("gravity") * (1 - a.growth * 0.3)
            d0 = R @ a.direction
            perp = np.cross(d0, [0, 0, 1.0])
            perp = perp / max(np.linalg.norm(perp), 1e-6)
            prev = nd.pos[a.origin] + d0 * nd.radius[a.origin] * 0.5
            wave = a.spec("wave") * (0.5 + pr["tendril_activity"])
            vib = 0.03 * inp.high * self.cfg.audio.high_vibration
            for i, idx in enumerate(ids):
                u = (i + 1) / K
                ang = a.spec("curl") * u * 1.5 + wave * math.sin(2.2 * t - 3.0 * u + a.phase) * u
                if fext is not None:                       # a bent finger curls it
                    ang += (1.0 - fext) * 1.4 * u
                d = d0 * math.cos(ang) + perp * math.sin(ang)
                d[2] += vib * math.sin(40 * t + i) - 0.15 * a.spec("gravity") * u
                d = d / max(np.linalg.norm(d), 1e-6)
                nd.target[idx] = prev + d * seg
                prev = nd.pos[idx]                        # follow the *actual* leader: lag = secondary motion

    # ------------------------------------------------------------------ state
    def stretch(self) -> np.ndarray:
        s = np.ones((self.n, 3))
        for a in self.apps.items:
            if a.spec("flat") > 0.5:
                s[a.nodes] = (1.6, 1.6, 0.35)
        return s

    def state(self) -> CreatureState:
        nd = self.nodes
        return CreatureState(self.t, self.beh.state, self.morph.name, nd.pos.copy(), nd.radius.copy(), self.stretch(),
                             nd.kind.copy(), nd.anchor.copy(), nd.center_of_mass(), self.heading, self.params.values(),
                             dict(self.vol), self.surface, self.glow, self.beh.mem.arousal, self.beh.mem.instability,
                             list(self.events[-5:]))


__all__ = ["CreatureEngine", "CreatureState", "EVENTS", "PARAMS"]
