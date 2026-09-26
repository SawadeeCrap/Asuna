"""The Bionic line (v14-v18): each organism obeys its own law of form - checked on the physics itself."""
import math

import numpy as np
import pytest

from myrmex.creature.bionic import BIONIC_EVENTS, REGIMES, VARIANTS, meshes
from myrmex.creature.bionic.arbor import GAMMA, ArborEngine
from myrmex.creature.bionic.ferro import BC, FerroEngine
from myrmex.creature.bionic.fold import FoldEngine, miura
from myrmex.creature.bionic.tensor import TensorEngine
from myrmex.creature.bionic.truss import TrussEngine
from myrmex.creature.control import CreatureControlInput
from myrmex.creature.protocol import decode_creature, encode_creature


def quiet(e, regime=None):
    for k in ("instability", "mutation", "obstacle_rate"):
        e.set_parameter(k, 0.0)
    if regime:
        e.goal(regime, 3.2)
    e.dwell = 1e9
    return e


def run(e, seconds, on=True, dt=1 / 60, kick_every=0):
    st = None
    for k in range(int(seconds / dt)):
        hit = 0.9 if kick_every and k % kick_every == 0 else 0.0
        e.set_input(CreatureControlInput(energy=0.6 * on, bass=0.45 * on, high=0.2 * on, beat=k * dt * 2, playing=on,
                                         tempo=120, transient=hit))
        st = e.update(dt)
    return st


def test_tensor_struts_stay_rigid_cables_only_pull_and_forms_differ():
    chords = {}
    for reg in ("COIL", "REACH"):
        e = quiet(TensorEngine(), reg)
        st = run(e, 5.0)
        x = e.x
        L = np.linalg.norm(x[e.struts[:, 1]] - x[e.struts[:, 0]], axis=1)
        assert np.abs(L / e.L_strut - 1).max() < 0.01                       # struts are rigid
        assert (e.tension >= 0).all()                                        # cables never push
        head, tail = x[-3:].mean(0), x[:3].mean(0)
        chords[reg] = float(np.linalg.norm(head - tail))
        assert np.isfinite(st.pos).all() and st.morphology == reg
    assert chords["REACH"] > 1.8 * chords["COIL"]                            # a needle vs a curl


def test_fold_is_exact_rigid_origami_and_its_regimes_differ():
    a, b, g = 0.17, 0.15, math.radians(55)
    for th in (0.1, 0.7, 1.3):                                               # every panel keeps its shape
        P = miura(np.full((5, 6), th), a, b, g)
        assert np.allclose(np.linalg.norm(P[:, 1:] - P[:, :-1], axis=2), b)
        assert np.allclose(np.linalg.norm(P[1:] - P[:-1], axis=2), a)
    boxes = {}
    for reg in ("GLIDER", "PLEAT", "BELL"):
        e = quiet(FoldEngine(), reg)
        run(e, 4.0)
        L = np.linalg.norm(e.x[e.edges[:, 1]] - e.x[e.edges[:, 0]], axis=1)
        assert np.abs(L / e.L0 - 1).max() < 0.03                              # the panels stay (nearly) rigid
        X = e.x - e.x.mean(0)
        boxes[reg] = np.sort(np.ptp(X @ np.linalg.eigh(np.cov(X.T))[1], axis=0))[::-1]
    assert boxes["GLIDER"][0] * boxes["GLIDER"][1] > 1.8 * boxes["PLEAT"][0] * boxes["PLEAT"][1]   # wide vs folded
    assert boxes["BELL"][2] > 3 * boxes["GLIDER"][2]                          # the bell has depth


def test_arbor_grows_by_colonisation_obeys_the_pipe_model_and_is_pruned_in_silence():
    e = quiet(ArborEngine(), "SPHERE")
    st = run(e, 4.0)
    al = e.alive & ~e.shed
    assert al.sum() > 150 and np.isfinite(st.pos).all()
    assert abs(e.V_used / e.V_max - 1) < 0.05                                 # a fixed amount of material
    kids = np.flatnonzero(al & (e.parent >= 0))
    p = e.parent[kids]
    s = np.zeros(e.N)
    np.add.at(s, p, e.r_goal[kids] ** GAMMA)
    inner = np.unique(p[p > 0])
    f = (e.r_goal[inner] ** GAMMA / np.maximum(s[inner], 1e-30))             # one common scale (the budget)
    ok = np.isclose(f, np.median(f), rtol=0.05)
    assert ok.mean() > 0.9                                                    # r_parent^2.5 = sum r_child^2.5
    busy = int(al.sum())
    run(e, 10.0, on=False)
    assert int((e.alive & ~e.shed).sum()) < 0.75 * busy                       # silence prunes it back
    before = int((e.alive & ~e.shed).sum())
    assert e.trigger_event("SHED")
    e.update(1 / 60)
    dropped = int((e.alive & e.shed).sum())
    assert 0 < dropped <= 0.3 * before + 1                                    # a limb, never the body
    run(e, 3.0, on=False)
    assert not (e.alive & e.shed).any()                                       # it dissolved


def test_ferro_rosensweig_spikes_hysteresis_labyrinth_and_volume():
    e = quiet(FerroEngine(), "CROWN")
    run(e, 3.0, kick_every=30)
    assert e.B / BC > 1.3 and e.sp_on.sum() > 10 and e.sp_h.max() > 0.05     # supercritical: a field of spikes
    n = e.sp_n[e.sp_on]
    C = n @ n.T
    np.fill_diagonal(C, -1)
    nn = np.arccos(np.clip(C.max(1), -1, 1))
    assert nn.std() / nn.mean() < 0.3                                         # evenly packed (hexagonal-ish)
    assert e.trigger_event("SPLIT")
    e.update(1 / 60)
    assert (e.sat_V > 0).sum() >= 2
    assert abs((e.V_main + e.sat_V.sum()) / e.V_total - 1) < 1e-9             # volume is exact
    assert e.trigger_event("CALM")
    run(e, 3.0, on=False)
    assert e.B < 0.85 * BC * 1.2 and e.sp_h.max() < 0.02                      # below critical: a smooth mirror
    assert abs(e.V_main / e.V_total - 1) < 1e-9                               # the droplets came home
    lab = quiet(FerroEngine(), "LABYRINTH")
    run(lab, 5.0)
    U = np.abs(np.fft.fft2(lab.lab.u)) ** 2
    U[0, 0] = 0
    k = np.fft.fftfreq(lab.lab.n, 1 / lab.lab.n)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    assert (U * KY ** 2).sum() > 5 * (U * KX ** 2).sum()                      # stripes along the in-plane field


def test_truss_holds_its_machine_and_moves_bone_to_the_load():
    e = quiet(TrussEngine(), "FUSELAGE")
    st = run(e, 8.0, kick_every=30)
    al = e.alive & ~e.dying
    R = e._R() @ e.glove.G
    x = e.x - e.x.mean(0)
    T = (e.T_body - e.T_body.mean(0)) @ R.T
    assert np.sqrt(((x - T) ** 2).sum(1).mean()) < 0.2                        # it keeps its shape
    V = float((math.pi * e.r[al] ** 2 * e.L0[al]).sum())
    assert abs(V / e.V_budget - 1) < 0.05                                     # a fixed amount of bone
    r = np.sort(e.r[al])
    gini = (2 * np.arange(1, len(r) + 1) - len(r) - 1).dot(r) / (len(r) * r.sum())
    assert gini > 0.2                                                         # bone went to the load paths
    thr = e.thrust > 0.3
    near = al & (thr[e.ei[:, 0]] | thr[e.ei[:, 1]])
    assert e.r[near].mean() > 1.5 * e.r[al & ~near].mean()                    # thrusters' struts are the thickest
    assert np.isfinite(st.pos).all() and (st.members[:, 0] >= 0).sum() == int(e.alive.sum())


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_bionic_protocol_meshes_and_events(variant):
    E, Cfg = VARIANTS[variant]
    e = E(Cfg(seed=11))
    st = run(e, 1.0)
    fr = decode_creature(encode_creature(st, 7, 2.0, 120.0))
    assert fr.bkind == e.KIND and fr.style == 8 + e.KIND and fr.morphology == st.morphology
    assert np.allclose(fr.members[:, 3:], st.members[:, 3:], atol=2e-3) if len(st.members) else True
    assert np.allclose(fr.extra, st.extra, atol=1e-4)
    parts = meshes.parts(fr.bkind, fr.pos, fr.radius, fr.members, fr.extra,
                         **({"seeds": np.ones((10, 3)) * 1.5, "t": 1.0} if e.KIND == 3 else {}))
    for p in parts.values():
        assert np.isfinite(p.verts).all() and p.faces.max() < len(p.verts)
        assert all(len(a) == len(p.verts) for a in p.attrs.values())
    own = [ev for ev in E.EVENTS if ev in BIONIC_EVENTS]
    assert len(own) == 3
    for ev in own:
        assert e.trigger_event(ev)
    st = run(e, 1.0)
    assert np.isfinite(st.pos).all() and st.morphology in E.REGIMES


def test_bionic_session_take_and_app(tmp_path):
    from myrmex.app import controllers as C
    from myrmex.creature.backend import CONTROL_NOTES
    from myrmex.creature.take import CreatureTake
    from myrmex.realtime.glove import SCULPT_SHAPES
    from myrmex.realtime.session import LiveConfig, LiveSession

    class Sink:
        def send_raw(self, b):
            pass

        def close(self):
            pass
    ses = LiveSession(LiveConfig(backend="ferro", clock="internal", record=str(tmp_path), out=[]), start_inputs=False,
                      sink=Sink(), now=0.0)
    now = 0.0
    for _ in range(240):
        now += 1 / 120
        ses.step(now)
    assert ses.creature.trigger("SURGE") and ses.creature.trigger("SPLIT")
    take = CreatureTake(ses.save_take())
    assert take.variant == "ferro" and C.take_variant(take.path) == "ferro"
    assert take.resampled("members", 30).shape[2] == 6 and take.resampled("extra", 30).shape[1] == meshes.FERRO_LEN
    assert set(VARIANTS) <= set(C.CREATURE_BACKENDS)
    for v, (E, _) in VARIANTS.items():
        assert set(SCULPT_SHAPES[v]) <= set(E.REGIMES)
    for ev in BIONIC_EVENTS:
        assert ev in CONTROL_NOTES.values()
    assert len(REGIMES) == len(set(REGIMES)) >= 28
