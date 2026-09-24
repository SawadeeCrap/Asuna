import math

import numpy as np

from myrmex.util import mathutil as mu
from myrmex.util.noise import LatentField, Noise1D
from myrmex.util.rng import Rng, RngStreams
from myrmex.util.springs import SecondOrder, Spring


def test_quaternion_roundtrip_batch():
    rng = np.random.default_rng(0)
    for _ in range(50):
        axis = rng.normal(size=3)
        R = mu.axis_angle_matrix(axis, rng.uniform(-3, 3))
        q = mu.quat_from_matrix(R)
        assert np.allclose(mu.matrix_from_quat(q), R, atol=1e-9)
    Rs = np.stack([mu.axis_angle_matrix(rng.normal(size=3), rng.uniform(-3.1, 3.1)) for _ in range(200)])
    qs = mu.quats_from_matrices(Rs)
    assert np.allclose(mu.matrices_from_quats(qs), Rs, atol=1e-8)


def test_quat_continuity_no_flips():
    angles = np.linspace(0, 6 * math.pi, 300)
    qs = np.stack([mu.quat_from_matrix(mu.rot_z(a)) for a in angles])
    qc = mu.quat_continuity(qs)
    dots = np.einsum("ij,ij->i", qc[1:], qc[:-1])
    assert (dots > 0).all()


def test_support_margin_inside_outside():
    hull = mu.convex_hull_2d([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert len(hull) == 4
    assert mu.support_margin((0.5, 0.5), hull) > 0.49
    assert mu.support_margin((1.5, 0.5), hull) < 0


def test_rng_determinism_and_independence():
    a = RngStreams(42)
    b = RngStreams(42)
    xs = [a.stream("gait").random() for _ in range(5)]
    ys = [b.stream("gait").random() for _ in range(5)]
    assert xs == ys
    # Consuming another stream must not change this one.
    c = RngStreams(42)
    for _ in range(100):
        c.stream("other").random()
    assert [c.stream("gait").random() for _ in range(5)] == xs
    assert RngStreams(43).stream("gait").random() != xs[0]
    r = Rng(1)
    vals = [r.normal() for _ in range(4000)]
    assert abs(np.mean(vals)) < 0.06 and abs(np.std(vals) - 1.0) < 0.06


def test_noise_smooth_and_bounded():
    n = Noise1D(3, frequency=1.0, octaves=3)
    ts = np.linspace(0, 20, 4001)
    v = n.sample_many(ts)
    assert np.abs(v).max() < 1.6
    assert np.abs(np.diff(v)).max() < 0.1          # smooth
    assert abs(n.sample(3.3) - n.sample_many(np.array([3.3]))[0]) < 1e-12
    lf = LatentField(7, [0.2, 0.5, 1.1], ["a", "b", "c"])
    s = lf.sample(1.0)
    assert set(s) == {"a", "b", "c"}


def test_second_order_converges_and_is_stable():
    s = SecondOrder(2.0, 0.5, 0.0, 0.0)
    for _ in range(600):
        y = s.update(1 / 120, 1.0)
    assert abs(y - 1.0) < 1e-3
    # Large time step must not explode (stability clamp).
    s2 = SecondOrder(20.0, 0.3, 2.0, 0.0)
    for _ in range(100):
        y2 = s2.update(0.2, 1.0)
    assert abs(y2) < 10
    sp = Spring(0.2, 0.0)
    for _ in range(300):
        sp.update(1 / 120, 1.0)
    assert abs(sp.x - 1.0) < 1e-3


def test_anticipation_parameter_winds_up():
    s = SecondOrder(2.0, 0.8, -1.0, 0.0)
    ys = [s.update(1 / 120, 1.0) for _ in range(20)]
    assert min(ys) < 0.0                              # r < 0 first moves against the change
