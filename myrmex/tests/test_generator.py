import numpy as np

from myrmex.music import synthetic
from myrmex.performance.generator import GenerateOptions, generate_biped
from myrmex.performance.performance import Performance


def _short(tl, seconds):
    tl.notes = [n for n in tl.notes if n.time < seconds]
    tl.duration = seconds
    return tl


def test_generation_is_deterministic_per_seed(biped_plan):
    tl = _short(synthetic.test_a_four_on_floor(bars=6), 8.0)
    a = generate_biped(tl, biped_plan, GenerateOptions(seed=1, fps=24, tail=0.0))
    b = generate_biped(tl, biped_plan, GenerateOptions(seed=1, fps=24, tail=0.0))
    c = generate_biped(tl, biped_plan, GenerateOptions(seed=2, fps=24, tail=0.0))
    assert a.frames == b.frames == c.frames == 192
    assert np.array_equal(a.deltas, b.deltas)
    assert not np.array_equal(a.deltas, c.deltas)
    assert np.isfinite(a.deltas).all()


def test_performance_save_load(tmp_path, biped_plan):
    tl = _short(synthetic.test_c_ambient(bars=3), 4.0)
    p = generate_biped(tl, biped_plan, GenerateOptions(seed=0, fps=12, tail=0.0))
    path = tmp_path / "perf.npz"
    p.save(str(path))
    q = Performance.load(str(path))
    assert q.frames == p.frames
    assert q.bone_names == p.bone_names
    assert np.allclose(q.deltas, p.deltas)
    assert "behavior" in q.labels and len(q.labels["behavior"]) == q.frames
