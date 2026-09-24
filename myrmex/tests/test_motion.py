import math

import numpy as np

from myrmex.motion.biped import BipedMotor
from myrmex.motion.choreo import CommandScript
from myrmex.motion.skeleton import Pose, SkeletonRest


def test_rest_pose_is_identity(humanoid_rig):
    sk = SkeletonRest.from_rig(humanoid_rig)
    pose = Pose(sk)
    D = pose.solve()
    assert np.allclose(D, np.eye(4)[None], atol=1e-12)


def _run(plan, script, seconds, seed=3, dt=1 / 120):
    motor = BipedMotor(plan, seed=seed)
    slide, reach_violation, nan = 0.0, 0.0, False
    for k in range(int(seconds / dt)):
        pose = motor.update(dt, script.command(k * dt))
        if not np.isfinite(pose.delta).all():
            nan = True
            break
        for f in motor.feet.values():
            if f.planted:
                D = pose.delta[plan.sk.index[f.leg.foot]]
                P0 = f.rest_pivot(f.pivot)
                slide = max(slide, float(np.linalg.norm(D[:3, :3] @ P0 + D[:3, 3] - f.pivot_world)))
            hip = pose.world_head(f.leg.thigh)
            ankle = pose.world_head(f.leg.foot)
            reach_violation = max(reach_violation, float(np.linalg.norm(ankle - hip)) - f.leg.length)
    return motor, slide, reach_violation, nan


def test_walk_turn_stop_without_foot_sliding(biped_plan):
    S = CommandScript()
    S.at(0.0, speed=0.0, energy=0.5)
    S.at(0.8, speed=0.9, blend=0.6)
    S.at(3.5, turn_rate=0.6)
    S.at(5.5, speed=0.0, turn_rate=0.0, blend=0.8)
    S.at(7.0, heading=-math.pi / 2 + 2.0)
    motor, slide, reach, nan = _run(biped_plan, S, 9.0)
    assert not nan
    assert slide < 1e-6, f"planted foot slid {slide} m"
    assert reach < 1e-3, "leg overstretched"
    steps = {s: f.steps for s, f in motor.feet.items()}
    assert steps["l"] >= 4 and steps["r"] >= 4
    assert abs(steps["l"] - steps["r"]) <= 3


def test_standing_still_does_not_step(biped_plan):
    S = CommandScript().at(0.0, speed=0.0, energy=0.4)
    motor, slide, reach, nan = _run(biped_plan, S, 4.0)
    assert not nan and slide < 1e-6
    assert sum(f.steps for f in motor.feet.values()) == 0


def test_same_seed_same_motion(biped_plan):
    S = CommandScript().at(0.0, speed=0.0).at(0.5, speed=0.8, weight_bias=0.3)
    a = BipedMotor(biped_plan, seed=5)
    b = BipedMotor(biped_plan, seed=5)
    for k in range(240):
        pa = a.update(1 / 120, S.command(k / 120)).delta.copy()
        pb = b.update(1 / 120, S.command(k / 120)).delta.copy()
    assert np.array_equal(pa, pb)
