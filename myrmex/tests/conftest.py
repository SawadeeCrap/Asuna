import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

FIXTURES = os.path.join(ROOT, "tests", "fixtures")


@pytest.fixture(scope="session")
def humanoid_rig():
    from myrmex.rig.rigdesc import RigDescription
    return RigDescription.load(os.path.join(FIXTURES, "humanoid_heels_rig.json"))


@pytest.fixture(scope="session")
def biped_plan(humanoid_rig):
    from myrmex.motion.bodyplan import BipedPlan
    return BipedPlan.from_rig(humanoid_rig)
