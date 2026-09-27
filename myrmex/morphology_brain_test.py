#!/usr/bin/env python3
"""The morphology brain without Blender - a sandbox, the benchmarks, the measurements.

    python morphology_brain_test.py              1 minute of the Spear with the decision log
    python morphology_brain_test.py --help       everything else (benchmark A-G, sweeps, --ipc-bench, --kev-bench)

See docs/MORPHOLOGY_BRAIN.md.  (The code: src/myrmex/brain/bench.py.)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from myrmex.brain.bench import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
