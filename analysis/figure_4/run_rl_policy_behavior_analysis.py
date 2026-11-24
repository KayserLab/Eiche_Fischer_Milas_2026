#!/usr/bin/env python3
"""Figure 4 launcher for the PPO policy-behaviour data analysis.

The implementation stays in ``PhysiLearning`` because it imports the trained
PPO/LV environment and checkpoints from that codebase.  This launcher makes
the Figure 4 dependency explicit: run this file from PyCharm to generate the
canonical Figure 4 policy-behaviour data, then run
``figures/Figure_4/d2_replot_rl_policy_behavior.py`` to render panel 4d.

Any command-line arguments supplied here are passed unchanged to the
implementation (for example ``--mode bootstrap``).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION = (
    REPOSITORY_ROOT
    / "PhysiLearning"
    / "Agent_Behaviour_Analysis"
    / "rl_policy_behavior_analysis.py"
)


def main() -> None:
    if not IMPLEMENTATION.is_file():
        raise FileNotFoundError(f"RL policy-behaviour implementation not found: {IMPLEMENTATION}")
    sys.argv[0] = str(IMPLEMENTATION)
    runpy.run_path(str(IMPLEMENTATION), run_name="__main__")


if __name__ == "__main__":
    main()
