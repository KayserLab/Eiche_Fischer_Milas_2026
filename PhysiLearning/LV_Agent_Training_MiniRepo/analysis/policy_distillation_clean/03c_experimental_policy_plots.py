"""Run with PyCharm's green Run button to create both ATAgent2 plot analyses.

First set EXPERIMENTAL_DATA_ROOT in 02d_load_experimental_decisions.py and run
that script. This script reads its experimental_decisions.csv and saves the
overall portrait, exact-history panels, grouped-history plot, and mismatch
portrait in figures/.
"""
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from common import EXPERIMENTAL_DATA_FILE


def main():
    if not EXPERIMENTAL_DATA_FILE.is_file():
        raise FileNotFoundError(
            "No ATAgent2 table yet. Set EXPERIMENTAL_DATA_ROOT in "
            "02d_load_experimental_decisions.py, then run that script first.")
    folder = Path(__file__).resolve().parent
    previous_arguments = sys.argv[:]
    try:
        for script in ("03_policy_portrait.py", "03b_history_groups.py", "03d_experimental_mismatches.py"):
            sys.argv = [str(folder / script), "experimental"]
            runpy.run_path(str(folder / script), run_name="__main__")
    finally:
        sys.argv = previous_arguments
    plt.show()


if __name__ == "__main__":
    main()
