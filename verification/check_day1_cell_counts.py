"""Check day-1 counting, zero populations, fractions and cohort aggregation."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.Suppl_Figures import Day1_Cell_Counts_By_Experimental_Group as analysis


def main():
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as folder:
        path = Path(folder) / "synthetic_day1.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("sus/day1/block0_values", data=np.zeros((90, 2)))
            h5.create_dataset("res/day1", data=np.zeros((10, 2)))
            h5.create_dataset("res/day10", data=np.zeros((999, 2)))
        counts = analysis.read_day1(path)
        assert counts["total_cells"] == 100
        assert counts["sensitive_cells"] == 90 and counts["resistant_cells"] == 10
        assert counts["resistant_fraction"] == 0.1 and counts["resistant_percent"] == 10
        with h5py.File(path, "w") as h5:
            h5.create_dataset("sus/TP1/values", data=np.zeros((25, 2)))
            h5.create_dataset("res/TP1/values", data=np.zeros((0, 2)))
        counts = analysis.read_day1(path)
        assert counts["total_cells"] == 25 and counts["resistant_percent"] == 0
        with h5py.File(path, "w") as h5:
            h5.create_dataset("sus/day2", data=np.zeros((90, 2)))
            h5.create_dataset("res/day2", data=np.zeros((10, 2)))
        try:
            analysis.read_day1(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Day 2 was substituted for missing day 1")
    rows = []
    for j, group in enumerate(analysis.GROUPS):
        for i in range(3):
            sensitive, resistant = 90 + 10 * j + i, 10 + j
            total = sensitive + resistant
            rows.append({"experimental_strategy": group,
                         "exp_total_cells": total,
                         "exp_sensitive_cells": sensitive, "exp_resistant_cells": resistant,
                         "exp_resistant_percent": 100 * resistant / total})
    counts = pd.DataFrame(rows)
    summary = analysis.group_summary(counts)
    assert len(summary) == 16
    ct_total = summary.loc[summary["metric"].eq("total_cells")
                           & summary["experimental_strategy"].eq("CT")].iloc[0]
    assert ct_total["median"] == 101 and ct_total["mean"] == 101 and ct_total["n_wells"] == 3
    global_tests, pairwise = analysis.comparisons(counts, n_permutations=99)
    assert len(global_tests) == 4 and len(pairwise) == 24
    assert (global_tests["p_holm_four_metrics"] >= global_tests["p_permutation"]).all()
    assert (pairwise["p_holm_all_24_comparisons"] >= pairwise["p_permutation_two_sided"]).all()
    print("Passed: direct/pandas HDF5 counts, exact acquisition day 1, empty population, "
          "per-well resistant fraction, group medians/means, comparison counts and multiplicity.")


if __name__ == "__main__":
    main()
