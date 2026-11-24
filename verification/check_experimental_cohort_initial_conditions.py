"""Check cohort assignment, censoring-aware medians, RMST and logrank math."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import logrank, CensoredData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.Suppl_Figures import TTF_TTE_Experimental_Cohort_Initial_Conditions as analysis


def main():
    median, rmst = analysis.km_summary([1, 2, 3], [True, False, True], 3)
    assert median == 3 and np.isclose(rmst, 7 / 3)
    median, rmst = analysis.km_summary([2, 3], [False, False], 3)
    assert np.isnan(median) and rmst == 3
    _, rmst = analysis.km_summary([2, 3], [False, False], 4)
    assert np.isnan(rmst)
    # Independent SciPy implementation for ties and right censoring.
    x = CensoredData(uncensored=[1, 3, 4], right=[2])
    y = CensoredData(uncensored=[2, 3, 6], right=[5])
    expected = logrank(x, y)
    actual = analysis.multigroup_logrank(
        [1, 2, 3, 4, 2, 3, 5, 6],
        [True, False, True, True, True, True, False, True],
        ["A"] * 4 + ["B"] * 4)
    assert actual["df"] == 1
    assert np.isclose(actual["logrank_chi2"], expected.statistic ** 2)
    assert np.isclose(actual["p_logrank"], expected.pvalue)
    # Membership follows the experimental assignment, not the simulated treatment.
    pairs = pd.DataFrame([
        {"initial_condition": "seed_CT", "strategy": "CT", "exp_plate": "P1",
         "exp_sample": "seed_CT", "match_status": "exact_match"},
        {"initial_condition": "seed_AT50", "strategy": "AT50", "exp_plate": "P2",
         "exp_sample": "seed_AT50", "match_status": "exact_match"},
    ])
    simulations = pd.DataFrame([
        {"event": "TTF", "initial_condition": seed, "strategy": "CT",
         "event_status": "observed", "time": 10.0, "observed": True}
        for seed in ("seed_CT", "seed_AT50", "not_in_retained_experiment")
    ])
    cohorts, joined = analysis.join_experimental_cohorts(pairs, simulations)
    assert len(cohorts) == 2 and len(joined) == 2
    assert joined.loc[joined["initial_condition"].eq("seed_AT50"),
                      "experimental_strategy"].iloc[0] == "AT50"
    assert joined["strategy"].eq("CT").all()
    try:
        analysis.join_experimental_cohorts(pairs, pd.concat([simulations, simulations.iloc[[0]]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated simulation outcome accepted")
    print("Passed: experimental-cohort assignment, exact matching, censoring-aware median/RMST, "
          "logrank agreement with SciPy, no tail extrapolation, duplicate rejection.")


if __name__ == "__main__":
    main()
