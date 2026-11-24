"""Synthetic checks for exact matching, ranks, ties, censoring and uncertainty."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.Suppl_Figures import TTF_TTE_Matched_Simulation_Strategies as analysis


def main():
    x = np.array([1, 1, 2, 4, 5, 8.0])
    y = np.array([3, 2, 2, 6, 1, 7.0])
    assert np.isclose(analysis.spearman_statistic(x, y), spearmanr(x, y).statistic)
    assert np.allclose(analysis.spearman_statistic(
        np.stack([x, x]), np.stack([x, -x]), axis=1), [1, -1])
    assert np.isnan(analysis.spearman_statistic(np.ones(4), np.arange(4)))
    rows = []
    for index in range(6):
        for event in analysis.EVENTS:
            for strategy, time in (("CT", index + 1.0), ("AT50", 6.0 - index)):
                rows.append({"event": event, "initial_condition": f"seed_{index}",
                             "strategy": strategy, "time": time, "observed": True,
                             "event_status": "observed"})
    # A censored endpoint must not be treated as a known late event or be joined.
    rows.append({"event": "TTF", "initial_condition": "censored_seed", "strategy": "CT",
                 "time": 99.0, "observed": False, "event_status": "event_not_observed"})
    rows.append({"event": "TTF", "initial_condition": "censored_seed", "strategy": "AT50",
                 "time": 7.0, "observed": True, "event_status": "observed"})
    # Same run order but different seed IDs cannot form a matched pair.
    rows.append({"event": "TTF", "initial_condition": "CT_only_seed", "strategy": "CT",
                 "time": 8.0, "observed": True, "event_status": "observed"})
    values = analysis.validate_values(pd.DataFrame(rows))
    ranks = analysis.build_rank_table(values)
    censored = ranks.loc[ranks["initial_condition"].eq("censored_seed")
                         & ranks["strategy"].eq("CT")]
    assert censored["rank_percentile"].isna().all()
    observed_ranks = ranks.loc[ranks["rank_eligible"], "rank_percentile"]
    assert observed_ranks.between(0, 100).all()
    table, joined = analysis.correlations(values, n_bootstrap=199, n_permutations=499)
    pair = table.loc[table["event"].eq("TTF") & table["strategy_a"].eq("CT")
                     & table["strategy_b"].eq("AT50")].iloc[0]
    assert pair["n_pairs"] == 6 and np.isclose(pair["rho"], -1)
    assert "censored_seed" not in set(joined["initial_condition"])
    assert "CT_only_seed" not in set(joined["initial_condition"])
    assert pair["ci_low"] <= pair["rho"] <= pair["ci_high"]
    assert 0 < pair["p_permutation_two_sided"] <= 1
    assert np.allclose(analysis.benjamini_hochberg(np.array([0.01, 0.04, 0.03, np.nan]))[:3],
                       [0.03, 0.04, 0.04])
    assert np.isnan(analysis.benjamini_hochberg(np.array([np.nan]))[0])
    try:
        analysis.validate_values(pd.concat([values, values.iloc[[0]]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate seed/outcome accepted")
    print("Passed: tied/vectorized Spearman, exact pair joins, censored exclusion, "
          "rank percentiles, paired bootstrap, permutation bounds, BH adjustment, duplicates.")


if __name__ == "__main__":
    main()
