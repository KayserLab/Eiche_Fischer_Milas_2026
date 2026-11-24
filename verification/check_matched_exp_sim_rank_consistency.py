"""Synthetic checks for exact well matches and within-strategy rank agreement."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.Suppl_Figures import TTF_TTE_Matched_Exp_Sim_Rank_Consistency as analysis


def main():
    rows = []
    for strategy in ("CT", "AT50"):
        for event in analysis.EVENTS:
            for i in range(6):
                sample = f"A{i + 1}_{strategy}-A{i + 1}"
                rows.append({"event": event, "strategy": strategy, "initial_condition": sample,
                             "exp_sample": sample, "exp_plate": "Plate_test", "sim_sample": f"R{i + 10}_{strategy}",
                             "match_status": "exact_match", "exp_time": float(i + 1),
                             "sim_time": float(i + 1 if strategy == "CT" else 6 - i),
                             "exp_observed": i != 5, "sim_observed": True,
                             "exp_reason": "", "sim_reason": ""})
    pairs = analysis.prepare_pairs(pd.DataFrame(rows))
    assert pairs["both_observed"].sum() == 20
    ranks = analysis.rank_table(pairs)
    assert ranks.loc[~ranks["rank_eligible"], "rank_percentile"].isna().all()
    for _, group in ranks.loc[ranks["rank_eligible"]].groupby(["event", "strategy", "source"]):
        assert np.isclose(group["rank_percentile"].min(), 0)
        assert np.isclose(group["rank_percentile"].max(), 100)
    stats = analysis.strategy_correlations(pairs, n_bootstrap=199, n_permutations=499)
    ct = stats.loc[stats["strategy"].eq("CT")]
    at = stats.loc[stats["strategy"].eq("AT50")]
    assert ct["n_pairs"].eq(5).all() and np.allclose(ct["rho"], 1)
    assert at["n_pairs"].eq(5).all() and np.allclose(at["rho"], -1)
    assert np.allclose(ct["pearson_r"], 1) and np.allclose(at["pearson_r"], -1)
    x = np.arange(1.0, 6.0)
    assert np.isclose(analysis.pearson_correlation(x, x ** 2), np.corrcoef(x, x ** 2)[0, 1])
    assert not np.isclose(analysis.pearson_correlation(x, x ** 2), 1)
    assert np.isnan(analysis.pearson_correlation([1, 2], [2, 4]))
    assert np.isnan(analysis.pearson_correlation([1, 1, 1], [1, 2, 3]))
    pooled = stats.loc[stats["strategy"].eq("All points")]
    assert pooled["n_pairs"].eq(10).all()
    assert np.allclose(pooled["pearson_r"], 0, atol=1e-12)
    assert np.allclose(pooled["pearson_p_two_sided"], 1)
    assert pooled["q_bh_both_events"].isna().all()
    from scipy.stats import pearsonr
    expected = pearsonr(x, x ** 2)
    result = analysis.pearson_statistics(x, x ** 2)
    assert np.isclose(result["pearson_r"], expected.statistic)
    assert np.isclose(result["pearson_p_two_sided"], expected.pvalue)
    assert stats.loc[stats["strategy"].eq("LV-Agent"), "rho"].isna().all()
    invalid = pd.DataFrame(rows)
    invalid.loc[0, "initial_condition"] = "different_seed"
    try:
        analysis.prepare_pairs(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched seed accepted")
    try:
        analysis.prepare_pairs(pd.concat([pd.DataFrame(rows), pd.DataFrame(rows[:1])]))
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated experimental well accepted")
    print("Passed: exact IDs, duplicate rejection, separate strategy/source ranks, "
          "censored exclusion, positive/negative within-strategy correlation, missing strategy.")


if __name__ == "__main__":
    main()
