"""Verify per-seed mean ranks precede cohort medians, including missingness."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.Suppl_Figures import TTF_TTE_Experimental_Cohort_Mean_Ranks as analysis


def main():
    specifications = [
        ("A1", "CT", 10, True), ("A1", "AT50", 30, True), ("A1", "AT100", 40, True),
        ("A2", "CT", 20, True), ("A2", "AT50", 100, False),
        ("B1", "CT", 30, True), ("B1", "AT50", 10, True), ("B1", "AT100", 10, True),
        ("D1", "CT", 50, False), ("D1", "AT50", 50, False),
    ]
    rows = []
    for event in analysis.EVENTS:
        for seed, strategy, time, observed in specifications:
            rows.append({"event": event, "initial_condition": seed, "strategy": strategy,
                         "time": float(time), "observed": observed,
                         "event_status": "observed" if observed else "event_not_observed"})
    pairs = pd.DataFrame([
        {"initial_condition": seed, "strategy": group, "exp_sample": seed,
         "exp_plate": "P1", "match_status": "exact_match"}
        for seed, group in (("A1", "CT"), ("A2", "CT"), ("B1", "AT50"), ("D1", "AT100+"))
    ])
    _, ranks, scores = analysis.compute_scores(pd.DataFrame(rows), pairs)
    summary = analysis.summarize_scores(scores)
    for event in analysis.EVENTS:
        panel = scores.loc[scores["event"].eq(event)].set_index("initial_condition")
        assert np.isclose(panel.loc["A1", "mean_rank_percentile"], 200 / 3)
        assert panel.loc["A1", "n_observed_strategies"] == 3
        assert panel.loc["A2", "mean_rank_percentile"] == 50
        assert panel.loc["A2", "n_observed_strategies"] == 1
        assert pd.isna(panel.loc["D1", "mean_rank_percentile"])
        assert panel.loc["D1", "score_status"] == "no_observed_events"
        ct = summary.loc[summary["event"].eq(event)
                         & summary["experimental_strategy"].eq("CT")].iloc[0]
        assert np.isclose(ct["median_mean_rank_percentile"], (200 / 3 + 50) / 2)
        assert not np.isclose(ct["median_mean_rank_percentile"], 75)
    assert ranks.loc[~ranks["rank_eligible"], "rank_percentile"].isna().all()
    assert np.allclose(analysis.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert analysis.median_range(np.array([0, 1]), np.array([4, 5])) == 4
    synthetic = pd.DataFrame([
        {"event": event, "initial_condition": f"{group}_{i}",
         "experimental_strategy": group, "mean_rank_percentile": float(j * 10 + i),
         "score_status": "included"}
        for event in analysis.EVENTS for j, group in enumerate(analysis.GROUPS) for i in range(3)
    ])
    omnibus, pairwise = analysis.compare_medians(synthetic, n_permutations=99)
    assert len(omnibus) == 2 and len(pairwise) == 12
    assert pairwise["test_mode"].eq("exact").all()
    assert pairwise["n_resamples_used"].eq(20).all()
    assert pairwise["p_permutation_two_sided"].between(0, 1).all()
    print("Passed: per-seed mean before cohort median, equal strategy weighting, "
          "normalized ranks, missing/censored exclusions, zero-event audit, exact tests and Holm adjustment.")


if __name__ == "__main__":
    main()
