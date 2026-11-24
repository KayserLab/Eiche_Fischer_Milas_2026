"""Check that all observed simulation events survive incomplete matching."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import pandas as pd
import numpy as np
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.Suppl_Figures import plot_TTF_TTE_Matched_Simulation_Strategies as plot


def main():
    rows = []
    for event in plot.EVENTS:
        for seed_index, seed in enumerate(("complete_seed", "censored_seed", "missing_seed")):
            for index, strategy in enumerate(plot.STRATEGIES):
                if seed == "missing_seed" and strategy == "LV-Agent":
                    continue
                observed = not (seed == "censored_seed" and event == "TTF"
                                and strategy == "AT100+")
                rows.append({"event": event, "initial_condition": seed,
                             "strategy": strategy, "time": float(20 + index + seed_index),
                             "observed": observed,
                             "event_status": "observed" if observed else "event_not_observed"})
        rows.append({"event": event, "initial_condition": "single_seed",
                     "strategy": "CT", "time": 25.0, "observed": True,
                     "event_status": "observed"})
    values = pd.DataFrame(rows)
    plotted, audit = plot.select_observed_cohorts(values)
    for event in plot.EVENTS:
        assert set(plotted.loc[plotted["event"].eq(event), "initial_condition"]) == {
            "complete_seed", "censored_seed", "missing_seed", "single_seed"}
    assert plotted["observed"].all()
    assert len(plotted) == int(values["observed"].sum())
    assert audit.loc[audit["initial_condition"].eq("missing_seed"), "included"].all()
    assert not audit.loc[audit["initial_condition"].eq("missing_seed"),
                         "complete_all_strategies"].any()
    incomplete = plotted.loc[plotted["event"].eq("TTF")
                             & plotted["initial_condition"].eq("censored_seed")]
    assert len(incomplete) == 4 and not incomplete["strategy"].eq("AT100+").any()
    assert audit.loc[audit["initial_condition"].eq("missing_seed"),
                     "missing_strategies"].eq("LV-Agent").all()
    try:
        plot.select_observed_cohorts(pd.concat([values, values.iloc[[0]]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate seed/strategy/event accepted")
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as folder:
        output = Path(folder) / "synthetic_cohort_check.pdf"
        validated = plot.validate_values(values)
        table, _ = plot.correlations(validated, n_bootstrap=199, n_permutations=499)
        savefig = plot.ft.savefig
        def check_layout(fig, path, **kwargs):
            # Matched times occupy the top row; correlations occupy the bottom.
            panels = fig.axes[:4]
            assert [ax.get_title(loc="left").split()[0].rstrip(":") for ax in panels] == [
                "TTE", "TTF", "TTE", "TTF"]
            assert [ax.get_ylabel() for ax in (panels[0], panels[1])] == [
                "TTE in silico [days]", "TTF in silico [days]"]
            for ax in (panels[2], panels[3]):
                assert [tick.get_text() for tick in ax.get_yticklabels()] == [
                    plot.LABELS[strategy] for strategy in plot.STRATEGIES]
                assert ax.images[0].get_array().shape == (5, 5)
            for ax, event in ((panels[0], "TTE"), (panels[1], "TTF")):
                assert sum(len(points.get_offsets()) for points in ax.collections) == int(
                    plotted["event"].eq(event).sum())
            for ax, event in ((panels[2], "TTE"), (panels[3], "TTF")):
                matrix = ax.images[0].get_array().filled(np.nan)
                annotations = {text.get_position(): text.get_text() for text in ax.texts}
                for i, a in enumerate(plot.STRATEGIES):
                    for j in range(i + 1, len(plot.STRATEGIES)):
                        b = plot.STRATEGIES[j]
                        pair = table.loc[table["event"].eq(event)
                                         & table["strategy_a"].eq(a)
                                         & table["strategy_b"].eq(b)].iloc[0]
                        assert np.isclose(matrix[i, j], pair["rho"], equal_nan=True)
                        assert np.isclose(matrix[j, i], pair["rho"], equal_nan=True)
                        assert f"n={pair['n_pairs']}" in annotations[i, j]
            colorbar_box = fig.axes[4].get_position()
            matrix_box = panels[3].get_position()
            assert np.isclose(colorbar_box.y0, matrix_box.y0)
            assert np.isclose(colorbar_box.height, matrix_box.height)
            savefig(fig, path, **kwargs)
        with patch.object(plot.ft, "savefig", side_effect=check_layout):
            plot.plot_matched_and_correlations(plotted, validated, table, output)
        assert output.stat().st_size > 1000
        assert {path.suffix for path in Path(folder).iterdir()} == {".pdf"}
    print("Passed: all observed events retained, missing/censored strategies allowed, "
          "single outcomes retained, duplicate rejection, 2x2 strategy-correlation alignment, PDF-only export.")


if __name__ == "__main__":
    main()
