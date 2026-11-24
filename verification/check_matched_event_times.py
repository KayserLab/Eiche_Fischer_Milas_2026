"""Focused checks for exact seed/strategy matching and event censoring.

Uses only temporary synthetic inputs inside the repository, never study data.
"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.Suppl_Figures import TTF_TTE_Matched_InVitro_DPD as plot


def write_counts(path, resistant):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for day, nres in enumerate(resistant, 1):
            h5.create_dataset(f"sus/day{day}", data=np.zeros((100 - nres, 2)))
            h5.create_dataset(f"res/day{day}", data=np.zeros((nres, 2)))


def main():
    (ROOT / "tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT / "tmp") as folder:
        base = Path(folder)
        exp = base / "exp"
        sim = base / "sim"
        seed = "B2_AT50-B2"
        exp_path = exp / "Plate_2/hdf5" / f"{seed}{plot.EXPERIMENT_H5_SUFFIX}"
        sim_path = sim / "Plate_AT50/hdf5" / f"R11_AT50{plot.SIMULATION_H5_SUFFIX}"
        write_counts(exp_path, [10, 30, 50])
        write_counts(sim_path, [10, 50, 70])
        # Same well coordinate is insufficient if the seed ID differs.
        write_counts(sim / "Plate_AT50/hdf5" / f"R12_AT50{plot.SIMULATION_H5_SUFFIX}",
                     [10, 40, 60])
        # An identical seed from a different treatment cannot be paired.
        write_counts(sim / "Plate_CT/hdf5" / f"R11_CT{plot.SIMULATION_H5_SUFFIX}",
                     [10, 40, 60])
        metadata = pd.DataFrame([
            {"protocol": "AT50", "run": "11", "initial_condition": seed},
            {"protocol": "AT50", "run": "12", "initial_condition": "B2_CT-B2"},
            {"protocol": "CT", "run": "11", "initial_condition": seed},
        ])
        matches = plot.discover_pairs(metadata, exp, sim)
        assert matches["match_status"].eq("exact_match").sum() == 1
        assert matches["match_status"].eq(
            "no_exact_experimental_seed_with_same_strategy").sum() == 2
        margins = pd.DataFrame([
            {"dataset_type": kind, "plate": plate, "sample": sample,
             "strategy": "AT50", "day": day, "signed_escape_margin": margin}
            for kind, plate, sample, values in (
                ("experimental", "Plate_2", seed, [-1, 0, 1]),
                ("simulation", "Plate_AT50", "R11_AT50", [-1, -1, 0]),
            ) for day, margin in enumerate(values, 1)
        ])
        pairs = plot.event_pairs(matches, margins, 300)
        ttf = pairs.loc[pairs["event"].eq("TTF")].iloc[0]
        tte = pairs.loc[pairs["event"].eq("TTE")].iloc[0]
        assert ttf["exp_time"] == 3 and ttf["sim_time"] == 2
        assert ttf["exp_observed"] and ttf["sim_observed"]
        assert tte["exp_time"] == 3 and tte["sim_time"] == 3
        assert tte["exp_observed"] and not tte["sim_observed"]
        assert ttf["plot_status"] == "included"
        assert tte["plot_status"] == "event_unobserved_in_one_or_both_sources"
        summary = plot.agreement_summary(pairs)
        censored = summary.loc[summary["event"].eq("TTE") & summary["strategy"].eq("AT50")].iloc[0]
        assert censored["n_pairs_with_censoring"] == 1
        assert censored["n_both_events_observed"] == 0
        assert censored["n_plotted"] == 0
        assert pd.isna(censored["mae_days"])
        # Exercise the complete two-panel export without touching study data.
        plot.plot_pairs(pairs, base / "synthetic_render_check")
        assert (base / "synthetic_render_check.pdf").stat().st_size > 1000
        assert not (base / "synthetic_render_check.svg").exists()
        assert not (base / "synthetic_render_check.png").exists()
        # A missing simulation day invalidates its first-crossing time.
        with h5py.File(sim_path, "a") as h5:
            del h5["sus/day2"]
            del h5["res/day2"]
        invalid = plot.failure_record(sim_path, "simulation", "Plate_AT50",
                                      "R11_AT50", "AT50", 300)
        assert invalid["reason"] == "missing_daily_population_observations"
        # Conflicting provenance must fail rather than pick an arbitrary seed.
        conflict = pd.concat([metadata, pd.DataFrame([
            {"protocol": "AT50", "run": "11", "initial_condition": "C2_AT50-C2"}
        ])], ignore_index=True)
        try:
            plot.discover_pairs(conflict, exp, sim)
        except ValueError:
            pass
        else:
            raise AssertionError("Conflicting seed provenance was accepted")
    print("Passed: exact seed + strategy joins, threshold/calendar, censoring, "
          "missing-day rejection, and conflicting-metadata rejection.")


if __name__ == "__main__":
    main()
