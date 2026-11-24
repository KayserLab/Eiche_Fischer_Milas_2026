"""Verify failure-only Supplementary Figure 6 against the full-cohort raw-data audit."""
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.figure_2.Select_Failing_Escape_Margin_Examples import OUT, METRICS
from figures.Suppl_Figures import Escape_Margin_Over_Time_By_Strategy as plot


def main():
    original = pd.read_csv(OUT / "original_selection.csv")
    audit = pd.read_csv(plot.DATA_DIR / plot.SELECTION_NAME)
    values = pd.read_csv(plot.DATA_DIR / plot.VALUES_NAME)
    raw_audit = pd.read_csv(OUT / "cohort_audit.csv")
    raw_values = pd.read_csv(OUT / "cohort_values.csv")
    slots = ["dataset_type", "strategy", "example"]
    keys = ["dataset_type", "plate", "sample"]
    assert len(audit) == 50 and audit.failure_observed.all()
    assert audit.groupby(["dataset_type", "strategy"]).size().eq(5).all()
    assert not audit.duplicated(keys).any()
    compared = original.merge(audit, on=slots, suffixes=("_old", "_new"), validate="one_to_one")
    changed = compared.sample_old.ne(compared.sample_new) | compared.plate_old.ne(compared.plate_new)
    assert (changed == ~compared.failure_observed_old).all()
    assert changed.sum() == 13
    assert compared.loc[changed].groupby("dataset_type").size().to_dict() == {"experimental": 7, "simulation": 6}
    independent = audit.merge(raw_audit, on=keys, suffixes=("_plot", "_raw"), validate="one_to_one")
    assert len(independent) == 50
    for column in ["ttf_days", "tte_days", "last_day", "initial_total"]:
        assert np.allclose(independent[column+"_plot"], independent[column+"_raw"], equal_nan=True)
    curve_check = values.merge(raw_values[keys+["day"]+METRICS], on=keys+["day"], suffixes=("_plot", "_raw"), validate="one_to_one")
    assert len(curve_check) == len(values)
    for metric in METRICS:
        assert np.allclose(curve_check[metric+"_plot"], curve_check[metric+"_raw"], atol=1e-14, rtol=0)
    assert np.allclose(values.normalized_total, values.normalized_sensitive+values.normalized_resistant, atol=1e-14, rtol=0)
    for r in audit.itertuples():
        g = values[values.dataset_type.eq(r.dataset_type) & values.plate.eq(r.plate) & values["sample"].eq(r.sample)]
        assert g.loc[g.normalized_resistant.ge(.5), "day"].min() == r.ttf_days
        assert g.day.max() <= plot.X_LIMITS[r.dataset_type][1]
        if pd.notna(r.manual_cutoff_day):
            assert g.day.max() <= r.manual_cutoff_day
    for source, filename in plot.PDF_NAMES.items():
        path = plot.PLOT_DIR / filename
        raw = path.read_bytes()
        assert b"/FontFile2" in raw and b"/Subtype /Image" not in raw
        reader = PdfReader(path)
        assert len(reader.pages) == 1
        page = reader.pages[0]
        assert np.allclose([float(page.mediabox.width), float(page.mediabox.height)], np.array([180,185])/25.4*72, atol=.01)
        text = page.extract_text()
        assert text.count("Time [days]") == 5
        pattern = r"P\d+\s+[A-H]\d+" if source == "experimental" else r"\bR\d+\b"
        assert len(re.findall(pattern, text)) == 25
        print(f"PASS: {source}: 25 panels, one page, correct dimensions, embedded fonts, vector artwork.")
    print("PASS: all 50 failures, exactly 13 replacements, all 37 original failures locked to their original slots.")
    print("PASS: all five curves and both event times agree with the independent full-cohort HDF5 scan; cutoffs respected.")


if __name__ == "__main__":
    main()
