"""Audit all available trajectories and replace only Supplementary Figure 6 nonfailures.

Run before changing the plotting constants; --baseline can reuse its saved original
selection on later runs. Raw inputs are never modified. Selection is conditional on
observed failure, with the previously failing panels held fixed. See the companion
selection document for interpretation and numerical/visual review.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.figure_2.Event_Time_Analysis import EXPERIMENT_ROOT, SIMULATION_ROOT, _read_population_file
from shared.data_paths import DATA_ROOT
from shared.experiment_filters import EXCLUDE_SAMPLES
from figures.Suppl_Figures import Escape_Margin_Over_Time_By_Strategy as plot
from shared.strategy_colors import normalize_strategy

OUT = DATA_ROOT / "verification" / "supp6_failure_reselection_20261112"
METRICS = [m for m, _, _ in plot.METRICS]
WEIGHTS = {"population": .30, "margin": .25, "modulation": .15, "post": .10, "timing": .20}


def read_population_job(job):
    key, strategy, path = job
    source, plate, sample = key
    frame = _read_population_file(path, dataset_type=source, plate=plate, sample=sample, strategy=strategy)
    print(f"[population] {source} {plate}/{sample}", flush=True)
    return key, frame


def well_notes():
    """Read original workbook values and fill flags, without editing the workbook."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(DATA_ROOT / "Comments_on_wells.xlsx") as z:
        strings = ["".join(x.itertext()) for x in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        styles = ET.fromstring(z.read("xl/styles.xml"))
        xfs = styles.find("m:cellXfs", ns)
        rows = ET.fromstring(z.read("xl/worksheets/sheet1.xml")).findall("m:sheetData/m:row", ns)
        parsed = []
        for row in rows:
            cells = {}
            for c in row:
                col = 0
                for char in re.sub(r"\d", "", c.attrib["r"]):
                    col = col * 26 + ord(char) - 64
                v = c.find("m:v", ns)
                value = strings[int(v.text)] if c.attrib.get("t") == "s" else (v.text if v is not None else "")
                cells[col] = (value, int(xfs[int(c.attrib.get("s", 0))].attrib.get("fillId", 0)))
            parsed.append(cells)
        result = {}
        starts = [1, 9, 17, 26, 35, 44, 53, 62, 71]
        for row in parsed[2:26]:
            for plate, (start, end) in enumerate(zip(starts[:-1], starts[1:]), 1):
                fields = {parsed[1].get(i, (str(i), 0))[0]: row.get(i, ("", 0)) for i in range(start+1, end)}
                comment = fields.get("Special Events and comments", ("", 0))[0]
                review = any(fill in (2, 6) for key, (_, fill) in fields.items() if key != "Truncated [timestep]")
                result[(f"Plate_{plate}", row[start][0])] = dict(
                    comment=comment, wash_note=fields.get("washed res cells [timestep]", ("", 0))[0],
                    unresolved_flag=review,
                    quality_block=review or "hole in the cent" in comment.lower())
        return result


def census():
    table = pd.read_csv(plot.INPUT_CSV)
    table["strategy"] = table.strategy.map(normalize_strategy)
    table = table.drop_duplicates(["dataset_type", "plate", "sample", "day"], keep="last")
    groups = {key: g.sort_values("day") for key, g in table.groupby(["dataset_type", "plate", "sample"])}
    notes = well_notes()
    inputs = {}
    for path in sorted(EXPERIMENT_ROOT.glob("Plate_*/hdf5/*_total_poisson_padded.h5")):
        sample = path.name.removeprefix("G418_a2a4_").removesuffix("_total_poisson_padded.h5")
        strategy = normalize_strategy(sample.split("_", 1)[1].split("-", 1)[0])
        if strategy in plot.STRATEGIES:
            inputs[("experimental", path.parent.parent.name, sample)] = (strategy, path)
    for path in sorted(SIMULATION_ROOT.glob("Plate_*/hdf5/*_simulation.h5")):
        strategy = normalize_strategy(path.parent.parent.name.removeprefix("Plate_"))
        if strategy in plot.STRATEGIES:
            inputs[("simulation", path.parent.parent.name, path.stem.removesuffix("_simulation"))] = (strategy, path)
    for key, g in groups.items():
        if g.strategy.iloc[0] in plot.STRATEGIES and key not in inputs:
            inputs[key] = (g.strategy.iloc[0], None)
    jobs = [(key, strategy, path) for key, (strategy, path) in inputs.items()
            if path is not None
            and not (key[0] == "experimental" and EXCLUDE_SAMPLES.get(key[2], "unlisted") is None)
            and not (key[0] == "simulation" and int(re.match(r"R(\d+)", key[2])[1]) < 10)]
    with ProcessPoolExecutor(max_workers=4) as executor:
        populations = dict(executor.map(read_population_job, jobs))
    records, frames = [], []
    for (source, plate, sample), (strategy, path) in inputs.items():
        row = dict(dataset_type=source, strategy=strategy, plate=plate, sample=sample,
                   input_file=str(path), status="", failure_observed=False)
        cutoff = EXCLUDE_SAMPLES.get(sample, "unlisted") if source == "experimental" else "unlisted"
        row["manual_cutoff_day"] = cutoff if cutoff != "unlisted" else np.nan
        row.update(notes.get((plate, sample.rsplit("-", 1)[-1]), {}))
        key = (source, plate, sample)
        if cutoff is None:
            row["status"] = "fully excluded by canonical filter"
        elif source == "simulation" and int(re.match(r"R(\d+)", sample)[1]) < 10:
            row["status"] = "pilot run (<10)"
        elif path is None:
            row["status"] = "missing canonical population file"
        else:
            c = populations[key]
            g = groups.get(key, pd.DataFrame()).copy()
            if cutoff != "unlisted":
                c = c[c.day.le(cutoff)]
                if not g.empty:
                    g = g[g.day.le(cutoff)]
            if c.empty or c.day.min() != 1 or c.total.iloc[0] <= 0:
                row["status"] = "missing/invalid day-1 baseline"
            else:
                initial = c.total.iloc[0]
                hit = c.loc[c.res.ge(.5 * initial), "day"]
                row.update(ttf_days=hit.min() if len(hit) else np.nan, failure_observed=bool(len(hit)),
                           last_day=c.day.max(), initial_total=initial)
                if g.empty:
                    row["status"] = "no escape-margin trajectory"
                elif set(g.day) != set(c.day):
                    row["status"] = "population/margin days differ"
                elif source == "simulation" and not c.day.diff().dropna().eq(1).all():
                    row["status"] = "incomplete daily simulation"
                else:
                    g = g.merge(c[["day", "sus", "res", "total"]], on="day", validate="one_to_one")
                    g["normalized_sensitive"] = g.sus / initial
                    g["normalized_resistant"] = g.res / initial
                    g["normalized_total"] = (g.sus + g.res) / initial
                    g["normalized_confinement_margin"] = g.confinement_count.fillna(0) / initial
                    g["normalized_escape_margin"] = g.escaped_resistant_count.fillna(0) / initial
                    escape = g.loc[g.signed_escape_margin.gt(0), "day"]
                    row["tte_days"] = escape.min() if len(escape) else np.nan
                    row["status"] = "valid"
                    if not np.isfinite(g[METRICS]).all().all():
                        row["status"] = "nonfinite curve values"
                    else:
                        frames.append(g)
        records.append(row)
        print(f"[audit] {source} {plate}/{sample}: {row['status']}", flush=True)
    audit = pd.DataFrame(records)
    audit.to_csv(OUT / "cohort_audit.csv", index=False)
    values = pd.concat(frames, ignore_index=True)
    values.to_csv(OUT / "cohort_values.csv", index=False)
    return audit, values


def features(g, r):
    day = g.day.to_numpy()
    end = r.ttf_days if np.isfinite(r.ttf_days) else r.last_day
    phase = np.linspace(1, end, 21)
    out = {}
    for metric in METRICS:
        block = "margin" if "margin" in metric else "population"
        y = g[metric].to_numpy()
        pre = np.interp(phase, day, y)
        for j, v in enumerate(pre):
            out[f"{block}:{metric}:{j}"] = v
        observed = y[day <= end]
        # Excess variation measures rebounds/cycling beyond a monotone trajectory.
        for name, v in (("min", observed.min()), ("max", observed.max()),
                        ("rebound", np.abs(np.diff(observed)).sum() - abs(observed[-1]-observed[0]))):
            out[f"modulation:{metric}:{name}"] = v
        if np.isfinite(r.ttf_days) and day[-1] > end:
            for j, v in enumerate(np.interp(np.linspace(end, day[-1], 11), day, y)):
                out[f"post:{metric}:{j}"] = v
        else:
            for j in range(11):
                out[f"post:{metric}:{j}"] = np.nan
    out.update({"timing:ttf": r.ttf_days, "timing:tte": r.tte_days,
                "timing:lag": r.ttf_days-r.tte_days,
                "timing:escape_observed": float(np.isfinite(r.tte_days))})
    return out


def distances(frame, weights):
    # Empirical ranks give amplitude, timing and cycling equal opportunity without
    # allowing a single large population excursion to dominate every other feature.
    ranked = frame.rank(pct=True).fillna(.5).to_numpy()
    blocks = [c.split(":")[0] for c in frame]
    d2 = np.zeros((len(frame), len(frame)))
    for block, weight in weights.items():
        x = ranked[:, np.array(blocks) == block]
        d2 += weight * ((x[:, None, :] - x[None, :, :]) ** 2).mean(axis=2)
    return np.sqrt(np.maximum(d2, 0))


def objective(d, chosen):
    # Equal-weight energy distance to the empirical failure cohort, plus coverage.
    cross = d[:, chosen]
    energy = 2*cross.mean() - d[np.ix_(chosen, chosen)].mean() - d.mean()
    return .5*energy + .5*cross.min(axis=1).mean()


def choose(audit, values, baseline):
    valid = audit[audit.status.eq("valid")].copy()
    feature_records = []
    for r in valid.itertuples():
        g = values[values.dataset_type.eq(r.dataset_type) & values.plate.eq(r.plate) & values["sample"].eq(r.sample)]
        feature_records.append(dict(dataset_type=r.dataset_type, strategy=r.strategy, plate=r.plate,
                                    sample=r.sample, **features(g, r)))
    feature_table = pd.DataFrame(feature_records)
    feature_table.to_csv(OUT / "dynamic_features.csv", index=False)
    selection, scores = [], []
    for (source, strategy), cohort in valid.groupby(["dataset_type", "strategy"], sort=False):
        candidates = cohort[cohort.failure_observed].sort_values(["plate", "sample"]).reset_index(drop=True)
        old = baseline[baseline.dataset_type.eq(source) & baseline.strategy.eq(strategy)].sort_values("example")
        lookup = {(r.plate, r.sample): i for i, r in enumerate(candidates.itertuples())}
        locked = [lookup[(r.plate, r.sample)] for r in old.itertuples() if r.failure_observed]
        pool = [i for i, r in enumerate(candidates.itertuples())
                if i not in locked and not (pd.notna(r.quality_block) and bool(r.quality_block))]
        count = 5-len(locked)
        if len(pool) < count:
            raise ValueError(f"Insufficient reviewed failures in {source}/{strategy}: {len(pool)} for {count} slots")
        f = candidates[["dataset_type", "strategy", "plate", "sample"]].merge(feature_table, validate="one_to_one")
        f = f[[c for c in f if ":" in c]]
        d = distances(f, WEIGHTS)
        ranked = sorted((objective(d, locked+list(combo)), combo) for combo in combinations(pool, count))
        best_score, combo = ranked[0]
        # Display replacements from earlier to later failure in the vacated slots.
        additions = iter(sorted(combo, key=lambda i: (candidates.iloc[i].ttf_days, candidates.iloc[i]["sample"])))
        for r in old.itertuples():
            index = lookup[(r.plate, r.sample)] if r.failure_observed else next(additions)
            record = candidates.iloc[index].to_dict()
            record.update(example=r.example, original_plate=r.plate, original_sample=r.sample,
                          changed=not r.failure_observed)
            selection.append(record)
        for rank, (score, indexes) in enumerate(ranked[:10], 1):
            scores.append(dict(dataset_type=source, strategy=strategy, rank=rank, score=score,
                               replacements=";".join(candidates.iloc[i]["sample"] for i in indexes)))
        # Sensitivity to a timing-light and a shape-light weighting.
        for name, weights in (("timing_10pct", dict(population=.35, margin=.30, modulation=.15, post=.10, timing=.10)),
                              ("timing_30pct", dict(population=.25, margin=.20, modulation=.15, post=.10, timing=.30))):
            ds = distances(f, weights)
            alt_score, alt = min((objective(ds, locked+list(c)), c) for c in combinations(pool, count))
            scores.append(dict(dataset_type=source, strategy=strategy, rank=name, score=alt_score,
                               replacements=";".join(candidates.iloc[i]["sample"] for i in alt)))
        print(f"[choice] {source} {strategy}: {len(cohort)} valid, {len(candidates)} failures; "
              f"add {[candidates.iloc[i]['sample'] for i in combo]}; score={best_score:.5f}", flush=True)
    selected = pd.DataFrame(selection)
    selected.to_csv(OUT / "replacement_selection.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "selection_scores.csv", index=False)
    return selected


def review_plots(audit, values, selection):
    colors = {m: c for m, c, _ in plot.METRICS}
    for (source, strategy), group in audit[audit.status.eq("valid")].groupby(["dataset_type", "strategy"]):
        group = group.sort_values(["ttf_days", "plate", "sample"])
        chosen = selection[selection.dataset_type.eq(source) & selection.strategy.eq(strategy)]
        for page, start in enumerate(range(0, len(group), 12), 1):
            part = group.iloc[start:start+12]
            fig, axes = plt.subplots(int(np.ceil(len(part)/3)), 3, figsize=(15, 3*np.ceil(len(part)/3)), squeeze=False)
            for ax, r in zip(axes.flat, part.itertuples()):
                g = values[values.dataset_type.eq(source) & values.plate.eq(r.plate) & values["sample"].eq(r.sample)]
                for metric in METRICS:
                    ax.plot(g.day, g[metric], color=colors[metric], lw=1)
                ax.axhline(.5, color="grey", ls="--", lw=.5)
                if np.isfinite(r.ttf_days):
                    ax.axvline(r.ttf_days, color="grey", ls=":", lw=.8)
                match = chosen[chosen.plate.eq(r.plate) & chosen["sample"].eq(r.sample)]
                tag = " NEW" if len(match) and match.changed.iloc[0] else " KEEP" if len(match) else ""
                ax.set_title(f"{r.plate} {r.sample}{tag}\nTTE {r.tte_days:g}, TTF {r.ttf_days:g}; end {r.last_day:g}", fontsize=9)
                ax.set_xlim(plot.X_LIMITS[source]); ax.set_ylim(bottom=0)
            for ax in list(axes.flat)[len(part):]:
                ax.set_visible(False)
            fig.suptitle(f"{source} {strategy}: canonical counts and margins; grey line = failure")
            fig.tight_layout()
            fig.savefig(OUT / f"review_{source}_{strategy}_{page}.png", dpi=100)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached", action="store_true", help="Use the census saved by an earlier full raw-data scan.")
    parser.add_argument("--baseline", type=Path, default=OUT / "original_selection.csv")
    args = parser.parse_args()
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Data root is unavailable; reconnect the shared drive or set EICHE_FISCHER_DATA_ROOT: {DATA_ROOT}")
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.baseline.exists():
        pd.read_csv(plot.DATA_DIR / plot.SELECTION_NAME).to_csv(args.baseline, index=False)
    baseline = pd.read_csv(args.baseline)
    if args.cached:
        audit, values = (pd.read_csv(OUT / name) for name in ("cohort_audit.csv", "cohort_values.csv"))
    else:
        audit, values = census()
    selected = choose(audit, values, baseline)
    review_plots(audit, values, selected)
    print(selected[["dataset_type", "strategy", "example", "sample", "ttf_days", "tte_days", "changed"]].to_string(index=False))


if __name__ == "__main__":
    main()
