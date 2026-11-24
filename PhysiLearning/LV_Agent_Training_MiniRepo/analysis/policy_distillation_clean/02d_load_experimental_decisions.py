"""Step 2d: reconstruct experimental histories and ask the PPO agent.

Only ATAgent2 wells are included.
Reads Results/*.txt (name_dXX sensitive_count resistant_count) and the plate's
treatment_schedule.txt. Integers in the schedule are calendar days: listed
day = treatment ON, unlisted = OFF. Decisions occur on days 1, 3, 5, 8, ...
(2-2-3-day intervals). Daily treatment entries do not add history slots.
A token like 12s stops observation before calendar day 12.

Optional --format hdf5 counts rows in sus/d_XX and res/d_XX point clouds
(including pandas fixed-format block0_values), or reads scalar counts at those
paths. Narrow --pattern to exactly one corrected file per well.

Output: data/experimental_decisions.csv. logged_action is the agent prediction
for downstream compatibility; executed_action is the actual schedule.
These predictions describe what the agent would do on measured histories.
They are not proof that a well was controlled by this agent.
"""
import argparse
import io
import re
import sys
import warnings
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import AGENT_PATH, EXPERIMENTAL_DATA_FILE, FEATURES, load_agent, p_treat

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent.parent))
from shared.data_paths import EXP_RAW_DATA
DEFAULT_ROOT = EXP_RAW_DATA
# PyCharm Run-button setting: folder containing the agent plate folders.
# Each plate must contain Results/*.txt and treatment_schedule.txt.
EXPERIMENTAL_DATA_ROOT = EXP_RAW_DATA
LOCAL_AGENT = AGENT_PATH
SAMPLE_RE = re.compile(r"(?P<sample>[A-H]\d+_[A-Za-z0-9]+-[A-H]\d+)(?:_|\.|$)", re.I)


def is_agent_strategy(strategy, selected="ATAgent2"):
    """Allow ATAgent2 wells only (case-insensitive)."""
    return strategy.casefold() == selected.casefold() == "atagent2"


def read_schedule(path, samples=None):
    """Sample -> (treated calendar days, optional stop day)."""
    schedules = {}
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            sample, *tokens = line.split()
            if samples is not None and sample not in samples:
                continue
            if sample in schedules:
                raise ValueError(f"{path}:{number}: duplicate sample {sample}")
            treated, stop = set(), None
            for token in tokens:
                match = re.fullmatch(r"(\d+)(s?)", token)
                if not match or int(match[1]) < 1:
                    raise ValueError(f"{path}:{number}: unsupported token {token!r}")
                day = int(match[1])
                if match[2]:
                    if stop is not None:
                        raise ValueError(f"{path}:{number}: multiple stop days")
                    stop = day
                else:
                    treated.add(day)
            if stop is not None and any(i >= stop for i in treated):
                warnings.warn(f"{path}:{number}: ignoring treatment entries at/after stop day {stop}")
                treated = {day for day in treated if day < stop}
            schedules[sample] = treated, stop
    return schedules


def validate_counts(rows, path):
    table = pd.DataFrame(rows, columns=["measurement_day", "sensitive_count", "resistant_count"])
    if table.empty or table["measurement_day"].duplicated().any():
        raise ValueError(f"{path}: empty data or duplicate measurement days")
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path}: nonfinite or negative day/count")
    table = table.sort_values("measurement_day").reset_index(drop=True)
    table["total_count"] = table["sensitive_count"] + table["resistant_count"]
    if table["total_count"].iloc[0] <= 0:
        raise ValueError(f"{path}: initial cell count must be positive")
    return table


def read_results(path):
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.split()
            match = re.search(r"_d(\d+)$", parts[0])
            if len(parts) != 3 or not match:
                raise ValueError(f"{path}:{number}: expected name_dXX sensitive resistant")
            rows.append((int(match[1]), float(parts[1]), float(parts[2])))
    return validate_counts(rows, path)


def read_hdf5(path):
    """Read scalar counts or count rows in the established channel/day layout."""
    rows = []
    with h5py.File(path, "r") as handle:
        if not all(c in handle for c in ("sus", "res")):
            raise ValueError(f"{path}: expected sus/d_XX and res/d_XX")
        for key in set(handle["sus"]) | set(handle["res"]):
            match = re.fullmatch(r"d_(\d+)", key)
            if not match:
                raise ValueError(f"{path}: unsupported time key {key!r}")
            counts = []
            for channel in ("sus", "res"):
                name = f"{channel}/{key}"
                if name not in handle:
                    raise ValueError(f"{path}: missing {name}; cannot assume zero cells")
                node = handle[name]
                if isinstance(node, h5py.Group):
                    if "block0_values" not in node:
                        raise ValueError(f"{path}: unsupported pandas layout at {name}")
                    node = node["block0_values"]
                if node.ndim == 0:
                    counts.append(float(node[()]))
                elif node.ndim == 2 and node.shape[1] in (2, 3):
                    counts.append(float(node.shape[0]))
                else:
                    raise ValueError(f"{path}: unsupported shape at {name}: {node.shape}")
            rows.append((int(match[1]), *counts))
    return validate_counts(rows, path)


def decisions_of_well(counts, treated, stop, run, sample, source, cutoff=None):
    """Normalize to the first count and interleave burdens with PRIOR actions.

    Sample the DAILY schedule at complete M/W/F decision visits, holding each
    action for 2, 2, then 3 days. Validate that the daily schedule stays constant
    between observed decisions. Never add a history slot for an intervening day
    or substitute predicted actions for recorded experimental treatment.
    """
    if stop is not None:
        counts = counts[counts["measurement_day"] < stop]
    if cutoff is not None:
        counts = counts[counts["measurement_day"] <= cutoff]
    if counts.empty:
        return []
    days = counts["measurement_day"].to_numpy(dtype=int)
    expected = np.array([7 * (i // 3) + (0, 2, 4)[i % 3] for i in range(len(days))])
    if not np.array_equal(days - days[0], expected):
        raise ValueError(f"{source}: missing/irregular visits; expected 2-2-3 decision cadence")
    initial = float(counts["total_count"].iloc[0])
    history, previous, rows = [[0.0, 0]] * 3, 0, []
    for index, row in enumerate(counts.itertuples(index=False), 1):
        day = int(row.measurement_day)
        if (stop is not None and day >= stop) or (cutoff is not None and day > cutoff):
            break
        if index > 1:
            previous_day = int(days[index - 2])
            if any(int(d in treated) != previous for d in range(previous_day, day)):
                raise ValueError(
                    f"{source}: treatment changes between decision days {previous_day} and {day}; "
                    "cannot represent this interval as one binary action")
        action = int(day in treated)
        history = (history + [[float(row.total_count / initial), previous]])[-4:]
        rows.append({
            "run": run, "day": int(row.measurement_day - days[0]), "sample": sample,
            "measurement_day": int(row.measurement_day), "measurement_index": index,
            "initial_count": initial, "sensitive_count": row.sensitive_count,
            "resistant_count": row.resistant_count, "total_count": row.total_count,
            "source_file": str(source),
            **dict(zip(FEATURES, [v for pair in history for v in pair])),
            "executed_action": action,
        })
        previous = action
    return rows


def load_experimental_agent(path):
    """Also accept the repository's unpacked SB3 agent, without writing a zip."""
    if not path.is_dir():
        return load_agent(path)
    for name in ("data", "policy.pth", "pytorch_variables.pth"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Incomplete unpacked SB3 agent: {path / name}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for item in path.iterdir():
            if item.is_file():
                archive.writestr(item.name, item.read_bytes())
    buffer.seek(0)
    return load_agent(buffer)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EXPERIMENTAL_DATA_ROOT)
    parser.add_argument("--format", choices=("results", "hdf5"), default="results")
    parser.add_argument("--pattern", help="Glob relative to data root, one file per well")
    parser.add_argument("--agent", type=Path, default=LOCAL_AGENT if LOCAL_AGENT.is_dir() else AGENT_PATH,
                        help="SB3 zip or unpacked directory; defaults to the repository's experimental agent")
    parser.add_argument("--output", type=Path, default=EXPERIMENTAL_DATA_FILE)
    parser.add_argument("--strategy", choices=("ATAgent2",), default="ATAgent2",
                        help="Only ATAgent2 wells are supported")
    parser.add_argument("--include-excluded", action="store_true", help="Disable manual exclusions/truncations")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO.parent.parent))
    from shared.experiment_filters import EXCLUDE_SAMPLES

    pattern = args.pattern or ("Plate*/Results/Results_*.txt" if args.format == "results" else "Plate*/hdf5/*_total.h5")
    files = sorted(args.data_root.glob(pattern))
    if not files:
        parser.error(f"No input files matching {pattern!r} under {args.data_root}")
    rows, seen, skipped, schedules = [], set(), 0, {}
    for path in files:
        match = SAMPLE_RE.search(path.name)
        if not match:
            raise ValueError(f"Cannot extract sample identifier from {path}")
        sample = match["sample"]
        # Results_<assay>_<sample>.txt contains total counts. Companion _seg
        # and _overexposed files are components, not additional trajectories.
        if args.format == "results" and not path.stem.endswith(sample):
            continue
        strategy = sample.split("_", 1)[1].split("-", 1)[0]
        if not is_agent_strategy(strategy, args.strategy):
            continue
        cutoff = EXCLUDE_SAMPLES.get(sample, "keep") if not args.include_excluded else "keep"
        if cutoff is None:
            skipped += 1
            continue
        plate = next((p for p in path.parents if (p / "treatment_schedule.txt").is_file()), None)
        if plate is None:
            raise FileNotFoundError(f"No plate treatment_schedule.txt for {path}")
        run = f"{plate.relative_to(args.data_root).as_posix()}/{sample}"
        if run in seen:
            raise ValueError(f"Multiple input files for {run}; narrow --pattern")
        seen.add(run)
        schedule_key = (plate, sample)
        if schedule_key not in schedules:
            schedules[schedule_key] = read_schedule(plate / "treatment_schedule.txt", {sample})
        if sample not in schedules[schedule_key]:
            raise ValueError(f"{sample} missing from {plate}/treatment_schedule.txt")
        treated, stop = schedules[schedule_key][sample]
        counts = read_results(path) if args.format == "results" else read_hdf5(path)
        rows.extend(decisions_of_well(counts, treated, stop, run, sample, path,
                                     None if cutoff == "keep" else float(cutoff)))
    if not rows:
        parser.error(f"No {args.strategy} decisions found after filters under {args.data_root}. "
                     "Supply --data-root pointing to the plates containing agent wells (e.g. plates 6–7). "
                     "No output was written.")
    table = pd.DataFrame(rows)
    agent = load_experimental_agent(args.agent)
    table["p_treat"] = p_treat(agent, table[FEATURES])
    table["agent_action"] = (table["p_treat"] > 0.5).astype(int)
    table["logged_action"] = table["agent_action"]  # downstream teacher-label compatibility
    table["agrees_with_schedule"] = table["agent_action"] == table["executed_action"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} decisions from {table['run'].nunique()} wells; {skipped} wells excluded.")
    print(f"Agent/schedule agreement: {table['agrees_with_schedule'].mean():.2%} (descriptive across strategies).")
    print(table.head(8).round(3).to_string(index=False))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
