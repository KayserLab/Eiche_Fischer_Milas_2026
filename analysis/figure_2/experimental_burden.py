"""Experimental burden reader shared by the current Figure 2 plots."""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from shared.data_paths import EXP_RAW_DATA
from shared.strategy_colors import normalize_strategy

EXPERIMENT_ROOT = EXP_RAW_DATA
CORE_STRATEGIES = {"CT", "AT50", "AT100"}
EXPANDED_STRATEGIES = CORE_STRATEGIES | {"LV-Agent", "AT100+"}

def read_experimental_burdens(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    pattern = re.compile(r"Results_G418_a2a4_(?P<sample>[A-H]\d+_(?P<strategy>[^-]+)-[A-H]\d+)\.txt$")
    for plate_dir in sorted(root.glob("Plate_*")):
        for path in sorted((plate_dir / "Results").glob("Results_G418_a2a4_*.txt")):
            match = pattern.match(path.name)
            if not match:
                continue
            strategy = normalize_strategy(match.group("strategy"))
            if strategy not in EXPANDED_STRATEGIES:
                continue
            table = pd.read_csv(path, sep=r"\s+", header=None, names=["token", "sensitive", "resistant"])
            table["day"] = pd.to_numeric(table["token"].str.extract(r"_d(\d+)$", expand=False), errors="coerce")
            table["plate"] = plate_dir.name
            table["sample"] = match.group("sample")
            table["strategy"] = strategy
            table["dataset_type"] = "In vitro"
            frames.append(table[["dataset_type", "plate", "sample", "strategy", "day", "sensitive", "resistant"]])
    if not frames:
        raise ValueError(f"No CT/AT50/AT100 result files found below {root}")
    data = pd.concat(frames, ignore_index=True).dropna(subset=["day", "sensitive", "resistant"])
    return data.sort_values(["plate", "sample", "day"])


