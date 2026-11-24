"""Canonical read-only inputs and derived-data locations for the STEM paper.

Raw experimental and simulation folders are inputs only.  Scripts must write
derived data through one of the output helpers below, never into either raw
folder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


# Edit this directory to switch data for the pipeline and standalone scripts.
# It contains exp_raw_data, sim_raw_data, intermediate_analysis_data,
# figure_data, and figure_plots.
DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "Data"

DATA_ROOT_ENV = "EICHE_FISCHER_DATA_ROOT"
LEGACY_DATA_ROOT = Path(r"Y:\Members\MaxE\Publications\STEM Paper\Data")


def _configured_data_root() -> Path:
    """Use the per-run environment override when supplied."""
    configured = os.environ.get(DATA_ROOT_ENV)
    path = Path(configured or DATA_DIRECTORY).expanduser().absolute()
    text = os.fspath(path)
    # Explicit extended paths keep the pinned Windows 3.10 environments usable
    # for the study's long well/day and figure filenames, including deep builds.
    if os.name == 'nt' and len(text) >= 100 and not text.startswith('\\\\?\\'):
        text = '\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text
        path = Path(text)
    return path


DATA_ROOT = _configured_data_root()
EXP_RAW_DATA = DATA_ROOT / "exp_raw_data"
SIM_RAW_DATA = DATA_ROOT / "sim_raw_data"
SIMULATION_DATABASE = SIM_RAW_DATA / "data-DB"
SIMULATION_FITTING = SIM_RAW_DATA / "experiment_fitting"

INTERMEDIATE_ANALYSIS_DATA = DATA_ROOT / "intermediate_analysis_data"
FIGURE_DATA = DATA_ROOT / "figure_data"
FIGURE_PLOTS = DATA_ROOT / "figure_plots"
VERIFICATION_DATA = DATA_ROOT / "verification"
MODELS = DATA_ROOT / "Models"
FITTING = DATA_ROOT / "Fitting"

# Spawned analysis workers import this module too, so the clean-run guard also
# covers their direct HDF5 reads rather than only the coordinating interpreter.
from reproduction.isolation import install as _install_reproduction_isolation
_install_reproduction_isolation(Path(__file__).resolve().parents[1], DATA_ROOT)

_DATASETS = {"experimental", "simulation", "combined", "legacy"}


def rebase_legacy_data_path(path: str | os.PathLike[str]) -> Path:
    """Rebase a stored path below the former shared-drive root.

    Several versioned YAML files record the exact paths used for the published
    analyses. This helper replaces the former shared-drive prefix with the
    configured data directory.
    """
    text = os.fspath(path)
    normalized_text = text.replace("\\", "/")
    normalized_legacy = os.fspath(LEGACY_DATA_ROOT).replace("\\", "/")
    if normalized_text.casefold() == normalized_legacy.casefold():
        return DATA_ROOT
    prefix = normalized_legacy + "/"
    if normalized_text.casefold().startswith(prefix.casefold()):
        relative = normalized_text[len(prefix):]
        return DATA_ROOT.joinpath(*[part for part in re.split(r"[\\/]+", relative) if part])
    return Path(text).expanduser()


def resolve_stored_data_path(path: str | os.PathLike[str]) -> Path:
    """Resolve archive manifest paths after moving the Data directory.

    When using a relocated data directory, retain the suffix beginning at a
    canonical archive directory, even for paths saved on another OS or share.
    This is for manifest data references, not arbitrary CLI or repository paths.
    """
    text = os.fspath(path)
    if DATA_ROOT != LEGACY_DATA_ROOT or os.environ.get(DATA_ROOT_ENV):
        parts = [part for part in re.split(r"[\\/]+", text) if part]
        roots = {"exp_raw_data", "sim_raw_data", "intermediate_analysis_data",
                 "figure_data", "figure_plots", "verification"}
        for index, part in enumerate(parts):
            if part in roots:
                suffix = parts[index:]
                if ".." in suffix:
                    raise ValueError(f"Archive path contains parent traversal: {text}")
                return DATA_ROOT.joinpath(*suffix)
    return rebase_legacy_data_path(path)


def _validate_dataset(dataset: str) -> str:
    normalized = dataset.strip().lower()
    if normalized not in _DATASETS:
        raise ValueError(f"Unknown dataset category {dataset!r}; use one of {sorted(_DATASETS)}")
    return normalized


def intermediate_dir(dataset: str, *parts: str | Path, create: bool = True) -> Path:
    """Return a derived-analysis directory, optionally creating it."""
    path = INTERMEDIATE_ANALYSIS_DATA / _validate_dataset(dataset)
    for part in parts:
        path /= part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def figure_data_dir(figure: str, *parts: str | Path, create: bool = True) -> Path:
    """Return plot-ready source-data storage organised by manuscript figure.

    The canonical layout is ``figure_data/Figure_X/...``.  Calls using the
    short-lived dataset-first API are accepted for compatibility, but the
    dataset label is deliberately omitted from the resulting path.
    """
    if figure.strip().lower() in _DATASETS:
        if not parts:
            raise ValueError("A figure name is required after the legacy dataset label")
        figure, *parts = parts
    path = FIGURE_DATA / figure
    for part in parts:
        path /= part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def figure_plot_dir(figure: str, *parts: str | Path, create: bool = True) -> Path:
    """Return the final graphics directory for one manuscript figure."""
    path = FIGURE_PLOTS / figure
    for part in parts:
        path /= part
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


COMBINED_RESULTS_CSV = intermediate_dir("simulation", "basic", create=False) / "combined_results.csv"
