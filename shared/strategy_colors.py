from __future__ import annotations

"""
TL;DR
-----
This module is the canonical strategy naming and color registry for the
paper figures. Any script that needs legend order, display labels, or
consistent treatment colors should import from here instead of hard-coding
 its own palette.

Detailed
--------
Inputs:
- Raw strategy names coming from filenames, sample IDs, or older aliases
  such as `ATAgent1`, `ATAgent2`, or `Dens`.

Outputs:
- Canonical ordering (`STRATEGY_ORDER`).
- Canonical display labels (`DISPLAY_NAME`).
- Canonical colors (`COLOR_MAP`).
- Shared non-strategy figure colors (`FIGURE_COLOR_MAP`).
- Dataset fill conventions (`DATASET_FILL_STYLE`).
- Small helpers to normalize aliases and validate coverage.

How it works:
- `normalize_strategy(...)` collapses historical naming variants onto one
  shared vocabulary.
- `get_color(...)` and `get_display_name(...)` ensure that the same therapy
  is drawn and labeled the same way across all panels.
- `validate_strategies(...)` makes missing palette entries fail loudly
  instead of silently producing inconsistent figure styling.

Pipeline role:
- Shared plotting metadata layer used by the active figure scripts.
"""

# Canonical order for legends / plotting order when needed.
STRATEGY_ORDER: list[str] = [
    "CT",
    "NT",
    "AT50",
    "AT100",
    "LV-Agent",
    "KPP-Agent",
    "AT100+",
    "Pulse",
]

# Raw analysis names that should resolve to canonical strategy keys.
STRATEGY_ALIASES: dict[str, str] = {
    "ATAgent2": "LV-Agent",
    "Agent": "LV-Agent",
    "ATAgent1": "KPP-Agent",
    "ATAgent": "KPP-Agent",
    "Dens": "AT100+",
}

# Dens is the experimental-schedule simulation used in the main figures.
# ATplusdayone is the distinct, older simulation, shown only where requested.

# Canonical color map.
COLOR_MAP: dict[str, str] = {
    "CT": "#117733",
    "NT": "#332288",
    "AT50": "#88CCEE",
    "AT100": "#CC6677",
    "LV-Agent": "#882255",
    "KPP-Agent": "#8C564B",
    "AT100+": "#44AA99",
    "ATplusdayone": "#332288",
    "Pulse": "#AA4499",
}

# Shared non-strategy colors used in figure annotations and encodings.
FIGURE_COLOR_MAP: dict[str, str] = {
    "yellow": "#DAA520",
    "blue": "#4169E1",
    "treatment": "#BFBFBF",
    "confinement margin": "#92278F",
    "escaped cells": "#ED1C24",
}

# Descriptive fill conventions; plotting code translates these labels into
# the appropriate solid fill or hatch pattern for its graphics library.
DATASET_FILL_STYLE: dict[str, str] = {
    "In vitro": "full",
    "In silico": "striped",
}

# Canonical display names.
DISPLAY_NAME: dict[str, str] = {
    "CT": "CT",
    "NT": "NT",
    "AT50": "AT50",
    "AT100": "AT100",
    "LV-Agent": "LV-Agent",
    "KPP-Agent": "KPP-Agent",
    "AT100+": r"AT$^{+}$",
    "ATplusdayone": r"AT$^{+}_{100}$",
    "Pulse": "Pulse",
}

# Matplotlib equivalents of the manuscript's AT\textsuperscript{...} labels.
# Keep these display strings separate from the stable analysis/file identifiers.
SUPERSCRIPT_DISPLAY_NAME: dict[str, str] = {
    # Render base names through the same text engine, including plain CT.
    "CT": r"$\mathrm{CT}$",
    "AT50": r"$\mathrm{AT}^{50}$",
    "AT100": r"$\mathrm{AT}^{100}$",
    "LV-Agent": r"$\mathrm{AT}^{\mathrm{agent}}$",
    "AT100+": r"$\mathrm{AT}^{+}$",
    "ATplusdayone": r"$\mathrm{AT}^{+}_{100}$",
}


def normalize_strategy(strategy: str) -> str:
    """Map raw strategy names to the canonical strategy key used in this repo."""
    key = str(strategy).strip()
    return STRATEGY_ALIASES.get(key, key)


def get_color(strategy: str, default: str = "#000000") -> str:
    """Return the canonical color for a strategy or alias."""
    return COLOR_MAP.get(normalize_strategy(strategy), default)


def get_display_name(strategy: str, *, superscript: bool = False) -> str:
    """Return a display label, optionally using manuscript AT superscripts."""
    canonical = normalize_strategy(strategy)
    if superscript and canonical in SUPERSCRIPT_DISPLAY_NAME:
        return SUPERSCRIPT_DISPLAY_NAME[canonical]
    return DISPLAY_NAME.get(canonical, canonical)


def validate_strategies(strategies: list[str]) -> None:
    """Raise a clear error if any strategy resolves to a missing color."""
    missing = [s for s in strategies if normalize_strategy(s) not in COLOR_MAP]
    if missing:
        raise KeyError(f"Missing colors for strategies: {missing}. Add them to COLOR_MAP.")
