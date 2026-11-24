"""
Central figure tools for the paper.

This module encodes the submission-oriented plotting standards used by the
figure scripts. The central rule is that requested width/height values refer to
the physical axes box: the length of the x-axis and y-axis. Labels, tick labels,
legends, titles, colorbars, and outside annotations live in the surrounding
margin and do not change the requested axes dimensions.

Key standards implemented here:
- Graphs and line art should be exported as editable vector files.
- Figure backgrounds are transparent by default.
- Text uses a sans-serif font, preferably Helvetica/Arial.
- Visible text should stay in the 5-7 pt range.
- Original research output is kept in RGB by Matplotlib's default color model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt

from shared.strategy_colors import get_color as get_strategy_color, validate_strategies
from shared.data_paths import FIGURE_PLOTS, figure_plot_dir
from reproduction.runtime import install_plot_capture

install_plot_capture()


Number = Union[int, float]
SizeValue = Union[str, Number]


# ============================================================
# Nature / manuscript standards
# ============================================================

NATURE_TEXT_MIN_PT = 5.0
NATURE_TEXT_MAX_PT = 7.0
NATURE_DEFAULT_TEXT_PT = 7.0
NATURE_DEFAULT_TICK_PT = 6.0
NATURE_DEFAULT_LEGEND_PT = 6.0
NATURE_DEFAULT_DPI = 300
NATURE_DEFAULT_AXIS_LINEWIDTH_PT = 0.8
NATURE_DEFAULT_TICK_LENGTH_PT = 2.5

# Nature research artwork widths. In this repo these presets are used as axes
# box widths unless a script explicitly chooses otherwise.
COL_WIDTHS_MM: Dict[str, float] = {
    "nature_single": 88.0,
    "nature_double": 180.0,
    "single": 88.0,
    "double": 180.0,
    "thesis": 160.0,
    "onehalf": 114.0,
}

# Conservative defaults around an axes box. Scripts should tune these if labels,
# legends, or significance bars need extra room.
DEFAULT_MARGIN_LEFT_MM = 14.0
DEFAULT_MARGIN_RIGHT_MM = 4.0
DEFAULT_MARGIN_BOTTOM_MM = 12.0
DEFAULT_MARGIN_TOP_MM = 4.0
DEFAULT_PANEL_GAP_MM = 8.0

VECTOR_EXTENSIONS = {".ai", ".eps", ".pdf", ".svg"}
BITMAP_EXTENSIONS = {".bmp", ".gif", ".gimp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
# Final manuscript graphics only.  Plot-ready tables belong in ``figure_data``
# and are addressed with ``data_paths.figure_data_dir``.
PAPER_FIGURE_ROOT = FIGURE_PLOTS


@dataclass(frozen=True)
class AxesBoxSpec:
    """Editable per-plot size parameters for an axes box."""

    width_mm: float = COL_WIDTHS_MM["nature_single"]
    height_mm: float = 55.0
    left_mm: float = DEFAULT_MARGIN_LEFT_MM
    right_mm: float = DEFAULT_MARGIN_RIGHT_MM
    bottom_mm: float = DEFAULT_MARGIN_BOTTOM_MM
    top_mm: float = DEFAULT_MARGIN_TOP_MM


@dataclass(frozen=True)
class GridSpecMM:
    """Editable per-plot size parameters for a regular axes grid."""

    ax_width_mm: float = 42.0
    ax_height_mm: float = 32.0
    left_mm: float = DEFAULT_MARGIN_LEFT_MM
    right_mm: float = DEFAULT_MARGIN_RIGHT_MM
    bottom_mm: float = DEFAULT_MARGIN_BOTTOM_MM
    top_mm: float = DEFAULT_MARGIN_TOP_MM
    wspace_mm: float = DEFAULT_PANEL_GAP_MM
    hspace_mm: float = DEFAULT_PANEL_GAP_MM


def mm_to_inches(mm: Number) -> float:
    """Convert millimetres to inches."""
    return float(mm) / 25.4


def inches_to_mm(inches: Number) -> float:
    """Convert inches to millimetres."""
    return float(inches) * 25.4


def resolve_width_mm(width: SizeValue) -> float:
    """Resolve a width preset or numeric width in millimetres."""
    if isinstance(width, str):
        if width not in COL_WIDTHS_MM:
            raise KeyError(
                f"Unknown width preset '{width}'. Available: {sorted(COL_WIDTHS_MM)}"
            )
        return float(COL_WIDTHS_MM[width])
    return float(width)


def paper_figure_data_dir(script_path: Union[str, Path]) -> Path:
    """
    Return the final-graphics directory for a script under figures/Figure_X.

    Examples:
    - figures/Figure_2/foo.py -> .../figure_plots/Figure_2
    - figures/Suppl_Figures/foo.py -> .../figure_plots/Suppl_Figures
    """
    path = Path(script_path).resolve()
    figure_dir = path.parent.name
    return figure_plot_dir(figure_dir)


def assert_nature_text_size(size_pt: Number, *, name: str = "text") -> float:
    """Validate a text size against the 5-7 pt Nature guidance."""
    size = float(size_pt)
    if not (NATURE_TEXT_MIN_PT <= size <= NATURE_TEXT_MAX_PT):
        raise ValueError(
            f"{name} size must be between {NATURE_TEXT_MIN_PT:g} and "
            f"{NATURE_TEXT_MAX_PT:g} pt; got {size:g} pt."
        )
    return size


def nature_rcparams(
    *,
    text_pt: Number = NATURE_DEFAULT_TEXT_PT,
    tick_pt: Number = NATURE_DEFAULT_TICK_PT,
    legend_pt: Number = NATURE_DEFAULT_LEGEND_PT,
) -> Dict[str, object]:
    """Return rcParams matching the submission guidelines."""
    text_size = assert_nature_text_size(text_pt, name="text")
    tick_size = assert_nature_text_size(tick_pt, name="tick")
    legend_size = assert_nature_text_size(legend_pt, name="legend")

    return {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        # MathText has its own font configuration.  Keep mathematical axis
        # labels editable as Arial in exported PDF/SVG files as well.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial",
        "mathtext.sf": "Arial",
        "font.size": text_size,
        "axes.labelsize": text_size,
        "axes.titlesize": text_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "legend.title_fontsize": legend_size,
        "axes.linewidth": NATURE_DEFAULT_AXIS_LINEWIDTH_PT,
        "xtick.major.width": NATURE_DEFAULT_AXIS_LINEWIDTH_PT,
        "ytick.major.width": NATURE_DEFAULT_AXIS_LINEWIDTH_PT,
        "xtick.major.size": NATURE_DEFAULT_TICK_LENGTH_PT,
        "ytick.major.size": NATURE_DEFAULT_TICK_LENGTH_PT,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": NATURE_DEFAULT_DPI,
        "savefig.dpi": NATURE_DEFAULT_DPI,
        "savefig.transparent": True,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    }


def apply_nature_style(
    *,
    text_pt: Number = NATURE_DEFAULT_TEXT_PT,
    tick_pt: Number = NATURE_DEFAULT_TICK_PT,
    legend_pt: Number = NATURE_DEFAULT_LEGEND_PT,
    overrides: Optional[Dict[str, object]] = None,
) -> None:
    """Apply the central paper style globally for the current process."""
    params = nature_rcparams(text_pt=text_pt, tick_pt=tick_pt, legend_pt=legend_pt)
    if overrides:
        params.update(overrides)
    mpl.rcParams.update(params)


@contextmanager
def nature_style(
    *,
    text_pt: Number = NATURE_DEFAULT_TEXT_PT,
    tick_pt: Number = NATURE_DEFAULT_TICK_PT,
    legend_pt: Number = NATURE_DEFAULT_LEGEND_PT,
    overrides: Optional[Dict[str, object]] = None,
):
    """Temporarily apply the central paper style inside a plotting block."""
    with mpl.rc_context():
        apply_nature_style(
            text_pt=text_pt,
            tick_pt=tick_pt,
            legend_pt=legend_pt,
            overrides=overrides,
        )
        yield


# ============================================================
# Colour palette and therapy-strategy colours
# ============================================================

PALETTE: Dict[str, str] = {
    "primary": "#1F77B4",
    "secondary": "#FF7F0E",
    "tertiary": "#2CA02C",
    "black": "#000000",
    "grey": "#7F7F7F",
    "cat1": "#1B9E77",
    "cat2": "#D95F02",
    "cat3": "#7570B3",
    "cat4": "#E7298A",
}


def therapy_color(name: str) -> str:
    """Return the canonical colour for a therapy strategy name or alias."""
    validate_strategies([name])
    return get_strategy_color(name)


MARKERS: Sequence[str] = ["o", "s", "D", "^", "v", "<", ">", "P", "X"]
LINESTYLES: Sequence[str] = ["-", "--", "-.", ":"]


# ============================================================
# Axes-box layout helpers
# ============================================================

def set_transparent_background(fig: mpl.figure.Figure) -> None:
    """Force transparent figure and axes backgrounds."""
    fig.patch.set_alpha(0.0)
    fig.patch.set_facecolor("none")
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
        ax.set_facecolor("none")


def set_axes_size_mm(
    ax: plt.Axes,
    width_mm: Number,
    height_mm: Number,
    *,
    left_mm: Number = DEFAULT_MARGIN_LEFT_MM,
    right_mm: Number = DEFAULT_MARGIN_RIGHT_MM,
    bottom_mm: Number = DEFAULT_MARGIN_BOTTOM_MM,
    top_mm: Number = DEFAULT_MARGIN_TOP_MM,
) -> mpl.figure.Figure:
    """
    Set one axes box to an exact physical width/height in millimetres.

    Width and height apply to the data axes rectangle only. The surrounding
    margins reserve room for text or outside annotations.
    """
    width = float(width_mm)
    height = float(height_mm)
    left = float(left_mm)
    right = float(right_mm)
    bottom = float(bottom_mm)
    top = float(top_mm)

    total_w = left + width + right
    total_h = bottom + height + top

    fig = ax.figure
    fig.set_size_inches(mm_to_inches(total_w), mm_to_inches(total_h), forward=True)
    ax.set_position(
        [
            left / total_w,
            bottom / total_h,
            width / total_w,
            height / total_h,
        ]
    )
    set_transparent_background(fig)
    return fig


def set_axes_box(ax: plt.Axes, spec: AxesBoxSpec) -> mpl.figure.Figure:
    """Apply an AxesBoxSpec to a single axes."""
    return set_axes_size_mm(
        ax,
        spec.width_mm,
        spec.height_mm,
        left_mm=spec.left_mm,
        right_mm=spec.right_mm,
        bottom_mm=spec.bottom_mm,
        top_mm=spec.top_mm,
    )


def _axes_from_grid(axes, nrows: int, ncols: int) -> list[list[plt.Axes]]:
    if isinstance(axes, plt.Axes):
        return [[axes]]

    if hasattr(axes, "reshape"):
        reshaped = axes.reshape(nrows, ncols)
        return [[reshaped[r, c] for c in range(ncols)] for r in range(nrows)]

    flat = list(axes)
    if len(flat) != nrows * ncols:
        raise ValueError("Could not reshape axes into the requested grid.")
    return [flat[r * ncols : (r + 1) * ncols] for r in range(nrows)]


def set_axes_grid_size_mm(
    axes,
    *,
    nrows: int,
    ncols: int,
    ax_width_mm: Number,
    ax_height_mm: Number,
    left_mm: Number = DEFAULT_MARGIN_LEFT_MM,
    right_mm: Number = DEFAULT_MARGIN_RIGHT_MM,
    bottom_mm: Number = DEFAULT_MARGIN_BOTTOM_MM,
    top_mm: Number = DEFAULT_MARGIN_TOP_MM,
    wspace_mm: Number = DEFAULT_PANEL_GAP_MM,
    hspace_mm: Number = DEFAULT_PANEL_GAP_MM,
) -> mpl.figure.Figure:
    """
    Set every axes box in a regular grid to an exact physical size.

    `ax_width_mm` and `ax_height_mm` refer to each panel's axes rectangle.
    """
    grid = _axes_from_grid(axes, nrows=nrows, ncols=ncols)
    width = float(ax_width_mm)
    height = float(ax_height_mm)
    left = float(left_mm)
    right = float(right_mm)
    bottom = float(bottom_mm)
    top = float(top_mm)
    wspace = float(wspace_mm)
    hspace = float(hspace_mm)

    total_w = left + ncols * width + (ncols - 1) * wspace + right
    total_h = bottom + nrows * height + (nrows - 1) * hspace + top
    fig = grid[0][0].figure
    fig.set_size_inches(mm_to_inches(total_w), mm_to_inches(total_h), forward=True)

    for row in range(nrows):
        for col in range(ncols):
            x_mm = left + col * (width + wspace)
            # Matplotlib positions count from the bottom; row 0 is visually top.
            y_mm = bottom + (nrows - 1 - row) * (height + hspace)
            grid[row][col].set_position(
                [
                    x_mm / total_w,
                    y_mm / total_h,
                    width / total_w,
                    height / total_h,
                ]
            )

    set_transparent_background(fig)
    return fig


def set_axes_grid_variable_mm(
    axes,
    *,
    nrows: int,
    ncols: int,
    ax_widths_mm: Sequence[Number],
    ax_heights_mm: Sequence[Number],
    left_mm: Number = DEFAULT_MARGIN_LEFT_MM,
    right_mm: Number = DEFAULT_MARGIN_RIGHT_MM,
    bottom_mm: Number = DEFAULT_MARGIN_BOTTOM_MM,
    top_mm: Number = DEFAULT_MARGIN_TOP_MM,
    wspace_mm: Number = DEFAULT_PANEL_GAP_MM,
    hspace_mm: Number = DEFAULT_PANEL_GAP_MM,
) -> mpl.figure.Figure:
    """
    Set a regular axes grid with per-column widths and per-row heights.

    Width and height values still refer to axes boxes only. `ax_widths_mm`
    has one value per column; `ax_heights_mm` has one value per row.
    """
    if len(ax_widths_mm) != ncols:
        raise ValueError("ax_widths_mm must contain one width per column.")
    if len(ax_heights_mm) != nrows:
        raise ValueError("ax_heights_mm must contain one height per row.")

    grid = _axes_from_grid(axes, nrows=nrows, ncols=ncols)
    widths = [float(v) for v in ax_widths_mm]
    heights = [float(v) for v in ax_heights_mm]
    left = float(left_mm)
    right = float(right_mm)
    bottom = float(bottom_mm)
    top = float(top_mm)
    wspace = float(wspace_mm)
    hspace = float(hspace_mm)

    total_w = left + sum(widths) + (ncols - 1) * wspace + right
    total_h = bottom + sum(heights) + (nrows - 1) * hspace + top
    fig = grid[0][0].figure
    fig.set_size_inches(mm_to_inches(total_w), mm_to_inches(total_h), forward=True)

    row_tops = []
    running_top = total_h - top
    for height in heights:
        row_tops.append(running_top)
        running_top -= height + hspace

    for row in range(nrows):
        y_mm = row_tops[row] - heights[row]
        x_mm = left
        for col in range(ncols):
            grid[row][col].set_position(
                [
                    x_mm / total_w,
                    y_mm / total_h,
                    widths[col] / total_w,
                    heights[row] / total_h,
                ]
            )
            x_mm += widths[col] + wspace

    set_transparent_background(fig)
    return fig


def format_axes(
    ax: plt.Axes,
    spines: Iterable[str] = ("top", "right"),
    tick_params: Optional[dict] = None,
) -> None:
    """Apply unified spine and tick styling to one axes."""
    for spine in spines:
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)

    spine_width = float(mpl.rcParams.get("axes.linewidth", 0.5))
    tick_width = float(mpl.rcParams.get("xtick.major.width", spine_width))
    tick_length = float(mpl.rcParams.get("xtick.major.size", 2.0))

    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_linewidth(spine_width)
            spine.set_color("0.1")

    params = dict(
        direction="out",
        length=tick_length,
        width=tick_width,
        pad=2,
        top=False,
        right=False,
    )
    if tick_params is not None:
        params.update(tick_params)
    ax.tick_params(axis="both", **params)


def format_all_axes(fig: mpl.figure.Figure, **format_kwargs) -> None:
    """Apply `format_axes` to all axes in a figure."""
    for ax in fig.axes:
        format_axes(ax, **format_kwargs)


def apply_small_margins(ax: plt.Axes, x: float = 0.02, y: float = 0.05) -> None:
    """Add small relative data margins to axes."""
    ax.margins(x=x, y=y)


# ============================================================
# Constructors
# ============================================================

def new_figure(
    width: SizeValue = "nature_single",
    height_mm: Number = 55.0,
    *,
    margins: Optional[Tuple[Number, Number, Number, Number]] = None,
    dpi: Number = NATURE_DEFAULT_DPI,
    apply_style: bool = True,
) -> Tuple[mpl.figure.Figure, plt.Axes]:
    """
    Create a single-axes figure with exact axes-box size.

    `width` and `height_mm` refer to the axes box, not the full canvas.
    `margins` is `(left_mm, right_mm, bottom_mm, top_mm)`.
    """
    if apply_style:
        apply_nature_style()
    left, right, bottom, top = margins or (
        DEFAULT_MARGIN_LEFT_MM,
        DEFAULT_MARGIN_RIGHT_MM,
        DEFAULT_MARGIN_BOTTOM_MM,
        DEFAULT_MARGIN_TOP_MM,
    )
    fig, ax = plt.subplots(dpi=float(dpi))
    set_axes_size_mm(
        ax,
        resolve_width_mm(width),
        height_mm,
        left_mm=left,
        right_mm=right,
        bottom_mm=bottom,
        top_mm=top,
    )
    format_axes(ax)
    return fig, ax


def new_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    ax_width_mm: Number = 42.0,
    ax_height_mm: Number = 32.0,
    margins: Optional[Tuple[Number, Number, Number, Number]] = None,
    wspace_mm: Number = DEFAULT_PANEL_GAP_MM,
    hspace_mm: Number = DEFAULT_PANEL_GAP_MM,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    dpi: Number = NATURE_DEFAULT_DPI,
    apply_style: bool = True,
    **subplot_kwargs,
) -> Tuple[mpl.figure.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    """
    Create a subplot grid with exact axes-box size for each panel.

    `ax_width_mm` and `ax_height_mm` refer to each axes box, not the full
    canvas or the total grid.
    """
    if apply_style:
        apply_nature_style()
    left, right, bottom, top = margins or (
        DEFAULT_MARGIN_LEFT_MM,
        DEFAULT_MARGIN_RIGHT_MM,
        DEFAULT_MARGIN_BOTTOM_MM,
        DEFAULT_MARGIN_TOP_MM,
    )
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        dpi=float(dpi),
        **subplot_kwargs,
    )
    set_axes_grid_size_mm(
        axes,
        nrows=nrows,
        ncols=ncols,
        ax_width_mm=ax_width_mm,
        ax_height_mm=ax_height_mm,
        left_mm=left,
        right_mm=right,
        bottom_mm=bottom,
        top_mm=top,
        wspace_mm=wspace_mm,
        hspace_mm=hspace_mm,
    )
    for ax in fig.axes:
        format_axes(ax)
    return fig, axes


def new_subplots_variable(
    nrows: int,
    ncols: int,
    *,
    ax_widths_mm: Sequence[Number],
    ax_heights_mm: Sequence[Number],
    margins: Optional[Tuple[Number, Number, Number, Number]] = None,
    wspace_mm: Number = DEFAULT_PANEL_GAP_MM,
    hspace_mm: Number = DEFAULT_PANEL_GAP_MM,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    dpi: Number = NATURE_DEFAULT_DPI,
    apply_style: bool = True,
    **subplot_kwargs,
) -> Tuple[mpl.figure.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    """Create a subplot grid with per-column/per-row axes-box sizes."""
    if apply_style:
        apply_nature_style()
    left, right, bottom, top = margins or (
        DEFAULT_MARGIN_LEFT_MM,
        DEFAULT_MARGIN_RIGHT_MM,
        DEFAULT_MARGIN_BOTTOM_MM,
        DEFAULT_MARGIN_TOP_MM,
    )
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        dpi=float(dpi),
        **subplot_kwargs,
    )
    set_axes_grid_variable_mm(
        axes,
        nrows=nrows,
        ncols=ncols,
        ax_widths_mm=ax_widths_mm,
        ax_heights_mm=ax_heights_mm,
        left_mm=left,
        right_mm=right,
        bottom_mm=bottom,
        top_mm=top,
        wspace_mm=wspace_mm,
        hspace_mm=hspace_mm,
    )
    for ax in fig.axes:
        format_axes(ax)
    return fig, axes


# Backward-compatible canvas-size helper retained for old scripts. Prefer
# set_axes_size_mm/new_figure/new_subplots for publication figures.
def set_size(
    ax: plt.Axes,
    width: SizeValue = "thesis",
    ratio: float = 0.62,
) -> mpl.figure.Figure:
    """Set full canvas size from a width preset and height ratio."""
    w_in = mm_to_inches(resolve_width_mm(width))
    h_in = w_in * float(ratio)
    fig = ax.figure
    fig.set_size_inches(w_in, h_in)
    set_transparent_background(fig)
    return fig


# ============================================================
# Saving and validation
# ============================================================

def savefig(
    fig: mpl.figure.Figure,
    path: Union[str, Path],
    *,
    dpi: Optional[int] = None,
    transparent: bool = True,
    bbox_inches=None,
    pad_inches: float = 0.02,
    warn_on_bitmap: bool = True,
    **savefig_kwargs,
) -> None:
    """
    Save a figure with central export settings.

    For line-art plots, prefer PDF/EPS/SVG. Bitmap outputs are allowed for
    previews or raster content, but `warn_on_bitmap=True` prints a reminder.
    """
    out_path = Path(path)
    # ``figure_plots`` is reserved for manuscript-ready PDFs.  A plotting
    # script may still request a preview SVG/PNG, but its final-export path is
    # normalised to PDF so non-PDF graphics cannot accumulate there.
    try:
        out_path.relative_to(PAPER_FIGURE_ROOT)
    except ValueError:
        pass
    else:
        out_path = out_path.with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if warn_on_bitmap and ext in BITMAP_EXTENSIONS:
        print(
            f"[figtools] Warning: {out_path.name} is a bitmap export. "
            "Use PDF/EPS/SVG for final graph or line-art submission."
        )

    set_transparent_background(fig)
    fig.savefig(
        out_path,
        dpi=NATURE_DEFAULT_DPI if dpi is None else dpi,
        transparent=transparent,
        facecolor="none",
        edgecolor="none",
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        **savefig_kwargs,
    )


def audit_text_sizes(
    fig: mpl.figure.Figure,
    *,
    min_pt: float = NATURE_TEXT_MIN_PT,
    max_pt: float = NATURE_TEXT_MAX_PT,
) -> list[tuple[str, float]]:
    """
    Return visible text artists outside the recommended point-size range.

    This is intentionally non-fatal so scripts can print/report it during
    migration without breaking long analysis runs.
    """
    offenders: list[tuple[str, float]] = []
    for text in fig.findobj(match=mpl.text.Text):
        if not text.get_visible():
            continue
        size = float(text.get_fontsize())
        if size == 0:
            continue
        if size < min_pt or size > max_pt:
            label = text.get_text() or repr(text)
            offenders.append((label, size))
    return offenders


def axes_size_mm(ax: plt.Axes) -> tuple[float, float]:
    """Return the current physical axes-box size in millimetres."""
    fig = ax.figure
    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    return inches_to_mm(bbox.width * fig_w), inches_to_mm(bbox.height * fig_h)
