#!/usr/bin/env python3
"""
Interactive line-profile GUI for Epanechnikov resistant-density-change maps.

Select a plate, well, time interval, and change-map type, then click two points
on the image to define a radius/line direction. The profile averages values
along that line across an adjustable perpendicular width and can be saved as
CSV or PNG.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import tkinter as tk
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.image import imread as mpl_imread
from matplotlib.widgets import EllipseSelector, PolygonSelector, RectangleSelector, TextBox
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.ndimage import map_coordinates
import tifffile as tiff

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(PROJECT_ROOT))

from analysis.basic.density_maps.Density_Maps_Epanechnikov_coarsegrained import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    PIXEL_SIZE_UM,
    load_config,
    output_type_name,
    pointcloud_filenames,
    pointcloud_read,
    resolve_coarsening_factor,
    resolve_experiment_path,
    resolve_radius_um,
    well_identifier_from_pointcloud_filename,
)


DEFAULT_RUN_DIR = Path(
    r"D:\2025_08_CoreSeed_Large_Testdataset\output\run_20260630_145305_320_32"
)
IMAGE_CACHE_LIMIT = 96
INTERVAL_SLIDER_DEBOUNCE_MS = 60
MAX_AGGREGATED_SCATTER_POINTS = 120_000
POINTCLOUD_SUFFIX_PRIORITY = (
    "_simulation.h5",
    "_total_poisson_padded.h5",
    "_total_poisson.h5",
    "_total.h5",
)
KERNEL_SIZES_UM = (40, 80, 160, 320)

CHANGE_FILE_RE = re.compile(
    r"^(?P<well>.+)_(?P<first_tp>[^_]+)_to_(?P<next_tp>[^_]+)_"
    r"(?P<kind>res_density_epanechnikov_(?:abs|rel)_diff|"
    r"res_zero_initial_growth_density_epanechnikov_abs_diff)\.tif$"
)

CHANGE_TYPE_LABELS = {
    "res_density_epanechnikov_abs_diff": "absolute resistant density change",
    "res_density_epanechnikov_rel_diff": "relative resistant growth",
    "res_zero_initial_growth_density_epanechnikov_abs_diff": "zero-initial resistant growth",
}

Y_AXIS_LABELS = {
    "res_density_epanechnikov_abs_diff": "Resistant density change [cells/µm²]",
    "res_density_epanechnikov_rel_diff": "Relative resistant growth [ln(next/prev)/day]",
    "res_zero_initial_growth_density_epanechnikov_abs_diff": (
        "Zero-initial resistant density change [cells/µm²]"
    ),
}

BACKGROUND_CHOICES = {
    "Selected change map": "change",
    "Total density at first day": "total",
    "Resistant density at first day": "res",
    "Sensitive density at first day": "sus",
}

REFERENCE_MAP_CHOICES = {
    "Total density": "total",
    "Resistant density": "res",
    "Sensitive density": "sus",
    "HDF5 cell positions": "cell_positions",
}

REFERENCE_TIME_CHOICES = {
    "First timepoint": "first",
    "Next timepoint": "next",
}

COMPOSITION_CMAP = LinearSegmentedColormap.from_list(
    "sensitive_blue_to_resistant_goldenrod",
    ["#4169E1", "#DAA520"],
)


def line_to_image_edge(
    image_shape: tuple[int, int],
    center_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> tuple[float, float]:
    """Return the image-border point reached from center toward target."""
    height, width = image_shape
    cx, cy = center_xy
    tx, ty = target_xy
    dx = tx - cx
    dy = ty - cy
    if math.hypot(dx, dy) < 1e-9:
        raise ValueError("Click away from the image center to define a direction.")

    candidates = []
    if dx > 0:
        candidates.append(((width - 1 - cx) / dx, width - 1, cy + ((width - 1 - cx) / dx) * dy))
    elif dx < 0:
        candidates.append(((0 - cx) / dx, 0, cy + ((0 - cx) / dx) * dy))
    if dy > 0:
        candidates.append(((height - 1 - cy) / dy, cx + ((height - 1 - cy) / dy) * dx, height - 1))
    elif dy < 0:
        candidates.append(((0 - cy) / dy, cx + ((0 - cy) / dy) * dx, 0))

    valid = [
        (t, x, y)
        for t, x, y in candidates
        if t > 0 and -0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5
    ]
    if not valid:
        raise ValueError("Could not extend the selected direction to the image edge.")
    _, x_edge, y_edge = min(valid, key=lambda item: item[0])
    return float(x_edge), float(y_edge)


@dataclass(frozen=True)
class ChangeEntry:
    first_tp: str
    next_tp: str
    kind: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.first_tp}->{self.next_tp}"


@dataclass
class ProfileResult:
    distance_px: np.ndarray
    distance_um: np.ndarray
    values: np.ndarray
    samples: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    width_px: int


def colored_profile_line(
    ax,
    distance_um: np.ndarray,
    values: np.ndarray,
    color_values: np.ndarray,
    *,
    cmap: str = "viridis",
    linewidth: float = 1.6,
    color_limits: tuple[float, float] | None = None,
) -> LineCollection:
    points = np.column_stack((distance_um, values))
    if len(points) < 2:
        collection = LineCollection([], cmap=cmap)
        ax.add_collection(collection)
        return collection

    segments = np.stack((points[:-1], points[1:]), axis=1)
    segment_colors = 0.5 * (color_values[:-1] + color_values[1:])
    finite_colors = segment_colors[np.isfinite(segment_colors)]
    if color_limits is not None:
        norm = Normalize(vmin=float(color_limits[0]), vmax=float(color_limits[1]))
        if norm.vmin == norm.vmax:
            norm.vmax = norm.vmin + 1.0
    elif finite_colors.size:
        norm = Normalize(vmin=float(np.nanmin(finite_colors)), vmax=float(np.nanmax(finite_colors)))
        if norm.vmin == norm.vmax:
            norm.vmax = norm.vmin + 1.0
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
    collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    collection.set_array(segment_colors)
    ax.add_collection(collection)
    return collection


def timepoint_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def read_float_image(path: Path) -> np.ndarray:
    data = tiff.imread(path).astype(np.float32, copy=False)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF image, got shape {data.shape}: {path}")
    return data


def robust_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def discover_run_plates(run_dir: Path) -> list[str]:
    return sorted(
        [path.name for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("Plate_")],
        key=lambda name: timepoint_key(name.split("_", 1)[-1]),
    )


def discover_wells(run_dir: Path, plate: str) -> list[str]:
    plate_dir = run_dir / plate
    if not plate_dir.exists():
        return []
    return sorted(
        path.name
        for path in plate_dir.iterdir()
        if path.is_dir() and path.name.lower() not in {"hdf5", "summary_plots"}
    )


def discover_all_wells(run_dir: Path) -> tuple[list[str], dict[str, str]]:
    well_to_plate: dict[str, str] = {}
    for plate in discover_run_plates(run_dir):
        for well in discover_wells(run_dir, plate):
            well_to_plate[well] = plate
    return sorted(well_to_plate, key=str.casefold), well_to_plate


def discover_changes(changes_dir: Path, well: str) -> list[ChangeEntry]:
    entries: list[ChangeEntry] = []
    if not changes_dir.exists():
        return entries
    for path in changes_dir.glob("*.tif"):
        match = CHANGE_FILE_RE.match(path.name)
        if not match or match.group("well") != well:
            continue
        entries.append(
            ChangeEntry(
                first_tp=match.group("first_tp"),
                next_tp=match.group("next_tp"),
                kind=match.group("kind"),
                path=path,
            )
        )
    return sorted(entries, key=lambda e: (timepoint_key(e.first_tp), timepoint_key(e.next_tp), e.kind))


def sample_line_profile(
    image: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    width_px: int,
    pixel_size_um: float,
) -> ProfileResult:
    x0, y0 = start_xy
    x1, y1 = end_xy
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length < 1:
        raise ValueError("Select two points at least one pixel apart.")

    n_steps = max(2, int(math.ceil(length)) + 1)
    line_x = np.linspace(x0, x1, n_steps)
    line_y = np.linspace(y0, y1, n_steps)

    # Unit perpendicular to the selected line, used for width averaging.
    nx = -dy / length
    ny = dx / length
    half_width = max(0, int(width_px) // 2)
    offsets = np.arange(-half_width, half_width + 1, dtype=float)

    sample_stack = []
    sample_x_stack = []
    sample_y_stack = []
    for offset in offsets:
        sample_x = line_x + offset * nx
        sample_y = line_y + offset * ny
        coords = np.vstack((sample_y, sample_x))
        sampled = map_coordinates(
            image,
            coords,
            order=1,
            mode="constant",
            cval=np.nan,
        )
        sample_stack.append(sampled)
        sample_x_stack.append(sample_x)
        sample_y_stack.append(sample_y)
    samples = np.vstack(sample_stack)
    sample_x = np.vstack(sample_x_stack)
    sample_y = np.vstack(sample_y_stack)
    values = np.nanmean(samples, axis=0)
    distance_px = np.linspace(0.0, length, n_steps)
    return ProfileResult(
        distance_px=distance_px,
        distance_um=distance_px * pixel_size_um,
        values=values,
        samples=samples,
        sample_x=sample_x,
        sample_y=sample_y,
        start_xy=start_xy,
        end_xy=end_xy,
        width_px=int(width_px),
    )


class DensityChangeProfileGUI:
    def __init__(self, root: tk.Tk, run_dir: Path, config_path: Path):
        self.root = root
        self.run_dir = run_dir
        self.config_path = config_path
        self.config = load_config(os.fspath(config_path))
        self.density_maps_name = output_type_name(self.config, "density_maps_output_type", "density_maps")
        self.density_changes_name = output_type_name(self.config, "density_changes_output_type", "density_changes")
        self.scatter_movie_name = output_type_name(
            self.config,
            "scatter_movie_output_type",
            "hdf5_cell_position_scatter",
        )
        self.coarsening_factor = resolve_coarsening_factor(
            self.config,
            resolve_radius_um(self.config),
        )
        self.pixel_size_um = PIXEL_SIZE_UM * self.coarsening_factor

        self.image: np.ndarray | None = None
        self.display_image: np.ndarray | None = None
        self.current_change: ChangeEntry | None = None
        self.current_profile: ProfileResult | None = None
        self.click_points: list[tuple[float, float]] = []
        self.line_artist = None
        self.width_artists: list = []
        self._drag_start_xy: tuple[float, float] | None = None
        self._drag_start_canvas_xy: tuple[float, float] | None = None
        self._drag_axis = None
        self._drag_canvas = None
        self._drag_endpoint_index: int | None = None
        self._drag_moved = False
        self._drag_preview_artists: list = []
        self.image_colorbar = None
        self.reference_map_colorbar = None
        self.profile_colorbar = None
        self.lut_limits: dict[tuple[str, str], tuple[float, float] | None] = {}
        self.lut_textboxes: list[TextBox] = []
        self.scatter_selectors: list[RectangleSelector | EllipseSelector | PolygonSelector] = []
        self.scatter_selection_data: dict[object, dict[str, np.ndarray]] = {}
        self.hdf5_position_artists: list[dict[str, object]] = []
        self.selected_scatter_points: np.ndarray | None = None
        self.selected_overlay_artists: list = []
        self.scatter_selection_bounds: tuple[float, float, float, float] | None = None
        self.scatter_selection_polygon: np.ndarray | None = None
        self.scatter_selection_panel: str | None = None
        self.scatter_selection_shape: str = "Rectangle"
        self.scatter_selection_artists: list = []
        self.map_roi: dict[str, object] | None = None
        self.map_roi_artists: list = []
        self.res_density_gate_overlay_artists: list = []
        self._roi_drag_start_xy: tuple[float, float] | None = None
        self._roi_drag_axis = None
        self._roi_drag_canvas = None
        self._roi_points: list[tuple[float, float]] = []
        self._roi_preview_artists: list = []
        self.image_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.raw_image_shape_cache: dict[Path, tuple[int, int]] = {}
        self.shared_map_limits_cache: dict[tuple[str, ...], tuple[float, float] | None] = {}
        self._is_simulation_run_cache: bool | None = None
        self._stored_map_xlim: tuple[float, float] | None = None
        self._stored_map_ylim: tuple[float, float] | None = None
        self._map_view_context: tuple[str, str, str, int, int] | None = None
        self._syncing_map_limits = False
        self._map_limit_sync_axes: set = set()
        self._syncing_interval_slider = False
        # Keep Scale.set() callbacks quiet until both gate widgets exist.
        self._syncing_res_density_gate = True
        self._pending_interval_load: str | None = None
        self._pending_kernel_switch: str | None = None
        self._playback_after_id: str | None = None
        self._plate_by_well: dict[str, str] = {}
        self.res_density_gate_slider_min = float(
            self.config.get("resistant_density_gate_slider_min", 0.0)
        )
        self.res_density_gate_slider_max = float(
            self.config.get("resistant_density_gate_slider_max", 0.002)
        )
        self.res_density_gate_step = float(
            self.config.get("resistant_density_gate_step", 1.0e-5)
        )
        if self.res_density_gate_step <= 0:
            self.res_density_gate_step = 1.0e-5
        if self.res_density_gate_slider_max <= self.res_density_gate_slider_min:
            self.res_density_gate_slider_max = self.res_density_gate_slider_min + self.res_density_gate_step
        default_gate_lower = float(
            self.config.get("resistant_density_gate_default_lower", self.res_density_gate_slider_min)
        )
        default_gate_upper = float(
            self.config.get("resistant_density_gate_default_upper", self.res_density_gate_slider_max)
        )
        default_gate_lower = min(
            max(default_gate_lower, self.res_density_gate_slider_min),
            self.res_density_gate_slider_max,
        )
        default_gate_upper = min(
            max(default_gate_upper, self.res_density_gate_slider_min),
            self.res_density_gate_slider_max,
        )
        if default_gate_lower > default_gate_upper:
            default_gate_lower, default_gate_upper = default_gate_upper, default_gate_lower

        self.plate_var = tk.StringVar()
        self.well_var = tk.StringVar()
        self.interval_var = tk.StringVar()
        self.change_type_var = tk.StringVar()
        self.background_var = tk.StringVar(value="Selected change map")
        self.reference_map_var = tk.StringVar(value="Total density")
        self.reference_time_var = tk.StringVar(value="First timepoint")
        self.width_var = tk.IntVar(value=5)
        self.interval_index_var = tk.IntVar(value=0)
        self.playback_fps_var = tk.DoubleVar(value=float(self.config.get("playback_fps", 2.0)))
        self.playback_button_text_var = tk.StringVar(value="Play")
        self.keep_line_var = tk.BooleanVar(value=True)
        self.center_to_edge_var = tk.BooleanVar(value=False)
        self.map_roi_enabled_var = tk.BooleanVar(value=False)
        self.map_roi_shape_var = tk.StringVar(value="Polygon")
        self.scatter_selection_shape_var = tk.StringVar(value="Rectangle")
        self.res_density_gate_enabled_var = tk.BooleanVar(value=False)
        self.res_density_gate_lower_var = tk.DoubleVar(value=default_gate_lower)
        self.res_density_gate_upper_var = tk.DoubleVar(value=default_gate_upper)
        initial_kernel = self.infer_kernel_size_um(self.run_dir)
        self.kernel_index_var = tk.IntVar(value=KERNEL_SIZES_UM.index(initial_kernel))
        self.kernel_label_var = tk.StringVar(value=f"Kernel: {initial_kernel} um")
        self.kernel_dir_vars: dict[int, tk.StringVar] = {}
        for kernel_um in KERNEL_SIZES_UM:
            candidate = self.run_dir.parent / (
                f"run_kernel_{kernel_um}_coarse_{kernel_um // 10}"
            )
            value = self.run_dir if kernel_um == initial_kernel else candidate
            self.kernel_dir_vars[kernel_um] = tk.StringVar(
                value=os.fspath(value) if value.exists() or kernel_um == initial_kernel else ""
            )
        self.status_var = tk.StringVar(value="Select a plate, well, interval, then click or drag a profile line.")
        self.run_dir_info_var = tk.StringVar(value=self.run_dir_info_text())

        self.root.title("Density Change Line Profile")
        self._configure_style()
        self._build_ui()
        self._populate_plates()

    def _configure_style(self) -> None:
        self.root.minsize(1280, 800)
        self.root.configure(background="#e9eef5")
        self.root.option_add("*Font", "{Segoe UI} 9")

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        colors = {
            "bg": "#e9eef5",
            "panel": "#ffffff",
            "sidebar": "#ffffff",
            "sidebar_alt": "#eef2f6",
            "sidebar_text": "#18212f",
            "sidebar_muted": "#64748b",
            "panel_alt": "#eef2f6",
            "text": "#18212f",
            "muted": "#64748b",
            "border": "#d8dee8",
            "accent": "#4169e1",
            "accent_hover": "#3458c8",
            "button": "#e8edf4",
            "button_hover": "#dce4ef",
        }

        style.configure(".", font=("Segoe UI", 9), foreground=colors["text"])
        style.configure("App.TFrame", background=colors["bg"])
        style.configure("Sidebar.TFrame", background=colors["sidebar"], relief="flat")
        style.configure("SidebarPanel.TFrame", background=colors["sidebar"])
        style.configure("Panel.TFrame", background=colors["panel"])
        style.configure("Figures.TFrame", background=colors["bg"])
        style.configure("CanvasPanel.TFrame", background=colors["panel"])
        style.configure("TLabel", background=colors["panel"], foreground=colors["text"])
        style.configure("Muted.TLabel", background=colors["panel"], foreground=colors["muted"])
        style.configure(
            "SidebarTitle.TLabel",
            background=colors["sidebar"],
            foreground=colors["sidebar_text"],
            font=("Segoe UI Semibold", 14),
        )
        style.configure(
            "SidebarSubtitle.TLabel",
            background=colors["sidebar"],
            foreground=colors["sidebar_muted"],
            font=("Segoe UI", 8),
        )
        style.configure(
            "SidebarField.TLabel",
            background=colors["sidebar"],
            foreground=colors["sidebar_muted"],
            font=("Segoe UI Semibold", 8),
        )
        style.configure(
            "SidebarMuted.TLabel",
            background=colors["sidebar"],
            foreground=colors["sidebar_muted"],
            font=("Segoe UI", 8),
        )
        style.configure(
            "Section.TLabel",
            background=colors["sidebar"],
            foreground=colors["accent"],
            font=("Segoe UI Semibold", 8),
        )
        style.configure(
            "Status.TLabel",
            background="#ffffff",
            foreground=colors["text"],
            padding=(12, 8),
            font=("Segoe UI", 9),
        )
        style.configure(
            "TButton",
            background=colors["button"],
            foreground=colors["text"],
            bordercolor=colors["border"],
            focusthickness=0,
            focuscolor=colors["button"],
            padding=(9, 6),
            relief="flat",
        )
        style.map(
            "TButton",
            background=[("active", colors["button_hover"]), ("pressed", "#cbd5e1")],
            relief=[("pressed", "flat"), ("!pressed", "flat")],
        )
        style.configure(
            "Primary.TButton",
            background=colors["accent"],
            foreground="#ffffff",
            bordercolor=colors["accent"],
            focusthickness=0,
            padding=(9, 6),
            relief="flat",
        )
        style.map(
            "Primary.TButton",
            background=[("active", colors["accent_hover"]), ("pressed", "#2748a7")],
            foreground=[("disabled", "#dbe4ff"), ("!disabled", "#ffffff")],
        )
        style.configure("Icon.TButton", padding=(4, 5), width=3)
        style.configure(
            "TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground=colors["text"],
            bordercolor=colors["border"],
            arrowcolor=colors["muted"],
            padding=(6, 4),
        )
        style.map("TCombobox", fieldbackground=[("readonly", "#ffffff")])
        style.configure("TCheckbutton", background=colors["panel"], foreground=colors["text"])
        style.configure("Sidebar.TCheckbutton", background=colors["sidebar"], foreground=colors["sidebar_text"])
        style.map(
            "Sidebar.TCheckbutton",
            background=[("active", colors["sidebar"])],
            foreground=[("active", colors["sidebar_text"])],
        )
        style.configure("Horizontal.TScale", background=colors["panel"], troughcolor=colors["panel_alt"])
        style.configure(
            "Sidebar.Horizontal.TScale",
            background=colors["sidebar"],
            troughcolor=colors["sidebar_alt"],
        )

    def _style_toolbar(self, toolbar: NavigationToolbar2Tk) -> None:
        try:
            toolbar.configure(background="#e9eef5", borderwidth=0)
        except tk.TclError:
            pass
        for child in toolbar.winfo_children():
            try:
                child.configure(background="#e9eef5", relief="flat", borderwidth=0)
            except tk.TclError:
                pass

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10, style="App.TFrame")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls_shell = ttk.Frame(main, style="Sidebar.TFrame")
        controls_shell.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        controls_shell.rowconfigure(0, weight=1)
        controls_shell.columnconfigure(0, weight=1)

        controls_canvas = tk.Canvas(
            controls_shell,
            width=250,
            background="#ffffff",
            borderwidth=0,
            highlightthickness=0,
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_shell,
            orient="vertical",
            command=controls_canvas.yview,
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.grid(row=0, column=0, sticky="ns")
        controls_scrollbar.grid(row=0, column=1, sticky="ns")

        controls = ttk.Frame(controls_canvas, padding=(12, 12), style="Sidebar.TFrame")
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.columnconfigure(0, weight=1)

        def _resize_sidebar_scroll_region(_event=None) -> None:
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def _resize_sidebar_window(event) -> None:
            controls_canvas.itemconfigure(controls_window, width=event.width)

        def _scroll_sidebar(event) -> None:
            if event.delta:
                controls_canvas.yview_scroll(-int(event.delta / 120), "units")

        def _bind_sidebar_wheel(_event) -> None:
            controls_canvas.bind_all("<MouseWheel>", _scroll_sidebar)

        def _unbind_sidebar_wheel(_event) -> None:
            controls_canvas.unbind_all("<MouseWheel>")

        controls.bind("<Configure>", _resize_sidebar_scroll_region)
        controls_canvas.bind("<Configure>", _resize_sidebar_window)
        controls_canvas.bind("<Enter>", _bind_sidebar_wheel)
        controls_canvas.bind("<Leave>", _unbind_sidebar_wheel)
        controls.bind("<Enter>", _bind_sidebar_wheel)
        controls.bind("<Leave>", _unbind_sidebar_wheel)

        ttk.Label(controls, text="Density Profile", style="SidebarTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(controls, text="Line-based spatial analysis", style="SidebarSubtitle.TLabel").grid(
            row=1, column=0, sticky="w", pady=(0, 14)
        )
        ttk.Label(controls, text="SELECTION", style="Section.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 6)
        )

        ttk.Label(controls, text="Plate", style="SidebarField.TLabel").grid(row=3, column=0, sticky="w")
        self.plate_box = ttk.Combobox(controls, textvariable=self.plate_var, state="readonly", width=24)
        self.plate_box.grid(row=4, column=0, sticky="ew", pady=(0, 7))
        self.plate_box.bind("<<ComboboxSelected>>", lambda _event: self._on_plate_changed())

        ttk.Label(controls, text="Well", style="SidebarField.TLabel").grid(row=5, column=0, sticky="w")
        self.well_box = ttk.Combobox(controls, textvariable=self.well_var, state="readonly", width=24)
        self.well_box.grid(row=6, column=0, sticky="ew", pady=(0, 7))
        self.well_box.bind("<<ComboboxSelected>>", lambda _event: self._on_well_changed())

        ttk.Label(controls, text="Change Type", style="SidebarField.TLabel").grid(row=7, column=0, sticky="w")
        self.change_type_box = ttk.Combobox(
            controls,
            textvariable=self.change_type_var,
            state="readonly",
            width=24,
        )
        self.change_type_box.grid(row=8, column=0, sticky="ew", pady=(0, 7))
        self.change_type_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_intervals())

        ttk.Label(controls, text="Interval", style="SidebarField.TLabel").grid(row=9, column=0, sticky="w")
        self.interval_box = ttk.Combobox(controls, textvariable=self.interval_var, state="readonly", width=24)
        self.interval_box.grid(row=10, column=0, sticky="ew", pady=(0, 6))
        self.interval_box.bind("<<ComboboxSelected>>", lambda _event: self._on_interval_dropdown())

        interval_nav = ttk.Frame(controls, style="SidebarPanel.TFrame")
        interval_nav.grid(row=11, column=0, sticky="ew")
        interval_nav.columnconfigure(1, weight=1)
        ttk.Button(
            interval_nav,
            text="‹",
            width=3,
            style="Icon.TButton",
            command=lambda: self._step_interval(-1),
        ).grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.interval_slider = ttk.Scale(
            interval_nav,
            from_=0,
            to=0,
            orient="horizontal",
            style="Sidebar.Horizontal.TScale",
            command=self._on_interval_slider,
        )
        self.interval_slider.grid(row=0, column=1, sticky="ew")
        ttk.Button(
            interval_nav,
            text="›",
            width=3,
            style="Icon.TButton",
            command=lambda: self._step_interval(1),
        ).grid(row=0, column=2, sticky="e", padx=(4, 0))
        ttk.Button(
            interval_nav,
            textvariable=self.playback_button_text_var,
            width=5,
            command=self.toggle_interval_playback,
        ).grid(row=0, column=3, sticky="e", padx=(6, 0))
        ttk.Label(interval_nav, text="FPS", style="SidebarMuted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self.playback_fps_spinbox = ttk.Spinbox(
            interval_nav,
            from_=0.2,
            to=60.0,
            increment=0.5,
            width=6,
            textvariable=self.playback_fps_var,
        )
        self.playback_fps_spinbox.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0))
        self.interval_slider_label = ttk.Label(controls, text="Interval 0/0", style="SidebarMuted.TLabel")
        self.interval_slider_label.grid(row=12, column=0, sticky="w", pady=(0, 12))

        ttk.Label(controls, text="MAPS", style="Section.TLabel").grid(
            row=13, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(controls, text="Background", style="SidebarField.TLabel").grid(row=14, column=0, sticky="w")
        self.background_box = ttk.Combobox(
            controls,
            textvariable=self.background_var,
            state="readonly",
            values=list(BACKGROUND_CHOICES),
            width=24,
        )
        self.background_box.grid(row=15, column=0, sticky="ew", pady=(0, 7))
        self.background_box.bind("<<ComboboxSelected>>", lambda _event: self.load_image())

        ttk.Label(controls, text="Extra Map", style="SidebarField.TLabel").grid(row=16, column=0, sticky="w")
        self.reference_map_box = ttk.Combobox(
            controls,
            textvariable=self.reference_map_var,
            state="readonly",
            values=list(REFERENCE_MAP_CHOICES),
            width=24,
        )
        self.reference_map_box.grid(row=17, column=0, sticky="ew", pady=(0, 7))
        self.reference_map_box.bind("<<ComboboxSelected>>", lambda _event: self.update_reference_map())

        ttk.Label(controls, text="Extra Map Time", style="SidebarField.TLabel").grid(row=18, column=0, sticky="w")
        self.reference_time_box = ttk.Combobox(
            controls,
            textvariable=self.reference_time_var,
            state="readonly",
            values=list(REFERENCE_TIME_CHOICES),
            width=24,
        )
        self.reference_time_box.grid(row=19, column=0, sticky="ew", pady=(0, 12))
        self.reference_time_box.bind("<<ComboboxSelected>>", lambda _event: self.update_reference_map())

        ttk.Label(controls, text="LINE", style="Section.TLabel").grid(
            row=20, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(controls, text="Profile width (pixels)", style="SidebarField.TLabel").grid(row=21, column=0, sticky="w")
        self.width_slider = ttk.Scale(
            controls,
            from_=1,
            to=101,
            orient="horizontal",
            style="Sidebar.Horizontal.TScale",
            command=self._on_width_slider,
        )
        self.width_slider.set(self.width_var.get())
        self.width_slider.grid(row=22, column=0, sticky="ew")
        self.width_label = ttk.Label(controls, text=f"{self.width_var.get()} px", style="SidebarMuted.TLabel")
        self.width_label.grid(row=23, column=0, sticky="w", pady=(0, 8))

        ttk.Checkbutton(
            controls,
            text="Keep line across intervals",
            variable=self.keep_line_var,
            style="Sidebar.TCheckbutton",
        ).grid(row=24, column=0, sticky="w", pady=(0, 2))
        ttk.Checkbutton(
            controls,
            text="Center to edge mode",
            variable=self.center_to_edge_var,
            command=self._on_line_mode_changed,
            style="Sidebar.TCheckbutton",
        ).grid(row=25, column=0, sticky="w", pady=(0, 8))

        line_button_frame = ttk.Frame(controls, style="SidebarPanel.TFrame")
        line_button_frame.grid(row=26, column=0, sticky="ew", pady=2)
        line_button_frame.columnconfigure(0, weight=1)
        line_button_frame.columnconfigure(1, weight=1)
        ttk.Button(line_button_frame, text="Clear Line", command=self.clear_line).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(line_button_frame, text="Load Line", command=self.load_line_metadata).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )
        ttk.Label(controls, text="SPATIAL ROI", style="Section.TLabel").grid(
            row=27, column=0, sticky="w", pady=(10, 6)
        )
        ttk.Checkbutton(
            controls,
            text="Draw ROI on maps",
            variable=self.map_roi_enabled_var,
            command=self._on_map_roi_mode_changed,
            style="Sidebar.TCheckbutton",
        ).grid(row=28, column=0, sticky="w", pady=(0, 4))
        roi_frame = ttk.Frame(controls, style="SidebarPanel.TFrame")
        roi_frame.grid(row=29, column=0, sticky="ew", pady=(0, 8))
        roi_frame.columnconfigure(0, weight=1)
        roi_frame.columnconfigure(1, weight=1)
        ttk.Label(roi_frame, text="Map ROI", style="SidebarMuted.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 3)
        )
        ttk.Label(roi_frame, text="Scatter Gate", style="SidebarMuted.TLabel").grid(
            row=0, column=1, sticky="w", padx=(3, 0)
        )
        self.map_roi_shape_box = ttk.Combobox(
            roi_frame,
            textvariable=self.map_roi_shape_var,
            state="readonly",
            values=["Polygon", "Circle"],
            width=10,
        )
        self.map_roi_shape_box.grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=(0, 4))
        self.scatter_selection_shape_box = ttk.Combobox(
            roi_frame,
            textvariable=self.scatter_selection_shape_var,
            state="readonly",
            values=["Rectangle", "Ellipse", "Polygon"],
            width=10,
        )
        self.scatter_selection_shape_box.grid(row=1, column=1, sticky="ew", padx=(3, 0), pady=(0, 4))
        self.scatter_selection_shape_box.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._on_scatter_selection_shape_changed(),
        )
        ttk.Button(roi_frame, text="Clear ROI", command=self.clear_map_roi).grid(
            row=2, column=0, columnspan=2, sticky="ew"
        )
        ttk.Label(controls, text="RESISTANT DENSITY GATE", style="Section.TLabel").grid(
            row=30, column=0, sticky="w", pady=(10, 6)
        )
        gate_frame = ttk.Frame(controls, style="SidebarPanel.TFrame")
        gate_frame.grid(row=31, column=0, sticky="ew", pady=(0, 8))
        gate_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            gate_frame,
            text="Use resistant density gate",
            variable=self.res_density_gate_enabled_var,
            command=self._on_res_density_gate_changed,
            style="Sidebar.TCheckbutton",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(gate_frame, text="Lower", style="SidebarMuted.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w"
        )
        ttk.Button(
            gate_frame,
            text="<",
            width=3,
            style="Icon.TButton",
            command=lambda: self._step_res_density_gate("lower", -1),
        ).grid(row=2, column=0, sticky="w", padx=(0, 4))
        self.res_density_gate_lower_slider = ttk.Scale(
            gate_frame,
            from_=self.res_density_gate_slider_min,
            to=self.res_density_gate_slider_max,
            orient="horizontal",
            style="Sidebar.Horizontal.TScale",
            command=lambda value: self._on_res_density_gate_slider("lower", value),
        )
        self.res_density_gate_lower_slider.set(self.res_density_gate_lower_var.get())
        self.res_density_gate_lower_slider.grid(row=2, column=1, sticky="ew")
        ttk.Button(
            gate_frame,
            text=">",
            width=3,
            style="Icon.TButton",
            command=lambda: self._step_res_density_gate("lower", 1),
        ).grid(row=2, column=2, sticky="e", padx=(4, 0))
        self.res_density_gate_lower_label = ttk.Label(
            gate_frame,
            text="",
            style="SidebarMuted.TLabel",
        )
        self.res_density_gate_lower_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(gate_frame, text="Upper", style="SidebarMuted.TLabel").grid(
            row=4, column=0, columnspan=3, sticky="w"
        )
        ttk.Button(
            gate_frame,
            text="<",
            width=3,
            style="Icon.TButton",
            command=lambda: self._step_res_density_gate("upper", -1),
        ).grid(row=5, column=0, sticky="w", padx=(0, 4))
        self.res_density_gate_upper_slider = ttk.Scale(
            gate_frame,
            from_=self.res_density_gate_slider_min,
            to=self.res_density_gate_slider_max,
            orient="horizontal",
            style="Sidebar.Horizontal.TScale",
            command=lambda value: self._on_res_density_gate_slider("upper", value),
        )
        self.res_density_gate_upper_slider.set(self.res_density_gate_upper_var.get())
        self.res_density_gate_upper_slider.grid(row=5, column=1, sticky="ew")
        ttk.Button(
            gate_frame,
            text=">",
            width=3,
            style="Icon.TButton",
            command=lambda: self._step_res_density_gate("upper", 1),
        ).grid(row=5, column=2, sticky="e", padx=(4, 0))
        self.res_density_gate_upper_label = ttk.Label(
            gate_frame,
            text="",
            style="SidebarMuted.TLabel",
        )
        self.res_density_gate_upper_label.grid(row=6, column=0, columnspan=3, sticky="w")
        self._update_res_density_gate_labels()
        self._syncing_res_density_gate = False
        ttk.Label(controls, text="EXPORT", style="Section.TLabel").grid(
            row=32, column=0, sticky="w", pady=(10, 6)
        )
        save_frame = ttk.Frame(controls, style="SidebarPanel.TFrame")
        save_frame.grid(row=33, column=0, sticky="ew", pady=(2, 4))
        save_frame.columnconfigure(0, weight=1)
        save_frame.columnconfigure(1, weight=1)
        ttk.Label(save_frame, text="This Timepoint", style="SidebarMuted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 3))
        ttk.Label(save_frame, text="All Timepoints", style="SidebarMuted.TLabel").grid(row=0, column=1, sticky="w", padx=(3, 0))
        ttk.Button(save_frame, text="Profile CSV", command=self.save_csv, style="Primary.TButton").grid(
            row=1, column=0, sticky="ew", padx=(0, 3), pady=2
        )
        ttk.Button(save_frame, text="Profile CSV", command=self.save_all_timepoints_csv, style="Primary.TButton").grid(
            row=1, column=1, sticky="ew", padx=(3, 0), pady=2
        )
        ttk.Button(save_frame, text="Profile Plot", command=self.save_plot, style="Primary.TButton").grid(
            row=2, column=0, sticky="ew", padx=(0, 3), pady=2
        )
        ttk.Button(save_frame, text="Profile Plots", command=self.save_all_timepoints_plots, style="Primary.TButton").grid(
            row=2, column=1, sticky="ew", padx=(3, 0), pady=2
        )
        ttk.Button(save_frame, text="Line Scatter", command=self.save_line_scatterplot, style="Primary.TButton").grid(
            row=3, column=0, sticky="ew", padx=(0, 3), pady=2
        )
        ttk.Button(save_frame, text="Line Scatters", command=self.save_all_line_scatterplots, style="Primary.TButton").grid(
            row=3, column=1, sticky="ew", padx=(3, 0), pady=2
        )
        run_button_frame = ttk.Frame(controls, style="SidebarPanel.TFrame")
        run_button_frame.grid(row=34, column=0, sticky="ew", pady=(8, 2))
        run_button_frame.columnconfigure(0, weight=1)
        run_button_frame.columnconfigure(1, weight=1)
        ttk.Button(run_button_frame, text="Change Run Dir", command=self.choose_run_dir).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(run_button_frame, text="Reload Image", command=self.load_image).grid(
            row=0, column=1, sticky="ew", padx=(3, 0)
        )
        ttk.Button(run_button_frame, text="Fit Image", command=self.fit_maps_to_image_extent).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )

        ttk.Label(
            controls,
            textvariable=self.run_dir_info_var,
            wraplength=210,
            style="SidebarMuted.TLabel",
        ).grid(row=35, column=0, sticky="ew", pady=(12, 0))

        ttk.Label(controls, text="KERNEL RUN DIRECTORIES", style="Section.TLabel").grid(
            row=36, column=0, sticky="w", pady=(12, 6)
        )
        kernel_dirs_frame = ttk.Frame(controls, style="SidebarPanel.TFrame")
        kernel_dirs_frame.grid(row=37, column=0, sticky="ew")
        kernel_dirs_frame.columnconfigure(1, weight=1)
        for row, kernel_um in enumerate(KERNEL_SIZES_UM):
            ttk.Label(
                kernel_dirs_frame,
                text=f"{kernel_um} um",
                style="SidebarMuted.TLabel",
            ).grid(row=row, column=0, sticky="w", padx=(0, 4), pady=2)
            entry = ttk.Entry(
                kernel_dirs_frame,
                textvariable=self.kernel_dir_vars[kernel_um],
                width=18,
            )
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            entry.bind(
                "<Return>",
                lambda _event, kernel=kernel_um: self.use_kernel_directory(kernel),
            )
            ttk.Button(
                kernel_dirs_frame,
                text="...",
                width=3,
                command=lambda kernel=kernel_um: self.choose_kernel_directory(kernel),
            ).grid(row=row, column=2, sticky="e", padx=(4, 0), pady=2)

        figures = ttk.Frame(main, style="Figures.TFrame")
        figures.grid(row=0, column=1, sticky="nsew")
        figures.columnconfigure(0, weight=1)
        figures.rowconfigure(0, weight=3)
        figures.rowconfigure(1, weight=0)
        figures.rowconfigure(2, weight=2)

        top_figures = ttk.Frame(figures, style="Figures.TFrame")
        top_figures.grid(row=0, column=0, sticky="nsew")
        top_figures.columnconfigure(0, weight=1)
        top_figures.columnconfigure(1, weight=1)
        top_figures.rowconfigure(0, weight=1)

        self.image_fig = Figure(figsize=(6, 5), dpi=100, facecolor="#ffffff")
        self.image_ax = self.image_fig.add_subplot(111)
        self.image_canvas = FigureCanvasTkAgg(self.image_fig, master=top_figures)
        self.image_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.image_canvas.get_tk_widget().configure(background="#ffffff", highlightthickness=0)
        self.image_canvas.mpl_connect("button_press_event", self.on_image_press)
        self.image_canvas.mpl_connect("motion_notify_event", self.on_image_motion)
        self.image_canvas.mpl_connect("button_release_event", self.on_image_release)
        self.image_toolbar = NavigationToolbar2Tk(self.image_canvas, figures, pack_toolbar=False)
        self.image_toolbar.grid(row=1, column=0, sticky="ew")
        self._style_toolbar(self.image_toolbar)

        self.reference_map_fig = Figure(figsize=(6, 5), dpi=100, facecolor="#ffffff")
        self.reference_map_ax = self.reference_map_fig.add_subplot(111)
        self.reference_map_canvas = FigureCanvasTkAgg(self.reference_map_fig, master=top_figures)
        self.reference_map_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.reference_map_canvas.get_tk_widget().configure(background="#ffffff", highlightthickness=0)
        self.reference_map_canvas.mpl_connect("button_press_event", self.on_image_press)
        self.reference_map_canvas.mpl_connect("motion_notify_event", self.on_image_motion)
        self.reference_map_canvas.mpl_connect("button_release_event", self.on_image_release)
        self.clear_reference_map("Select an interval to preview a density map.")

        lower_figures = ttk.Frame(figures, style="Figures.TFrame")
        lower_figures.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        lower_figures.columnconfigure(0, weight=1)
        lower_figures.columnconfigure(1, weight=1)
        lower_figures.rowconfigure(0, weight=0)
        lower_figures.rowconfigure(1, weight=1)

        kernel_selector = ttk.Frame(lower_figures, style="Figures.TFrame")
        kernel_selector.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 3))
        kernel_selector.columnconfigure(0, weight=1)
        self.kernel_slider = ttk.Scale(
            kernel_selector,
            from_=0,
            to=len(KERNEL_SIZES_UM) - 1,
            orient="horizontal",
            command=self._on_kernel_slider,
        )
        self._syncing_kernel_slider = True
        self.kernel_slider.set(self.kernel_index_var.get())
        self._syncing_kernel_slider = False
        self.kernel_slider.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            kernel_selector,
            textvariable=self.kernel_label_var,
            style="SidebarMuted.TLabel",
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        self.profile_fig = Figure(figsize=(5, 3), dpi=100, facecolor="#ffffff")
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_canvas = FigureCanvasTkAgg(self.profile_fig, master=lower_figures)
        self.profile_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.profile_canvas.get_tk_widget().configure(background="#ffffff", highlightthickness=0)

        self.scatter_fig = Figure(figsize=(5, 3), dpi=100, facecolor="#ffffff")
        self.scatter_ax = self.scatter_fig.add_subplot(111)
        self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=lower_figures)
        self.scatter_canvas.get_tk_widget().grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        self.scatter_canvas.get_tk_widget().configure(background="#ffffff", highlightthickness=0)
        self.clear_scatter_preview("Draw a line to preview scatterplots.")

        status = ttk.Label(main, textvariable=self.status_var, anchor="w", style="Status.TLabel")
        status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def infer_kernel_size_um(self, run_dir: Path) -> int:
        match = re.search(r"kernel[_-]?(40|80|160|320)(?:\D|$)", run_dir.name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        radius_um = float(resolve_radius_um(self.config))
        return min(KERNEL_SIZES_UM, key=lambda value: abs(float(value) - radius_um))

    def _set_kernel_slider(self, kernel_um: int) -> None:
        index = KERNEL_SIZES_UM.index(kernel_um)
        self.kernel_index_var.set(index)
        self.kernel_label_var.set(f"Kernel: {kernel_um} um")
        self._syncing_kernel_slider = True
        try:
            self.kernel_slider.set(index)
        finally:
            self._syncing_kernel_slider = False

    def _on_kernel_slider(self, value: str) -> None:
        if getattr(self, "_syncing_kernel_slider", False):
            return
        index = max(0, min(int(round(float(value))), len(KERNEL_SIZES_UM) - 1))
        kernel_um = KERNEL_SIZES_UM[index]
        self.kernel_index_var.set(index)
        self.kernel_label_var.set(f"Kernel: {kernel_um} um")
        if self._pending_kernel_switch is not None:
            self.root.after_cancel(self._pending_kernel_switch)
        self._pending_kernel_switch = self.root.after(
            180,
            lambda: self.use_kernel_directory(kernel_um),
        )

    def choose_kernel_directory(self, kernel_um: int) -> None:
        current_text = self.kernel_dir_vars[kernel_um].get().strip()
        initial_dir = Path(current_text) if current_text else self.run_dir.parent
        if not initial_dir.exists():
            initial_dir = self.run_dir.parent
        selected = filedialog.askdirectory(
            title=f"Select {kernel_um} um kernel run directory",
            initialdir=os.fspath(initial_dir),
            mustexist=True,
        )
        if not selected:
            return
        self.kernel_dir_vars[kernel_um].set(selected)
        if KERNEL_SIZES_UM[self.kernel_index_var.get()] == kernel_um:
            self.use_kernel_directory(kernel_um)

    def use_kernel_directory(self, kernel_um: int) -> None:
        self._pending_kernel_switch = None
        directory_text = self.kernel_dir_vars[kernel_um].get().strip()
        if not directory_text:
            messagebox.showwarning(
                "Kernel directory missing",
                f"Enter or browse to the {kernel_um} um run directory on the left.",
            )
            self._set_kernel_slider(self.infer_kernel_size_um(self.run_dir))
            return
        target_dir = Path(directory_text).expanduser()
        if not target_dir.is_dir():
            messagebox.showerror(
                "Kernel directory not found",
                f"The {kernel_um} um run directory does not exist:\n{target_dir}",
            )
            self._set_kernel_slider(self.infer_kernel_size_um(self.run_dir))
            return

        old_plate = self.plate_var.get()
        old_well = self.well_var.get()
        old_change_type = self.change_type_var.get()
        old_interval = self.interval_var.get()
        old_factor = float(self.coarsening_factor)
        physical_line = [
            (float(x) * old_factor, float(y) * old_factor)
            for x, y in self.click_points
        ]

        plates = discover_run_plates(target_dir)
        target_plate = old_plate if old_plate in plates else (plates[0] if plates else "")
        target_wells = discover_wells(target_dir, target_plate) if target_plate else []
        target_well = old_well if old_well in target_wells else (target_wells[0] if target_wells else "")
        if not target_plate or not target_well:
            messagebox.showerror(
                "No matching maps",
                f"No usable plate/well directories were found in:\n{target_dir}",
            )
            self._set_kernel_slider(self.infer_kernel_size_um(self.run_dir))
            return

        self.stop_interval_playback(update_status=False)
        self.run_dir = target_dir
        self.coarsening_factor = kernel_um // 10
        self.pixel_size_um = PIXEL_SIZE_UM * self.coarsening_factor
        self.image_cache.clear()
        self.raw_image_shape_cache.clear()
        self.shared_map_limits_cache.clear()
        self.lut_limits.clear()
        self.clear_lut_textboxes()
        self._is_simulation_run_cache = None
        self._map_view_context = None
        self.reset_map_zoom()
        self.click_points.clear()
        self.current_profile = None

        self.plate_box["values"] = [""] + plates
        self.plate_var.set(target_plate)
        self.well_box["values"] = target_wells
        self.well_var.set(target_well)
        self._plate_by_well = {}

        entries = self._available_changes()
        kinds = sorted({entry.kind for entry in entries})
        labels = [CHANGE_TYPE_LABELS.get(kind, kind) for kind in kinds]
        self._kind_by_label = dict(zip(labels, kinds))
        self.change_type_box["values"] = labels
        selected_change_type = old_change_type if old_change_type in labels else (
            CHANGE_TYPE_LABELS["res_density_epanechnikov_rel_diff"]
            if CHANGE_TYPE_LABELS["res_density_epanechnikov_rel_diff"] in labels
            else (labels[0] if labels else "")
        )
        self.change_type_var.set(selected_change_type)
        self.interval_var.set(old_interval)
        self._refresh_intervals()

        new_factor = float(self.coarsening_factor)
        if len(physical_line) == 2:
            self.click_points = [
                (x_physical / new_factor, y_physical / new_factor)
                for x_physical, y_physical in physical_line
            ]
            self.redraw_image()
            self.update_profile(redraw_overlay=False)

        self.kernel_dir_vars[kernel_um].set(os.fspath(target_dir))
        self._set_kernel_slider(kernel_um)
        self.run_dir_info_var.set(self.run_dir_info_text())
        self.status_var.set(
            f"Loaded {kernel_um} um kernel: {target_dir}\n"
            f"Kept {target_plate} / {target_well} / {self.interval_var.get()} and profile line."
        )

    def run_dir_info_text(self) -> str:
        return (
            f"Run:\n{self.run_dir}\n\n"
            f"Pixel size:\n{self.pixel_size_um:.3f} um "
            f"({self.coarsening_factor}x)"
        )

    def choose_run_dir(self) -> None:
        initial_dir = self.run_dir if self.run_dir.exists() else self.run_dir.parent
        selected = filedialog.askdirectory(
            title="Select pipeline run directory",
            initialdir=os.fspath(initial_dir),
            mustexist=True,
        )
        if not selected:
            return
        self.set_run_dir(Path(selected).expanduser())

    def set_run_dir(self, run_dir: Path) -> None:
        if not run_dir.exists() or not run_dir.is_dir():
            messagebox.showerror("Missing run directory", f"Run directory does not exist:\n{run_dir}")
            return
        try:
            current = self.run_dir.resolve()
            selected = run_dir.resolve()
        except OSError:
            current = self.run_dir
            selected = run_dir
        if selected == current:
            self.status_var.set(f"Run directory unchanged: {run_dir}")
            return

        self.run_dir = run_dir
        kernel_um = self.infer_kernel_size_um(run_dir)
        self.coarsening_factor = kernel_um // 10
        self.pixel_size_um = PIXEL_SIZE_UM * self.coarsening_factor
        self.kernel_dir_vars[kernel_um].set(os.fspath(run_dir))
        self._set_kernel_slider(kernel_um)
        self.image_cache.clear()
        self.raw_image_shape_cache.clear()
        self.shared_map_limits_cache.clear()
        self.lut_limits.clear()
        self.clear_lut_textboxes()
        self._is_simulation_run_cache = None
        self._plate_by_well = {}
        self.run_dir_info_var.set(self.run_dir_info_text())
        self._populate_plates()
        self.status_var.set(f"Run directory changed:\n{self.run_dir}")

    def format_density_gate_value(self, value: float) -> str:
        return f"{value:.6g} cells/um^2"

    def res_density_gate_limits(self) -> tuple[float, float]:
        lower = float(self.res_density_gate_lower_var.get())
        upper = float(self.res_density_gate_upper_var.get())
        return min(lower, upper), max(lower, upper)

    def res_density_gate_active(self) -> bool:
        return bool(self.res_density_gate_enabled_var.get())

    def _update_res_density_gate_labels(self) -> None:
        if not hasattr(self, "res_density_gate_lower_label"):
            return
        lower, upper = self.res_density_gate_limits()
        self.res_density_gate_lower_label.configure(
            text=f"Lower: {self.format_density_gate_value(lower)}"
        )
        self.res_density_gate_upper_label.configure(
            text=f"Upper: {self.format_density_gate_value(upper)}"
        )

    def _set_res_density_gate_bound(
        self,
        bound: str,
        value: float,
        *,
        refresh: bool = True,
    ) -> None:
        value = min(
            max(float(value), self.res_density_gate_slider_min),
            self.res_density_gate_slider_max,
        )
        lower = float(self.res_density_gate_lower_var.get())
        upper = float(self.res_density_gate_upper_var.get())
        if bound == "lower":
            lower = min(value, upper)
        else:
            upper = max(value, lower)

        self.res_density_gate_lower_var.set(lower)
        self.res_density_gate_upper_var.set(upper)
        self._syncing_res_density_gate = True
        try:
            if hasattr(self, "res_density_gate_lower_slider"):
                self.res_density_gate_lower_slider.set(lower)
            if hasattr(self, "res_density_gate_upper_slider"):
                self.res_density_gate_upper_slider.set(upper)
        finally:
            self._syncing_res_density_gate = False
        self._update_res_density_gate_labels()
        if refresh:
            self._refresh_res_density_gate()

    def _on_res_density_gate_slider(self, bound: str, value: str) -> None:
        if self._syncing_res_density_gate:
            return
        self._set_res_density_gate_bound(bound, float(value), refresh=True)

    def _step_res_density_gate(self, bound: str, direction: int) -> None:
        current = (
            self.res_density_gate_lower_var.get()
            if bound == "lower"
            else self.res_density_gate_upper_var.get()
        )
        self._set_res_density_gate_bound(
            bound,
            float(current) + direction * self.res_density_gate_step,
            refresh=True,
        )

    def _on_res_density_gate_changed(self) -> None:
        self._update_res_density_gate_labels()
        self._refresh_res_density_gate()

    def _refresh_res_density_gate(self) -> None:
        if self.current_profile is not None:
            self.update_profile(redraw_overlay=False)
        else:
            self.update_scatter_preview()
        self.redraw_image()
        if self.res_density_gate_active():
            lower, upper = self.res_density_gate_limits()
            self.status_var.set(
                "Resistant density gate active: "
                f"{self.format_density_gate_value(lower)} to {self.format_density_gate_value(upper)}."
            )
        else:
            self.status_var.set("Resistant density gate off.")

    def _clear_current_selection(self, message: str) -> None:
        self.stop_interval_playback(update_status=False)
        self.current_change = None
        self.current_profile = None
        self.image = None
        self.display_image = None
        self.click_points.clear()
        self.reset_map_zoom()
        self.map_roi = None
        self.scatter_selection_bounds = None
        self.scatter_selection_polygon = None
        self.scatter_selection_panel = None
        self.scatter_selection_shape = "Rectangle"
        self.selected_scatter_points = None
        self.clear_res_density_gate_overlays()
        self._kind_by_label = {}
        self._entry_by_label = {}
        self.change_type_box["values"] = []
        self.change_type_var.set("")
        self.interval_box["values"] = []
        self.interval_var.set("")
        self._sync_interval_slider_to_labels([])

        if self.image_colorbar is not None:
            try:
                self.image_colorbar.remove()
            except Exception:
                pass
            self.image_colorbar = None
        if self.profile_colorbar is not None:
            try:
                self.profile_colorbar.remove()
            except Exception:
                pass
            self.profile_colorbar = None

        self.clear_lut_textboxes(self.image_fig)
        self.image_ax.clear()
        self.image_canvas.draw_idle()
        self.profile_fig.clear()
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_canvas.draw_idle()
        self.clear_reference_map("Select a well to preview a density map.")
        self.clear_scatter_preview("Draw a line to preview scatterplots.")
        self.status_var.set(message)

    def _populate_plates(self) -> None:
        if not self.run_dir.exists():
            messagebox.showerror("Missing run directory", f"Run directory does not exist:\n{self.run_dir}")
            return
        plates = discover_run_plates(self.run_dir)
        self.plate_box["values"] = [""] + plates
        self.plate_var.set("")
        self._on_plate_changed()

    def _on_plate_changed(self) -> None:
        plate = self.plate_var.get()
        if not plate:
            wells, self._plate_by_well = discover_all_wells(self.run_dir)
            self.well_box["values"] = wells
            self.well_var.set("")
            self._clear_current_selection(
                "No plate selected. Choose a well to select its matching plate automatically."
            )
            return

        wells = discover_wells(self.run_dir, plate)
        self.well_box["values"] = wells
        if wells:
            self.well_var.set(wells[0])
        else:
            self.well_var.set("")
        self._on_well_changed()

    def _on_well_changed(self) -> None:
        if not self.plate_var.get():
            plate = self._plate_by_well.get(self.well_var.get())
            if not plate:
                self._clear_current_selection("Select a well.")
                return
            self.plate_var.set(plate)
            plate_wells = discover_wells(self.run_dir, plate)
            self.well_box["values"] = plate_wells
            if self.well_var.get() not in plate_wells:
                self.well_var.set("")
                self._clear_current_selection("Selected well is not available for the detected plate.")
                return

        entries = self._available_changes()
        kinds = sorted({entry.kind for entry in entries})
        labels = [CHANGE_TYPE_LABELS.get(kind, kind) for kind in kinds]
        self._kind_by_label = dict(zip(labels, kinds))
        self.change_type_box["values"] = labels
        if not labels:
            self.change_type_var.set("")
            self._refresh_intervals()
            return
        if labels and self.change_type_var.get() not in labels:
            preferred = CHANGE_TYPE_LABELS["res_density_epanechnikov_rel_diff"]
            self.change_type_var.set(preferred if preferred in labels else labels[0])
        self._refresh_intervals()

    def _available_changes(self) -> list[ChangeEntry]:
        changes_dir = (
            self.run_dir
            / self.plate_var.get()
            / self.well_var.get()
            / self.density_changes_name
        )
        return discover_changes(changes_dir, self.well_var.get())

    def _selected_kind(self) -> str | None:
        return getattr(self, "_kind_by_label", {}).get(self.change_type_var.get())

    def _refresh_intervals(self) -> None:
        self.stop_interval_playback(update_status=False)
        kind = self._selected_kind()
        entries = [entry for entry in self._available_changes() if entry.kind == kind]
        self._entry_by_label = {entry.label: entry for entry in entries}
        labels = list(self._entry_by_label)
        self.interval_box["values"] = labels
        if labels and self.interval_var.get() not in labels:
            self.interval_var.set(labels[0])
        self._sync_interval_slider_to_labels(labels)
        self.load_image()

    def _sync_interval_slider_to_labels(self, labels: list[str]) -> None:
        self._syncing_interval_slider = True
        if not labels:
            self.interval_slider.configure(from_=0, to=0)
            self.interval_slider.set(0)
            self.interval_slider_label.configure(text="Interval 0/0")
            self._syncing_interval_slider = False
            return
        max_index = max(0, len(labels) - 1)
        self.interval_slider.configure(from_=0, to=max_index)
        try:
            index = labels.index(self.interval_var.get())
        except ValueError:
            index = 0
        self.interval_index_var.set(index)
        self.interval_slider.set(index)
        self.interval_slider_label.configure(
            text=f"Interval {index + 1}/{len(labels)}: {labels[index]}"
        )
        self._syncing_interval_slider = False

    def _on_interval_slider(self, value: str) -> None:
        if self._syncing_interval_slider:
            return
        labels = list(getattr(self, "_entry_by_label", {}).keys())
        if not labels:
            return
        index = int(round(float(value)))
        index = max(0, min(index, len(labels) - 1))
        self._set_interval_index(index, labels=labels, move_slider=False, load=False)
        if self._pending_interval_load is not None:
            self.root.after_cancel(self._pending_interval_load)
        self._pending_interval_load = self.root.after(
            INTERVAL_SLIDER_DEBOUNCE_MS,
            self._load_interval_after_slider,
        )

    def _load_interval_after_slider(self) -> None:
        self._pending_interval_load = None
        self.load_image()

    def interval_playback_fps(self) -> float:
        try:
            fps = float(self.playback_fps_var.get())
        except (tk.TclError, ValueError):
            fps = 2.0
        fps = min(max(fps, 0.2), 60.0)
        self.playback_fps_var.set(fps)
        return fps

    def interval_playback_delay_ms(self) -> int:
        return max(1, int(round(1000.0 / self.interval_playback_fps())))

    def toggle_interval_playback(self) -> None:
        if self._playback_after_id is not None:
            self.stop_interval_playback()
        else:
            self.start_interval_playback()

    def start_interval_playback(self) -> None:
        labels = list(getattr(self, "_entry_by_label", {}).keys())
        if len(labels) < 2:
            self.status_var.set("Need at least two intervals for playback.")
            return
        if self.interval_index_var.get() >= len(labels) - 1:
            self._set_interval_index(0, labels=labels, move_slider=True, load=True)
        self.playback_button_text_var.set("Stop")
        self.status_var.set(f"Playing intervals at {self.interval_playback_fps():g} FPS.")
        self.schedule_interval_playback()

    def stop_interval_playback(self, *, update_status: bool = True) -> None:
        if self._playback_after_id is not None:
            try:
                self.root.after_cancel(self._playback_after_id)
            except tk.TclError:
                pass
            self._playback_after_id = None
        if hasattr(self, "playback_button_text_var"):
            self.playback_button_text_var.set("Play")
        if update_status:
            self.status_var.set("Interval playback stopped.")

    def schedule_interval_playback(self) -> None:
        if self._playback_after_id is not None:
            try:
                self.root.after_cancel(self._playback_after_id)
            except tk.TclError:
                pass
        self._playback_after_id = self.root.after(
            self.interval_playback_delay_ms(),
            self.advance_interval_playback,
        )

    def advance_interval_playback(self) -> None:
        self._playback_after_id = None
        labels = list(getattr(self, "_entry_by_label", {}).keys())
        if len(labels) < 2:
            self.stop_interval_playback(update_status=False)
            return
        next_index = self.interval_index_var.get() + 1
        if next_index >= len(labels):
            self.stop_interval_playback(update_status=False)
            self.status_var.set("Interval playback reached the last interval.")
            return
        self._set_interval_index(next_index, labels=labels, move_slider=True, load=True)
        self.schedule_interval_playback()

    def _on_interval_dropdown(self) -> None:
        labels = list(getattr(self, "_entry_by_label", {}).keys())
        if self.interval_var.get() in labels:
            index = labels.index(self.interval_var.get())
            self._set_interval_index(index, labels=labels, move_slider=True, load=False)
        self.load_image()

    def _set_interval_index(
        self,
        index: int,
        *,
        labels: list[str] | None = None,
        move_slider: bool,
        load: bool,
    ) -> None:
        labels = labels or list(getattr(self, "_entry_by_label", {}).keys())
        if not labels:
            return
        index = max(0, min(index, len(labels) - 1))
        self.interval_index_var.set(index)
        self.interval_var.set(labels[index])
        if move_slider:
            self._syncing_interval_slider = True
            self.interval_slider.set(index)
            self._syncing_interval_slider = False
        self.interval_slider_label.configure(
            text=f"Interval {index + 1}/{len(labels)}: {labels[index]}"
        )
        if load:
            if self._pending_interval_load is not None:
                self.root.after_cancel(self._pending_interval_load)
                self._pending_interval_load = None
            self.load_image()

    def _step_interval(self, delta: int) -> None:
        labels = list(getattr(self, "_entry_by_label", {}).keys())
        if not labels:
            return
        self._set_interval_index(
            self.interval_index_var.get() + delta,
            labels=labels,
            move_slider=True,
            load=True,
        )

    def _on_width_slider(self, value: str) -> None:
        width = max(1, int(round(float(value))))
        if width % 2 == 0:
            width += 1
        self.width_var.set(width)
        if hasattr(self, "width_label"):
            self.width_label.configure(text=f"{width} px ({width * self.pixel_size_um:.1f} um)")
        if len(self.click_points) == 2 and self.image is not None:
            self.update_profile(redraw_overlay=True)

    def _on_line_mode_changed(self) -> None:
        self.clear_line()
        if self.center_to_edge_var.get():
            self.status_var.set("Center-to-edge mode: click or drag to set the outward direction.")
        else:
            self.status_var.set("Free-line mode: click two points or drag from start to endpoint.")

    def selected_entry(self) -> ChangeEntry | None:
        return getattr(self, "_entry_by_label", {}).get(self.interval_var.get())

    def y_axis_label(self, entry: ChangeEntry | None = None) -> str:
        kind = entry.kind if entry is not None else self._selected_kind()
        return Y_AXIS_LABELS.get(kind, self.change_type_var.get())

    def current_profile_y_axis_label(self, fallback_entry: ChangeEntry | None = None) -> str:
        return self.y_axis_label(self.current_change or fallback_entry)

    def image_colorbar_label(self, entry: ChangeEntry | None = None) -> str:
        background = BACKGROUND_CHOICES.get(self.background_var.get(), "change")
        if background == "change":
            return self.y_axis_label(entry)
        density_labels = {
            "total": "Total density [cells/µm²]",
            "res": "Resistant density [cells/µm²]",
            "sus": "Sensitive density [cells/µm²]",
        }
        return density_labels.get(background, "Image value")

    def density_channel_label(self, channel: str) -> str:
        density_labels = {
            "total": "Total density [cells/um^2]",
            "res": "Resistant density [cells/um^2]",
            "sus": "Sensitive density [cells/um^2]",
            "cell_positions": "HDF5 cell positions",
        }
        return density_labels.get(channel, "Density [cells/um^2]")

    def format_lut_value(self, value: float) -> str:
        return f"{float(value):.4g}"

    def main_map_lut_key(self) -> tuple[str, str]:
        background = BACKGROUND_CHOICES.get(self.background_var.get(), "change")
        if background == "change":
            descriptor = f"change:{self._selected_kind() or self.change_type_var.get()}"
        else:
            descriptor = f"density:{background}"
        return "main", descriptor

    def reference_map_lut_key(self, channel: str) -> tuple[str, str]:
        return "reference", str(channel)

    def effective_lut_limits(
        self,
        key: tuple[str, str],
        auto_limits: tuple[float, float],
    ) -> tuple[float, float]:
        manual = self.lut_limits.get(key)
        if manual is None:
            return auto_limits
        lower, upper = manual
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            return auto_limits
        return float(lower), float(upper)

    def clear_lut_textboxes(self, figure=None) -> None:
        remaining = []
        for textbox in self.lut_textboxes:
            if figure is not None and textbox.ax.figure is not figure:
                remaining.append(textbox)
                continue
            try:
                textbox.ax.remove()
            except Exception:
                pass
        self.lut_textboxes = remaining

    def add_lut_textboxes(
        self,
        colorbar,
        mappable,
        key: tuple[str, str],
        auto_limits: tuple[float, float],
        canvas,
    ) -> None:
        figure = colorbar.ax.figure
        position = colorbar.ax.get_position()
        box_height = 0.026
        gap = 0.006
        box_width = max(position.width * 3.2, 0.07)
        box_x = min(max(position.x0 + position.width / 2.0 - box_width / 2.0, 0.01), 0.99 - box_width)
        lower_y = max(position.y0 - box_height - gap, 0.01)
        upper_y = min(position.y1 + gap, 0.99 - box_height)
        current_lower, current_upper = mappable.get_clim()

        def _submit_limit(bound: str, text: str) -> None:
            stripped = str(text).strip()
            if stripped.lower() == "auto":
                self.lut_limits.pop(key, None)
                lower, upper = auto_limits
            else:
                try:
                    value = float(stripped)
                except ValueError:
                    self.status_var.set("LUT limit must be a number, or 'auto'.")
                    return
                lower, upper = self.lut_limits.get(key, mappable.get_clim())
                if bound == "lower":
                    lower = value
                else:
                    upper = value
                if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                    self.status_var.set("LUT lower limit must be smaller than upper limit.")
                    return
                self.lut_limits[key] = (float(lower), float(upper))

            mappable.set_clim(float(lower), float(upper))
            colorbar.update_normal(mappable)
            canvas.draw_idle()
            self.status_var.set(f"LUT set to {float(lower):.4g} - {float(upper):.4g}.")

        upper_ax = figure.add_axes([box_x, upper_y, box_width, box_height])
        upper_box = TextBox(upper_ax, "", initial=self.format_lut_value(current_upper))
        upper_box.on_submit(lambda text: _submit_limit("upper", text))
        lower_ax = figure.add_axes([box_x, lower_y, box_width, box_height])
        lower_box = TextBox(lower_ax, "", initial=self.format_lut_value(current_lower))
        lower_box.on_submit(lambda text: _submit_limit("lower", text))
        for textbox in (upper_box, lower_box):
            textbox.text_disp.set_fontsize(7)
            textbox.cursor.set_visible(False)
        self.lut_textboxes.extend([upper_box, lower_box])

    def background_path(self, entry: ChangeEntry) -> Path:
        background = BACKGROUND_CHOICES.get(self.background_var.get(), "change")
        if background == "change":
            return entry.path
        return self.density_map_path(entry, background)

    def background_limit_paths(self, entry: ChangeEntry) -> list[Path]:
        background = BACKGROUND_CHOICES.get(self.background_var.get(), "change")
        entries = [
            candidate
            for candidate in self._available_changes()
            if candidate.kind == entry.kind
        ]
        if background == "change":
            paths = [candidate.path for candidate in entries]
        else:
            paths = [self.density_map_path(candidate, background) for candidate in entries]
        return self.unique_existing_paths(paths)

    def reference_map_path_and_label(self, entry: ChangeEntry) -> tuple[Path, str, str]:
        channel = REFERENCE_MAP_CHOICES.get(self.reference_map_var.get(), "total")
        time_choice = REFERENCE_TIME_CHOICES.get(self.reference_time_var.get(), "first")
        timepoint = entry.next_tp if time_choice == "next" else entry.first_tp
        if channel == "cell_positions":
            return self.cell_position_hdf5_path(), channel, timepoint
        path = (
            self.run_dir
            / self.plate_var.get()
            / self.well_var.get()
            / self.density_maps_name
            / f"{self.well_var.get()}_{timepoint}_{channel}_density_epanechnikov.tif"
        )
        return path, channel, timepoint

    def reference_map_limit_paths(self, entry: ChangeEntry) -> list[Path]:
        channel = REFERENCE_MAP_CHOICES.get(self.reference_map_var.get(), "total")
        if channel == "cell_positions":
            return []
        time_choice = REFERENCE_TIME_CHOICES.get(self.reference_time_var.get(), "first")
        entries = [
            candidate
            for candidate in self._available_changes()
            if candidate.kind == entry.kind
        ]
        paths = []
        for candidate in entries:
            timepoint = candidate.next_tp if time_choice == "next" else candidate.first_tp
            paths.append(
                self.run_dir
                / self.plate_var.get()
                / self.well_var.get()
                / self.density_maps_name
                / f"{self.well_var.get()}_{timepoint}_{channel}_density_epanechnikov.tif"
            )
        return self.unique_existing_paths(paths)

    def available_well_timepoints(self) -> list[str]:
        timepoints: set[str] = set()
        for entry in self._available_changes():
            timepoints.add(entry.first_tp)
            timepoints.add(entry.next_tp)
        return sorted(timepoints, key=timepoint_key)

    def pointcloud_suffix_priority(self, path: Path) -> tuple[int, str]:
        for index, suffix in enumerate(POINTCLOUD_SUFFIX_PRIORITY):
            if path.name.endswith(suffix):
                return index, path.name
        return len(POINTCLOUD_SUFFIX_PRIORITY), path.name

    def experiment_path(self) -> Path:
        return Path(
            resolve_experiment_path(
                experiment_path=self.config.get("experiment_path"),
                mount_root=self.config.get("mount_root"),
                hdf5_path=self.config.get("hdf5_path"),
            )
        )

    def hdf5_directory(self) -> Path:
        run_local_hdf5 = self.run_dir / self.plate_var.get() / "hdf5"
        if run_local_hdf5.exists():
            return run_local_hdf5
        hdf5_path = self.config.get("hdf5_path")
        if hdf5_path:
            return Path(hdf5_path).expanduser()
        return self.experiment_path() / self.plate_var.get() / "hdf5"

    def cell_position_hdf5_path(self) -> Path:
        hdf5_dir = self.hdf5_directory()
        if not hdf5_dir.exists():
            return hdf5_dir / f"{self.well_var.get()}_simulation.h5"
        poisson = bool(self.config.get("poisson", True))
        try:
            filenames = pointcloud_filenames(
                os.fspath(hdf5_dir),
                poisson,
                well_designators=[self.well_var.get()],
            )
        except Exception:
            filenames = []
        candidates = [hdf5_dir / filename for filename in filenames]
        if not candidates:
            well = self.well_var.get()
            candidates = [
                path
                for path in hdf5_dir.iterdir()
                if path.is_file()
                and path.name.endswith(POINTCLOUD_SUFFIX_PRIORITY)
                and well_identifier_from_pointcloud_filename(path) == well
            ]
        if not candidates:
            return hdf5_dir / f"{self.well_var.get()}_simulation.h5"
        return sorted(candidates, key=self.pointcloud_suffix_priority)[0]

    def density_map_path_for_timepoint(self, timepoint: str, channel: str = "total") -> Path:
        return (
            self.run_dir
            / self.plate_var.get()
            / self.well_var.get()
            / self.density_maps_name
            / f"{self.well_var.get()}_{timepoint}_{channel}_density_epanechnikov.tif"
        )

    def cell_position_frame_extent(self, timepoint: str) -> tuple[float, float, float, float] | None:
        shape = self.cell_position_frame_coarse_shape(timepoint)
        if shape is None:
            return None
        height, width = shape
        return (-0.5, float(width) - 0.5, float(height) - 0.5, -0.5)

    def cell_position_frame_coarse_shape(self, timepoint: str) -> tuple[int, int] | None:
        density_path = self.density_map_path_for_timepoint(timepoint, "total")
        shape = None
        if density_path.exists():
            try:
                shape = self.read_cached_image(density_path).shape
            except Exception:
                shape = None
        if shape is None:
            if self.image is not None:
                shape = self.image.shape
            elif self.display_image is not None:
                shape = self.display_image.shape
        if shape is None:
            return None
        return int(shape[0]), int(shape[1])

    def crop_cell_position_frame_to_data_extent(
        self,
        image: np.ndarray,
        timepoint: str,
    ) -> np.ndarray:
        shape = self.cell_position_frame_coarse_shape(timepoint)
        if shape is None or image.ndim < 2:
            return image
        coarse_height, coarse_width = shape
        if coarse_height <= 0 or coarse_width <= 0:
            return image

        image_height, image_width = image.shape[:2]
        data_aspect = coarse_width / coarse_height
        image_aspect = image_width / image_height

        if math.isclose(data_aspect, image_aspect, rel_tol=1e-3, abs_tol=1e-3):
            return image

        if data_aspect < image_aspect:
            crop_width = max(1, int(round(image_height * data_aspect)))
            x0 = max(0, (image_width - crop_width) // 2)
            return image[:, x0 : x0 + crop_width, ...]

        crop_height = max(1, int(round(image_width / data_aspect)))
        y0 = max(0, (image_height - crop_height) // 2)
        return image[y0 : y0 + crop_height, :, ...]

    def unique_existing_paths(self, paths: list[Path]) -> list[Path]:
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = os.fspath(path)
            if key in seen or not path.exists():
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def robust_limits_for_paths(self, paths: list[Path]) -> tuple[float, float] | None:
        if not paths:
            return None
        key = tuple(os.fspath(path) for path in paths)
        if key in self.shared_map_limits_cache:
            return self.shared_map_limits_cache[key]

        finite_chunks: list[np.ndarray] = []
        for path in paths:
            try:
                image = self.read_cached_image(path)
            except Exception as exc:
                print(f"[WARN] Could not read map for shared color scaling: {path}: {exc}")
                continue
            finite = image[np.isfinite(image)].ravel()
            if finite.size:
                finite_chunks.append(finite)
        if not finite_chunks:
            self.shared_map_limits_cache[key] = None
            return None

        values = np.concatenate(finite_chunks)
        max_values = 2_000_000
        if values.size > max_values:
            step = int(math.ceil(values.size / max_values))
            values = values[::step]

        lo, hi = np.nanpercentile(values, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
        if lo == hi:
            hi = lo + 1.0
        limits = (float(lo), float(hi))
        self.shared_map_limits_cache[key] = limits
        return limits

    def density_map_path(self, entry: ChangeEntry, channel: str) -> Path:
        return (
            self.run_dir
            / self.plate_var.get()
            / self.well_var.get()
            / self.density_maps_name
            / f"{self.well_var.get()}_{entry.first_tp}_{channel}_density_epanechnikov.tif"
        )

    def relative_entry_for_current_interval(self) -> ChangeEntry | None:
        selected = self.selected_entry()
        if selected is None:
            return None
        if selected.kind == "res_density_epanechnikov_rel_diff":
            return selected
        for entry in self._available_changes():
            if (
                entry.kind == "res_density_epanechnikov_rel_diff"
                and entry.first_tp == selected.first_tp
                and entry.next_tp == selected.next_tp
            ):
                return entry
        return None

    def density_profile_for_entry(
        self,
        entry: ChangeEntry,
        channel: str,
        timepoint: str,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        use_cache: bool = True,
    ) -> ProfileResult | None:
        path = (
            self.run_dir
            / self.plate_var.get()
            / self.well_var.get()
            / self.density_maps_name
            / f"{self.well_var.get()}_{timepoint}_{channel}_density_epanechnikov.tif"
        )
        if not path.exists():
            print(f"[WARN] Missing {channel}-density map: {path}")
            return None
        image = self.read_image(path, use_cache=use_cache)
        return sample_line_profile(
            image,
            start_xy,
            end_xy,
            self.width_var.get(),
            self.pixel_size_um,
        )

    def total_density_profile_for_entry(
        self,
        entry: ChangeEntry,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        use_cache: bool = True,
    ) -> ProfileResult | None:
        total_path = self.density_map_path(entry, "total")
        if not total_path.exists():
            print(f"[WARN] Missing total-density map: {total_path}")
            return None
        total_image = self.read_image(total_path, use_cache=use_cache)
        return sample_line_profile(
            total_image,
            start_xy,
            end_xy,
            self.width_var.get(),
            self.pixel_size_um,
        )

    def resistant_fraction_profile_for_entry(
        self,
        entry: ChangeEntry,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        use_cache: bool = True,
    ) -> np.ndarray | None:
        res_profile = self.density_profile_for_entry(
            entry,
            "res",
            entry.first_tp,
            start_xy,
            end_xy,
            use_cache=use_cache,
        )
        sus_profile = self.density_profile_for_entry(
            entry,
            "sus",
            entry.first_tp,
            start_xy,
            end_xy,
            use_cache=use_cache,
        )
        if res_profile is None or sus_profile is None:
            return None
        if res_profile.values.shape != sus_profile.values.shape:
            return None

        denominator = res_profile.values + sus_profile.values
        valid = np.isfinite(res_profile.values) & np.isfinite(sus_profile.values) & (denominator > 0)
        return np.divide(
            res_profile.values,
            denominator,
            out=np.full(res_profile.values.shape, np.nan, dtype=float),
            where=valid,
        )

    def parse_axis_limits(self, value: object) -> tuple[float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        return float(value[0]), float(value[1])

    def config_axis_limits(self, key: str) -> tuple[float, float] | None:
        return self.parse_axis_limits(self.config.get(key))

    def scatter_y_axis_limits(self, entry: ChangeEntry) -> tuple[float, float] | None:
        by_kind = self.config.get("density_change_scatter_ylim_by_kind")
        if isinstance(by_kind, dict):
            limits = self.parse_axis_limits(by_kind.get(entry.kind))
            if limits is not None:
                return limits
            limits = self.parse_axis_limits(by_kind.get("default"))
            if limits is not None:
                return limits

        if entry.kind == "res_density_epanechnikov_rel_diff":
            return self.config_axis_limits("relative_density_change_scatter_ylim") or (-1.0, 2.0)
        return (-0.001, 0.003)

    def config_axis_scale(self, key: str) -> str:
        value = str(self.config.get(key, "linear")).strip().lower()
        return value if value in {"linear", "log"} else "linear"

    def zero_initial_threshold(self) -> float:
        return float(self.config.get("zero_initial_res_density_threshold_um2", 1e-12))

    def resistant_growth_threshold(self) -> float:
        return float(self.config.get("res_growth_threshold_um2", self.zero_initial_threshold()))

    def scatter_valid_mask(
        self,
        entry: ChangeEntry,
        x: np.ndarray,
        y: np.ndarray,
        marker_color: np.ndarray,
        next_res: np.ndarray,
    ) -> np.ndarray:
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(marker_color)
            & np.isfinite(next_res)
        )
        zero_threshold = self.zero_initial_threshold()
        if entry.kind == "res_density_epanechnikov_rel_diff":
            valid &= (marker_color > zero_threshold) & (next_res > zero_threshold)
        elif entry.kind == "res_zero_initial_growth_density_epanechnikov_abs_diff":
            valid &= (marker_color <= zero_threshold) & (y > self.resistant_growth_threshold())

        xscale = self.config_axis_scale("relative_density_change_scatter_xscale")
        yscale = self.config_axis_scale("relative_density_change_scatter_yscale")
        if xscale == "log":
            valid &= x > 0
        if yscale == "log":
            valid &= y > 0
        return valid

    def res_density_gate_mask(self, resistant_density: np.ndarray) -> np.ndarray:
        finite = np.isfinite(resistant_density)
        if not self.res_density_gate_active():
            return finite
        lower, upper = self.res_density_gate_limits()
        return finite & (resistant_density >= lower) & (resistant_density <= upper)

    def effective_scatter_valid_mask(
        self,
        entry: ChangeEntry,
        x: np.ndarray,
        y: np.ndarray,
        marker_color: np.ndarray,
        next_res: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_valid = self.scatter_valid_mask(entry, x, y, marker_color, next_res)
        density_gate = self.res_density_gate_mask(marker_color)
        return base_valid & density_gate, base_valid, density_gate

    def scatter_gate_description(self, entry: ChangeEntry) -> str:
        zero_threshold = self.zero_initial_threshold()
        density_gate = ""
        if self.res_density_gate_active():
            lower, upper = self.res_density_gate_limits()
            density_gate = (
                f"; initial res density {lower:g} to {upper:g} cells/um^2"
            )
        if entry.kind == "res_density_epanechnikov_rel_diff":
            return f"plotted where initial and next res > {zero_threshold:g}{density_gate}"
        if entry.kind == "res_zero_initial_growth_density_epanechnikov_abs_diff":
            return (
                f"plotted where initial res <= {zero_threshold:g} and "
                f"change > {self.resistant_growth_threshold():g}{density_gate}"
            )
        return f"plotted finite change-map pixels{density_gate}"

    def scatter_colorbar_vmax(self, values: np.ndarray) -> float:
        configured = self.config.get("density_change_scatter_colorbar_vmax")
        if configured is not None:
            vmax = float(configured)
        else:
            finite = values[np.isfinite(values)]
            vmax = float(np.nanmax(finite)) if finite.size else 1.0
        return vmax if vmax > 0 else 1.0

    def scatter_binned_median_iqr(
        self,
        x: np.ndarray,
        y: np.ndarray,
        point_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return fixed-width x-bin centers, median, IQR, and counts."""
        bin_width = float(self.config.get("scatter_binned_summary_bin_width_um2", 2.0e-4))
        min_points = int(self.config.get("scatter_binned_summary_min_points", 5))
        if bin_width <= 0:
            bin_width = 2.0e-4
        min_points = max(1, min_points)

        selected_x = np.asarray(x[point_indices], dtype=float)
        selected_y = np.asarray(y[point_indices], dtype=float)
        finite = np.isfinite(selected_x) & np.isfinite(selected_y) & (selected_x >= 0.0)
        selected_x = selected_x[finite]
        selected_y = selected_y[finite]
        if selected_x.size == 0:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty, empty, np.empty(0, dtype=int)

        xlim = self.config_axis_limits("relative_density_change_scatter_xlim")
        x_max = float(xlim[1]) if xlim is not None else float(np.nanmax(selected_x))
        x_max = max(x_max, float(np.nanmax(selected_x)), bin_width)
        n_bins = max(1, int(math.ceil(x_max / bin_width)))
        bin_ids = np.floor(selected_x / bin_width).astype(np.int64)
        in_range = (bin_ids >= 0) & (bin_ids < n_bins)
        bin_ids = bin_ids[in_range]
        selected_y = selected_y[in_range]

        centers = (np.arange(n_bins, dtype=float) + 0.5) * bin_width
        median = np.full(n_bins, np.nan, dtype=float)
        q25 = np.full(n_bins, np.nan, dtype=float)
        q75 = np.full(n_bins, np.nan, dtype=float)
        counts = np.bincount(bin_ids, minlength=n_bins)
        for bin_id in np.flatnonzero(counts >= min_points):
            values = selected_y[bin_ids == bin_id]
            q25[bin_id], median[bin_id], q75[bin_id] = np.percentile(
                values,
                [25.0, 50.0, 75.0],
            )
        return centers, median, q25, q75, counts

    def draw_scatter_binned_summary(
        self,
        ax,
        x: np.ndarray,
        y: np.ndarray,
        point_indices: np.ndarray,
    ) -> int:
        centers, median, q25, q75, counts = self.scatter_binned_median_iqr(
            x,
            y,
            point_indices,
        )
        shown = np.isfinite(median)
        if not np.any(shown):
            return 0
        ax.fill_between(
            centers,
            q25,
            q75,
            where=shown,
            color="#ffe600",
            alpha=0.28,
            linewidth=0,
            label="Binned IQR",
            zorder=5,
        )
        ax.plot(
            centers,
            median,
            color="#ffe600",
            linewidth=2.2,
            marker="o",
            markersize=3.0,
            markeredgecolor="#6b5d00",
            markeredgewidth=0.45,
            label="Binned median",
            zorder=6,
        )
        return int(np.count_nonzero(shown))

    def line_scatter_data(
        self,
        rel_entry: ChangeEntry,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        use_cache: bool = True,
    ) -> dict[str, np.ndarray] | None:
        rel_image = self.read_image(rel_entry.path, use_cache=use_cache)
        rel_profile = sample_line_profile(
            rel_image,
            start_xy,
            end_xy,
            self.width_var.get(),
            self.pixel_size_um,
        )
        total_profile = self.density_profile_for_entry(
            rel_entry,
            "total",
            rel_entry.first_tp,
            start_xy,
            end_xy,
            use_cache=use_cache,
        )
        res_profile = self.density_profile_for_entry(
            rel_entry,
            "res",
            rel_entry.first_tp,
            start_xy,
            end_xy,
            use_cache=use_cache,
        )
        next_res_profile = self.density_profile_for_entry(
            rel_entry,
            "res",
            rel_entry.next_tp,
            start_xy,
            end_xy,
            use_cache=use_cache,
        )
        if total_profile is None or res_profile is None or next_res_profile is None:
            return None

        x = total_profile.samples.ravel()
        y = rel_profile.samples.ravel()
        marker_color = res_profile.samples.ravel()
        next_res = next_res_profile.samples.ravel()
        image_x = total_profile.sample_x.ravel()
        image_y = total_profile.sample_y.ravel()
        valid, base_valid, density_gate = self.effective_scatter_valid_mask(
            rel_entry,
            x,
            y,
            marker_color,
            next_res,
        )

        return {
            "x": x,
            "y": y,
            "marker_color": marker_color,
            "valid": valid,
            "base_valid": base_valid,
            "res_density_gate": density_gate,
            "image_x": image_x,
            "image_y": image_y,
        }

    def full_well_scatter_data(
        self,
        rel_entry: ChangeEntry,
    ) -> dict[str, np.ndarray] | None:
        total_path = self.density_map_path(rel_entry, "total")
        res_path = self.density_map_path(rel_entry, "res")
        next_res_path = (
            self.run_dir
            / self.plate_var.get()
            / self.well_var.get()
            / self.density_maps_name
            / f"{self.well_var.get()}_{rel_entry.next_tp}_res_density_epanechnikov.tif"
        )
        if not total_path.exists() or not res_path.exists() or not next_res_path.exists():
            return None

        total_image = self.read_image(total_path, use_cache=True)
        x = total_image.ravel()
        y = self.read_image(rel_entry.path, use_cache=True).ravel()
        marker_color = self.read_image(res_path, use_cache=True).ravel()
        next_res = self.read_image(next_res_path, use_cache=True).ravel()
        if not (x.shape == y.shape == marker_color.shape == next_res.shape):
            return None
        grid_y, grid_x = np.indices(total_image.shape, dtype=float)

        valid, base_valid, density_gate = self.effective_scatter_valid_mask(
            rel_entry,
            x,
            y,
            marker_color,
            next_res,
        )

        return {
            "x": x,
            "y": y,
            "marker_color": marker_color,
            "valid": valid,
            "base_valid": base_valid,
            "res_density_gate": density_gate,
            "image_x": grid_x.ravel(),
            "image_y": grid_y.ravel(),
        }

    def concatenate_scatter_data(
        self,
        data_items: list[dict[str, np.ndarray]],
        *,
        max_points: int = MAX_AGGREGATED_SCATTER_POINTS,
    ) -> dict[str, np.ndarray] | None:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        base_valids: list[np.ndarray] = []
        density_gates: list[np.ndarray] = []
        for item in data_items:
            valid = item["valid"]
            if not np.any(valid):
                continue
            xs.append(item["x"][valid])
            ys.append(item["y"][valid])
            colors.append(item["marker_color"][valid])
            base_valids.append(np.ones(np.count_nonzero(valid), dtype=bool))
            density_gates.append(np.ones(np.count_nonzero(valid), dtype=bool))
        if not xs:
            return None

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        marker_color = np.concatenate(colors)
        base_valid = np.concatenate(base_valids)
        density_gate = np.concatenate(density_gates)
        if x.size > max_points:
            rng = np.random.default_rng(12345)
            idx = rng.choice(np.arange(x.size), size=max_points, replace=False)
            x = x[idx]
            y = y[idx]
            marker_color = marker_color[idx]
            base_valid = base_valid[idx]
            density_gate = density_gate[idx]

        valid = np.ones(x.shape, dtype=bool)
        return {
            "x": x,
            "y": y,
            "marker_color": marker_color,
            "valid": valid,
            "base_valid": base_valid,
            "res_density_gate": density_gate,
        }

    def scatter_indices_inside_map_roi(
        self,
        scatter_data: dict[str, np.ndarray],
        valid_idx: np.ndarray,
    ) -> np.ndarray:
        if self.map_roi is None or "image_x" not in scatter_data or "image_y" not in scatter_data:
            return np.empty(0, dtype=int)
        if valid_idx.size == 0:
            return valid_idx

        image_x = scatter_data["image_x"][valid_idx]
        image_y = scatter_data["image_y"][valid_idx]
        finite = np.isfinite(image_x) & np.isfinite(image_y)
        if not np.any(finite):
            return np.empty(0, dtype=int)

        roi_type = self.map_roi.get("type")
        selected = np.zeros(valid_idx.shape, dtype=bool)
        if roi_type == "circle":
            center_x, center_y = self.map_roi["center"]
            radius = float(self.map_roi["radius"])
            selected[finite] = (image_x[finite] - center_x) ** 2 + (image_y[finite] - center_y) ** 2 <= radius ** 2
        elif roi_type == "polygon":
            points = np.asarray(self.map_roi["points"], dtype=float)
            if points.shape[0] >= 3:
                selected[finite] = MplPath(points).contains_points(
                    np.column_stack((image_x[finite], image_y[finite]))
                )
        return valid_idx[selected]

    def draw_scatter_on_axis(
        self,
        fig: Figure,
        ax,
        rel_entry: ChangeEntry,
        scatter_data: dict[str, np.ndarray],
        title: str,
        *,
        point_size: float,
        register_selection: bool = False,
        selection_panel: str = "scatter",
        apply_map_roi: bool = False,
    ) -> int:
        x = scatter_data["x"]
        y = scatter_data["y"]
        marker_color = scatter_data["marker_color"]
        valid = scatter_data["valid"]
        base_valid = scatter_data.get("base_valid", valid)
        density_gate = scatter_data.get(
            "res_density_gate",
            np.ones(valid.shape, dtype=bool),
        )
        base_valid_idx = np.flatnonzero(base_valid)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            if self.res_density_gate_active() and base_valid_idx.size:
                ax.scatter(
                    x[base_valid_idx],
                    y[base_valid_idx],
                    color="#475569",
                    s=point_size,
                    alpha=0.35,
                    linewidths=0,
                )
                ax.text(
                    0.5,
                    0.5,
                    "No points in resistant-density gate",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xscale(self.config_axis_scale("relative_density_change_scatter_xscale"))
                ax.set_yscale(self.config_axis_scale("relative_density_change_scatter_yscale"))
                xlim = self.config_axis_limits("relative_density_change_scatter_xlim")
                ylim = self.scatter_y_axis_limits(rel_entry)
                if xlim is not None:
                    ax.set_xlim(*xlim)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.set_xlabel("Total density [cells/um^2]")
                ax.set_ylabel(self.current_profile_y_axis_label(rel_entry))
                ax.set_title(f"{title}\nn=0; base n={base_valid_idx.size:,}")
                ax.grid(True, color="0.9", linewidth=0.6)
                return 0
            ax.text(0.5, 0.5, "No gated points", ha="center", va="center")
            ax.set_axis_off()
            return 0

        if self.res_density_gate_active():
            outside_density_gate_idx = base_valid_idx[~density_gate[base_valid_idx]]
            if outside_density_gate_idx.size:
                ax.scatter(
                    x[outside_density_gate_idx],
                    y[outside_density_gate_idx],
                    color="#475569",
                    s=point_size,
                    alpha=0.32,
                    linewidths=0,
                    zorder=0,
                )

        order = valid_idx[np.argsort(marker_color[valid_idx])]
        color_max = self.scatter_colorbar_vmax(marker_color)
        color_norm = Normalize(vmin=0.0, vmax=color_max, clip=True)
        color_map = LinearSegmentedColormap.from_list(
            "black_to_goldenrod",
            ["#000000", "#DAA520"],
        )
        roi_idx = (
            self.scatter_indices_inside_map_roi(scatter_data, valid_idx)
            if apply_map_roi
            else np.empty(0, dtype=int)
        )
        if apply_map_roi and self.map_roi is not None:
            background_idx = np.setdiff1d(valid_idx, roi_idx, assume_unique=False)
            if background_idx.size:
                ax.scatter(
                    x[background_idx],
                    y[background_idx],
                    color="#64748b",
                    s=point_size,
                    alpha=0.45,
                    linewidths=0,
                    zorder=1,
                )
            order = roi_idx[np.argsort(marker_color[roi_idx])] if roi_idx.size else roi_idx
            scatter = ax.scatter(
                x[order],
                y[order],
                c=marker_color[order],
                cmap=color_map,
                norm=color_norm,
                s=max(point_size * 1.25, 3.0),
                alpha=1.0,
                linewidths=0,
                zorder=2,
            )
        else:
            scatter = ax.scatter(
                x[order],
                y[order],
                c=marker_color[order],
                cmap=color_map,
                norm=color_norm,
                s=point_size,
                alpha=1.0,
                linewidths=0,
            )
        if self.res_density_gate_active():
            ax.scatter(
                x[valid_idx],
                y[valid_idx],
                color="#16a34a",
                s=max(point_size * 1.8, 5.0),
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )
        summary_idx = (
            roi_idx
            if apply_map_roi and self.map_roi is not None
            else valid_idx
        )
        n_summary_bins = self.draw_scatter_binned_summary(
            ax,
            x,
            y,
            summary_idx,
        )
        ax.set_xscale(self.config_axis_scale("relative_density_change_scatter_xscale"))
        ax.set_yscale(self.config_axis_scale("relative_density_change_scatter_yscale"))
        xlim = self.config_axis_limits("relative_density_change_scatter_xlim")
        ylim = self.scatter_y_axis_limits(rel_entry)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Total density [cells/µm²]")
        ax.set_ylabel(self.current_profile_y_axis_label(rel_entry))
        gate_count = int(np.count_nonzero(base_valid & density_gate))
        if apply_map_roi and self.map_roi is not None:
            title_text = f"{title}\nn={valid_idx.size:,}; ROI n={roi_idx.size:,}"
        else:
            title_text = f"{title}\nn={valid_idx.size:,}"
        if self.res_density_gate_active():
            title_text += f"; res gate n={gate_count:,}"
        if n_summary_bins:
            title_text += f"; median bins={n_summary_bins}"
        ax.set_title(title_text)
        ax.grid(True, color="0.9", linewidth=0.6)
        if n_summary_bins:
            ax.legend(loc="upper right", fontsize=6, frameon=True)
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
        colorbar.set_label(f"Initial resistant density [cells/µm²]; vmax={color_max:.3g}")
        if register_selection:
            self.register_scatter_selector(
                ax,
                scatter_data,
                valid_idx=valid_idx,
                panel_id=selection_panel,
            )
        return int(valid_idx.size)

    def selected_line_scatter_title(self, interval_label: str) -> str:
        width_um = self.width_var.get() * self.pixel_size_um
        return f"Selected line band {interval_label}\nline width={width_um:.1f} um"

    def draw_line_scatterplot(
        self,
        fig: Figure,
        rel_entry: ChangeEntry,
        scatter_data: dict[str, np.ndarray],
    ) -> int:
        x = scatter_data["x"]
        y = scatter_data["y"]
        marker_color = scatter_data["marker_color"]
        valid = scatter_data["valid"]
        base_valid = scatter_data.get("base_valid", valid)
        density_gate = scatter_data.get(
            "res_density_gate",
            np.ones(valid.shape, dtype=bool),
        )
        base_valid_idx = np.flatnonzero(base_valid)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            return 0

        order = valid_idx[np.argsort(marker_color[valid_idx])]
        color_max = self.scatter_colorbar_vmax(marker_color)
        color_norm = Normalize(vmin=0.0, vmax=color_max, clip=True)
        color_map = LinearSegmentedColormap.from_list(
            "black_to_goldenrod",
            ["#000000", "#DAA520"],
        )

        ax = fig.add_subplot(111)
        if self.res_density_gate_active():
            outside_density_gate_idx = base_valid_idx[~density_gate[base_valid_idx]]
            if outside_density_gate_idx.size:
                ax.scatter(
                    x[outside_density_gate_idx],
                    y[outside_density_gate_idx],
                    color="#475569",
                    s=3,
                    alpha=0.32,
                    linewidths=0,
                    zorder=0,
                )
        scatter = ax.scatter(
            x[order],
            y[order],
            c=marker_color[order],
            cmap=color_map,
            norm=color_norm,
            s=3,
            alpha=1.0,
            linewidths=0,
        )
        if self.res_density_gate_active():
            ax.scatter(
                x[valid_idx],
                y[valid_idx],
                color="#16a34a",
                s=7,
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )
        xscale = self.config_axis_scale("relative_density_change_scatter_xscale")
        yscale = self.config_axis_scale("relative_density_change_scatter_yscale")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        xlim = self.config_axis_limits("relative_density_change_scatter_xlim")
        ylim = self.scatter_y_axis_limits(rel_entry)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(f"Total density at day {rel_entry.first_tp} [cells/µm²]")
        ax.set_ylabel(self.current_profile_y_axis_label(rel_entry))
        ax.set_title(
            f"{self.plate_var.get()} {self.well_var.get()}: "
            f"selected-line-band scatter {rel_entry.first_tp}->{rel_entry.next_tp}\n"
            f"n={valid_idx.size:,} gated points; width={self.width_var.get() * self.pixel_size_um:.1f} µm"
        )
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.text(
            0.02,
            0.98,
            self.scatter_gate_description(rel_entry),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.35",
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
        colorbar.set_label(
            f"Resistant density at day {rel_entry.first_tp} [cells/µm²]; "
            f"vmax={color_max:.3g}"
        )
        fig.tight_layout()
        return int(valid_idx.size)

    def clear_scatter_selectors(self) -> None:
        for selector in self.scatter_selectors:
            try:
                selector.set_active(False)
                selector.disconnect_events()
            except Exception:
                pass
        self.scatter_selectors = []
        self.scatter_selection_data = {}
        for artist in self.scatter_selection_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.scatter_selection_artists = []

    def clear_selected_scatter_points(
        self,
        *,
        redraw_maps: bool = True,
        clear_rectangle: bool = True,
    ) -> None:
        self.selected_scatter_points = None
        if clear_rectangle:
            self.scatter_selection_bounds = None
            self.scatter_selection_polygon = None
            self.scatter_selection_panel = None
            self.scatter_selection_shape = self.scatter_selection_shape_var.get()
        for artist in self.selected_overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.selected_overlay_artists = []
        if redraw_maps:
            self.image_canvas.draw_idle()
            self.reference_map_canvas.draw_idle()

    def register_scatter_selector(
        self,
        ax,
        scatter_data: dict[str, np.ndarray],
        *,
        valid_idx: np.ndarray,
        panel_id: str,
    ) -> None:
        if "image_x" not in scatter_data or "image_y" not in scatter_data:
            return
        self.scatter_selection_data[ax] = {
            "x": scatter_data["x"],
            "y": scatter_data["y"],
            "image_x": scatter_data["image_x"],
            "image_y": scatter_data["image_y"],
            "valid_idx": valid_idx,
            "panel_id": panel_id,
        }
        shape = self.scatter_selection_shape_var.get()
        if shape == "Polygon":
            selector = PolygonSelector(
                ax,
                lambda vertices, axis=ax: self.on_scatter_polygon_selected(axis, vertices),
                useblit=False,
            )
        else:
            selector_class = EllipseSelector if shape == "Ellipse" else RectangleSelector
            selector = selector_class(
                ax,
                lambda press, release, axis=ax: self.on_scatter_rectangle_selected(axis, press, release),
                useblit=False,
                button=[1],
                minspanx=0,
                minspany=0,
                spancoords="data",
                interactive=True,
            )
        self.scatter_selectors.append(selector)

    def scatter_indices_in_bounds(
        self,
        data: dict[str, np.ndarray],
        bounds: tuple[float, float, float, float],
    ) -> np.ndarray:
        xmin, xmax, ymin, ymax = bounds
        valid_idx = data["valid_idx"]
        x = data["x"]
        y = data["y"]
        in_bounds = (
            (x[valid_idx] >= xmin)
            & (x[valid_idx] <= xmax)
            & (y[valid_idx] >= ymin)
            & (y[valid_idx] <= ymax)
        )
        if self.scatter_selection_shape == "Polygon":
            if self.scatter_selection_polygon is None or self.scatter_selection_polygon.shape[0] < 3:
                return np.empty(0, dtype=int)
            points = np.column_stack((x[valid_idx], y[valid_idx]))
            return valid_idx[MplPath(self.scatter_selection_polygon).contains_points(points)]
        if self.scatter_selection_shape != "Ellipse":
            return valid_idx[in_bounds]

        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        radius_x = 0.5 * abs(xmax - xmin)
        radius_y = 0.5 * abs(ymax - ymin)
        if radius_x <= 0 or radius_y <= 0:
            return np.empty(0, dtype=int)
        ellipse_value = ((x[valid_idx] - center_x) / radius_x) ** 2 + (
            (y[valid_idx] - center_y) / radius_y
        ) ** 2
        return valid_idx[in_bounds & (ellipse_value <= 1.0)]

    def points_for_scatter_indices(
        self,
        data: dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        if indices.size == 0:
            return np.empty((0, 2), dtype=float)
        points = np.column_stack((data["image_x"][indices], data["image_y"][indices]))
        finite = np.isfinite(points).all(axis=1)
        return points[finite]

    def draw_scatter_selection_bounds_on_axis(
        self,
        ax,
        bounds: tuple[float, float, float, float],
    ) -> None:
        xmin, xmax, ymin, ymax = bounds
        if self.scatter_selection_shape == "Polygon":
            if self.scatter_selection_polygon is None or self.scatter_selection_polygon.shape[0] < 3:
                return
            artist = Polygon(
                self.scatter_selection_polygon,
                closed=True,
                fill=False,
                edgecolor="#ff2d75",
                linewidth=1.1,
                linestyle="--",
                zorder=8,
            )
        elif self.scatter_selection_shape == "Ellipse":
            artist = Ellipse(
                (0.5 * (xmin + xmax), 0.5 * (ymin + ymax)),
                width=xmax - xmin,
                height=ymax - ymin,
                fill=False,
                edgecolor="#ff2d75",
                linewidth=1.1,
                linestyle="--",
                zorder=8,
            )
        else:
            artist = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="#ff2d75",
                linewidth=1.1,
                linestyle="--",
                zorder=8,
            )
        ax.add_patch(artist)
        self.scatter_selection_artists.append(artist)

    def apply_persistent_scatter_selection(self, *, redraw_maps: bool = True) -> None:
        self.selected_scatter_points = None
        if self.scatter_selection_bounds is None or self.scatter_selection_panel is None:
            if redraw_maps:
                self.redraw_selected_scatter_overlays()
            return

        selected_points = None
        for ax, data in self.scatter_selection_data.items():
            if data.get("panel_id") != self.scatter_selection_panel:
                continue
            self.draw_scatter_selection_bounds_on_axis(ax, self.scatter_selection_bounds)
            selected_idx = self.scatter_indices_in_bounds(data, self.scatter_selection_bounds)
            selected_points = self.points_for_scatter_indices(data, selected_idx)
            break

        self.selected_scatter_points = selected_points
        if redraw_maps:
            self.redraw_selected_scatter_overlays()

    def apply_scatter_selection_to_axis(self, ax) -> None:
        data = self.scatter_selection_data.get(ax)
        if data is None or self.scatter_selection_bounds is None:
            return
        for artist in self.scatter_selection_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.scatter_selection_artists = []
        self.draw_scatter_selection_bounds_on_axis(ax, self.scatter_selection_bounds)
        selected_idx = self.scatter_indices_in_bounds(data, self.scatter_selection_bounds)
        if selected_idx.size == 0:
            self.clear_selected_scatter_points(clear_rectangle=False)
            self.status_var.set("No scatter points selected in the scatter gate.")
            self.scatter_canvas.draw_idle()
            return
        self.selected_scatter_points = self.points_for_scatter_indices(data, selected_idx)
        self.redraw_selected_scatter_overlays()
        self.scatter_canvas.draw_idle()
        self.status_var.set(f"Selected {self.selected_scatter_points.shape[0]:,} scatter points.")

    def on_scatter_rectangle_selected(self, ax, press, release) -> None:
        data = self.scatter_selection_data.get(ax)
        if data is None or press.xdata is None or press.ydata is None or release.xdata is None or release.ydata is None:
            return
        xmin, xmax = sorted((float(press.xdata), float(release.xdata)))
        ymin, ymax = sorted((float(press.ydata), float(release.ydata)))
        self.scatter_selection_bounds = (xmin, xmax, ymin, ymax)
        self.scatter_selection_polygon = None
        self.scatter_selection_panel = str(data.get("panel_id", ""))
        self.scatter_selection_shape = self.scatter_selection_shape_var.get()
        self.apply_scatter_selection_to_axis(ax)

    def on_scatter_polygon_selected(self, ax, vertices) -> None:
        data = self.scatter_selection_data.get(ax)
        if data is None:
            return
        points = np.asarray(vertices, dtype=float)
        finite = np.isfinite(points).all(axis=1) if points.ndim == 2 and points.shape[1] == 2 else []
        points = points[finite]
        if points.shape[0] < 3:
            self.status_var.set("Select at least three polygon vertices in the scatter gate.")
            return
        xmin = float(np.nanmin(points[:, 0]))
        xmax = float(np.nanmax(points[:, 0]))
        ymin = float(np.nanmin(points[:, 1]))
        ymax = float(np.nanmax(points[:, 1]))
        self.scatter_selection_bounds = (xmin, xmax, ymin, ymax)
        self.scatter_selection_polygon = points
        self.scatter_selection_panel = str(data.get("panel_id", ""))
        self.scatter_selection_shape = "Polygon"
        self.apply_scatter_selection_to_axis(ax)

    def draw_selected_points_on_axis(self, ax) -> None:
        if self.selected_scatter_points is None or self.selected_scatter_points.size == 0:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        artist = ax.scatter(
            self.selected_scatter_points[:, 0],
            self.selected_scatter_points[:, 1],
            s=5,
            color="#ff2d75",
            alpha=0.8,
            linewidths=0,
            zorder=6,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.selected_overlay_artists.append(artist)

    def clear_res_density_gate_overlays(self, ax=None) -> None:
        remaining = []
        for artist in self.res_density_gate_overlay_artists:
            if ax is not None and getattr(artist, "axes", None) is not ax:
                remaining.append(artist)
                continue
            try:
                artist.remove()
            except Exception:
                pass
        self.res_density_gate_overlay_artists = remaining

    def res_density_gate_points_for_entry(self, entry: ChangeEntry | None) -> np.ndarray:
        if entry is None or not self.res_density_gate_active():
            return np.empty((0, 2), dtype=float)
        res_path = self.density_map_path(entry, "res")
        if not res_path.exists():
            return np.empty((0, 2), dtype=float)
        try:
            res_image = self.read_cached_image(res_path)
        except Exception:
            return np.empty((0, 2), dtype=float)
        mask = self.res_density_gate_mask(res_image)
        if not np.any(mask):
            return np.empty((0, 2), dtype=float)
        y_idx, x_idx = np.nonzero(mask)
        if x_idx.size > 30000:
            step = int(math.ceil(x_idx.size / 30000))
            x_idx = x_idx[::step]
            y_idx = y_idx[::step]
        return np.column_stack((x_idx.astype(float), y_idx.astype(float)))

    def draw_res_density_gate_overlay_on_axis(
        self,
        ax,
        entry: ChangeEntry | None,
    ) -> None:
        points = self.res_density_gate_points_for_entry(entry)
        if points.size == 0:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        artist = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=4,
            color="#16a34a",
            alpha=0.42,
            linewidths=0,
            zorder=5,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.res_density_gate_overlay_artists.append(artist)

    def redraw_selected_scatter_overlays(self) -> None:
        for artist in self.selected_overlay_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.selected_overlay_artists = []
        self.draw_selected_points_on_axis(self.image_ax)
        self.draw_selected_points_on_axis(self.reference_map_ax)
        self.image_canvas.draw_idle()
        self.reference_map_canvas.draw_idle()

    def clear_roi_preview(self) -> None:
        for artist in self._roi_preview_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._roi_preview_artists = []

    def clear_map_roi(self, *, redraw: bool = True) -> None:
        self.map_roi = None
        self._roi_drag_start_xy = None
        self._roi_drag_axis = None
        self._roi_drag_canvas = None
        self._roi_points = []
        self.clear_roi_preview()
        if redraw:
            self.redraw_image()
            self.update_scatter_preview()
            self.status_var.set("Spatial ROI cleared.")

    def _on_map_roi_mode_changed(self) -> None:
        self.clear_roi_preview()
        if self.map_roi_enabled_var.get():
            self.status_var.set("ROI mode: drag on either map to draw a polygon or circle.")
        else:
            self.status_var.set("ROI mode off. Click or drag on maps to edit the profile line.")

    def _on_scatter_selection_shape_changed(self) -> None:
        for artist in self.scatter_selection_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.scatter_selection_artists = []
        self.clear_selected_scatter_points(redraw_maps=True)
        self.update_scatter_preview()
        self.status_var.set(f"Scatter gate shape set to {self.scatter_selection_shape_var.get()}.")

    def draw_map_roi_on_axis(self, ax) -> None:
        if self.map_roi is None:
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        roi_type = self.map_roi.get("type")
        if roi_type == "circle":
            artist = Circle(
                self.map_roi["center"],
                float(self.map_roi["radius"]),
                fill=False,
                edgecolor="#ff2d75",
                linewidth=1.2,
                linestyle="-",
                zorder=7,
            )
            ax.add_patch(artist)
        elif roi_type == "polygon":
            points = np.asarray(self.map_roi["points"], dtype=float)
            if points.shape[0] < 3:
                return
            artist = Polygon(
                points,
                closed=True,
                fill=False,
                edgecolor="#ff2d75",
                linewidth=1.2,
                linestyle="-",
                zorder=7,
            )
            ax.add_patch(artist)
        else:
            return
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.map_roi_artists.append(artist)

    def redraw_map_roi_overlays(self) -> None:
        for artist in self.map_roi_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.map_roi_artists = []
        self.draw_map_roi_on_axis(self.image_ax)
        self.draw_map_roi_on_axis(self.reference_map_ax)
        self.image_canvas.draw_idle()
        self.reference_map_canvas.draw_idle()

    def reset_map_zoom(self) -> None:
        self._stored_map_xlim = None
        self._stored_map_ylim = None

    def current_map_view_context(self) -> tuple[str, str, str, int, int] | None:
        image = self.display_image if self.display_image is not None else self.image
        if image is None or image.ndim < 2:
            return None
        height, width = image.shape[:2]
        try:
            run_dir = os.fspath(self.run_dir.resolve())
        except OSError:
            run_dir = os.fspath(self.run_dir)
        return run_dir, self.plate_var.get(), self.well_var.get(), int(height), int(width)

    def is_simulation_run(self) -> bool:
        if self._is_simulation_run_cache is not None:
            return self._is_simulation_run_cache
        if (self.run_dir / "simulation_density_run_manifest.tsv").exists():
            self._is_simulation_run_cache = True
            return True
        for plate_dir in self.run_dir.glob("Plate_*"):
            hdf5_dir = plate_dir / "hdf5"
            if hdf5_dir.exists() and any(hdf5_dir.glob("*_simulation.h5")):
                self._is_simulation_run_cache = True
                return True
        self._is_simulation_run_cache = False
        return False

    def display_rotation_k(self) -> int:
        if not self.is_simulation_run():
            return 0
        if "simulation_display_rotation_k" in self.config:
            value = int(self.config.get("simulation_display_rotation_k", 1))
        else:
            degrees = float(self.config.get("simulation_display_rotation_degrees", 90.0))
            value = int(round(degrees / 90.0))
        return value % 4

    def mirror_simulation_display_vertically(self) -> bool:
        if not self.is_simulation_run():
            return False
        return bool(self.config.get("simulation_display_mirror_y", True))

    def orient_image_for_display(self, image: np.ndarray) -> np.ndarray:
        k = self.display_rotation_k()
        if k != 0:
            image = np.rot90(image, k=k, axes=(0, 1))
        if self.mirror_simulation_display_vertically():
            image = np.flipud(image)
        return np.ascontiguousarray(image)

    def raw_image_shape(self, path: Path) -> tuple[int, int]:
        key = path.resolve()
        cached = self.raw_image_shape_cache.get(key)
        if cached is not None:
            return cached
        try:
            with tiff.TiffFile(path) as tif:
                shape = tuple(int(value) for value in tif.pages[0].shape[:2])
        except Exception:
            shape = tuple(int(value) for value in read_float_image(path).shape[:2])
        self.raw_image_shape_cache[key] = shape
        return shape

    def orient_xy_for_display(self, xy: np.ndarray, source_shape: tuple[int, int] | None) -> np.ndarray:
        k = self.display_rotation_k()
        mirror_y = self.mirror_simulation_display_vertically()
        if xy.size == 0 or source_shape is None or (k == 0 and not mirror_y):
            return xy
        height, width = source_shape
        x = xy[:, 0]
        y = xy[:, 1]
        if k == 1:
            rotated = np.column_stack((y, float(width) - 1.0 - x))
            rotated_width = height
            rotated_height = width
        elif k == 2:
            rotated = np.column_stack((float(width) - 1.0 - x, float(height) - 1.0 - y))
            rotated_width = width
            rotated_height = height
        else:
            rotated = np.column_stack((float(height) - 1.0 - y, x))
            rotated_width = height
            rotated_height = width
        if k == 0:
            rotated = xy.astype(float, copy=True)
            rotated_width = width
            rotated_height = height
        if mirror_y:
            rotated[:, 1] = float(rotated_height) - 1.0 - rotated[:, 1]
        return rotated.astype(float, copy=False)

    def read_image(self, path: Path, *, use_cache: bool = True) -> np.ndarray:
        if use_cache:
            return self.read_cached_image(path)
        return self.orient_image_for_display(read_float_image(path))

    def current_image_extent_limits(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        image = self.display_image if self.display_image is not None else self.image
        if image is None or image.ndim < 2:
            return None
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return None
        return (-0.5, float(width) - 0.5), (float(height) - 0.5, -0.5)

    def fit_maps_to_image_extent(self) -> None:
        limits = self.current_image_extent_limits()
        if limits is None:
            self.status_var.set("No image loaded to fit.")
            return

        self._stored_map_xlim, self._stored_map_ylim = limits
        was_syncing = self._syncing_map_limits
        self._syncing_map_limits = True
        try:
            for ax in (self.image_ax, self.reference_map_ax):
                if ax is None:
                    continue
                ax.set_xlim(*self._stored_map_xlim)
                ax.set_ylim(*self._stored_map_ylim)
                self.update_hdf5_marker_sizes(ax)
        finally:
            self._syncing_map_limits = was_syncing

        self.image_canvas.draw_idle()
        self.reference_map_canvas.draw_idle()
        self.status_var.set("Map view fitted to the full image.")

    def store_map_limits(self, ax) -> None:
        self._stored_map_xlim = tuple(float(value) for value in ax.get_xlim())
        self._stored_map_ylim = tuple(float(value) for value in ax.get_ylim())

    def apply_stored_map_limits(self, ax) -> None:
        if self._stored_map_xlim is None or self._stored_map_ylim is None:
            return
        was_syncing = self._syncing_map_limits
        self._syncing_map_limits = True
        try:
            ax.set_xlim(*self._stored_map_xlim)
            ax.set_ylim(*self._stored_map_ylim)
        finally:
            self._syncing_map_limits = was_syncing

    def connect_map_limit_sync(self, ax) -> None:
        if ax in self._map_limit_sync_axes:
            return
        self._map_limit_sync_axes.add(ax)
        ax.callbacks.connect("xlim_changed", self.on_map_limits_changed)
        ax.callbacks.connect("ylim_changed", self.on_map_limits_changed)

    def on_map_limits_changed(self, ax) -> None:
        if self._syncing_map_limits:
            return
        if ax is not self.image_ax and ax is not self.reference_map_ax:
            return
        self.store_map_limits(ax)
        self.update_hdf5_marker_sizes(ax)
        other_ax = self.reference_map_ax if ax is self.image_ax else self.image_ax
        other_canvas = self.reference_map_canvas if ax is self.image_ax else self.image_canvas
        if other_ax is None:
            return
        was_syncing = self._syncing_map_limits
        self._syncing_map_limits = True
        try:
            other_ax.set_xlim(*self._stored_map_xlim)
            other_ax.set_ylim(*self._stored_map_ylim)
            self.update_hdf5_marker_sizes(other_ax)
        finally:
            self._syncing_map_limits = was_syncing
        current_canvas = self.reference_map_canvas if ax is self.reference_map_ax else self.image_canvas
        current_canvas.draw_idle()
        other_canvas.draw_idle()

    def read_cached_image(self, path: Path) -> np.ndarray:
        key = path.resolve()
        cached = self.image_cache.get(key)
        if cached is not None:
            self.image_cache.move_to_end(key)
            return cached
        image = self.orient_image_for_display(read_float_image(path))
        self.image_cache[key] = image
        self.image_cache.move_to_end(key)
        while len(self.image_cache) > IMAGE_CACHE_LIMIT:
            self.image_cache.popitem(last=False)
        return image

    def read_cached_png_image(self, path: Path) -> np.ndarray:
        key = path.resolve()
        cached = self.image_cache.get(key)
        if cached is not None:
            self.image_cache.move_to_end(key)
            return cached
        image = mpl_imread(path)
        self.image_cache[key] = image
        self.image_cache.move_to_end(key)
        while len(self.image_cache) > IMAGE_CACHE_LIMIT:
            self.image_cache.popitem(last=False)
        return image

    def hdf5_timepoint_key(self, timepoint: str) -> str:
        text = str(timepoint).strip()
        if text.isdigit() and len(text) == 1:
            return f"{int(text):02d}"
        return text

    def hdf5_positions_for_timepoint(
        self,
        pointcloud_path: Path,
        timepoint: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        sus_positions, res_positions = pointcloud_read(
            os.fspath(pointcloud_path),
            self.hdf5_timepoint_key(timepoint),
            require_timepoint=True,
        )

        def _scaled_xy(frame) -> np.ndarray:
            if frame is None or len(frame) == 0:
                return np.empty((0, 2), dtype=float)
            xy = frame[["x", "y"]].to_numpy(dtype=float)
            xy = xy[np.isfinite(xy).all(axis=1)]
            return xy / float(self.coarsening_factor)

        source_shape = None
        density_path = self.density_map_path_for_timepoint(timepoint, "total")
        if density_path.exists():
            try:
                source_shape = self.raw_image_shape(density_path)
            except Exception:
                source_shape = None

        return (
            self.orient_xy_for_display(_scaled_xy(sus_positions), source_shape),
            self.orient_xy_for_display(_scaled_xy(res_positions), source_shape),
        )

    def hdf5_marker_size(self, ax, base_size: float) -> float:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        visible_width = abs(float(xlim[1]) - float(xlim[0]))
        visible_height = abs(float(ylim[1]) - float(ylim[0]))
        axis_box = ax.get_window_extent()
        if axis_box.width <= 0 or axis_box.height <= 0 or visible_width <= 0 or visible_height <= 0:
            zoom_factor = 1.0
        else:
            screen_px_per_data_px = min(
                float(axis_box.width) / visible_width,
                float(axis_box.height) / visible_height,
            )
            reference_px_per_data_px = float(
                self.config.get("hdf5_marker_reference_screen_px_per_data_px", 1.0)
            )
            if reference_px_per_data_px <= 0:
                reference_px_per_data_px = 1.0
            zoom_factor = max(1.0, screen_px_per_data_px / reference_px_per_data_px)
        max_size = float(self.config.get("hdf5_marker_max_size", 90.0))
        return min(max(base_size * (zoom_factor ** 1.2), base_size), max_size)

    def reference_timepoint_for_current_entry(self) -> str:
        entry = self.current_change or self.selected_entry()
        if entry is None:
            return ""
        time_choice = REFERENCE_TIME_CHOICES.get(self.reference_time_var.get(), "first")
        return entry.next_tp if time_choice == "next" else entry.first_tp

    def update_hdf5_marker_sizes(self, ax=None) -> None:
        remaining = []
        for item in self.hdf5_position_artists:
            artist = item.get("artist")
            artist_ax = getattr(artist, "axes", None)
            if artist_ax is None:
                continue
            if ax is not None and artist_ax is not ax:
                remaining.append(item)
                continue
            base_size = float(item.get("base_size", 1.0))
            artist.set_sizes(np.full(artist.get_offsets().shape[0], self.hdf5_marker_size(artist_ax, base_size)))
            remaining.append(item)
        self.hdf5_position_artists = remaining

    def plot_hdf5_cell_positions(
        self,
        ax,
        pointcloud_path: Path,
        timepoint: str,
    ) -> tuple[int, int]:
        sus_xy, res_xy = self.hdf5_positions_for_timepoint(pointcloud_path, timepoint)
        ax.set_facecolor("#05070A")
        self.hdf5_position_artists = [
            item
            for item in self.hdf5_position_artists
            if getattr(item.get("artist"), "axes", None) is not ax
        ]
        if sus_xy.size:
            base_size = 0.8
            artist = ax.scatter(
                sus_xy[:, 0],
                sus_xy[:, 1],
                s=self.hdf5_marker_size(ax, base_size),
                color="#5DA9FF",
                alpha=0.65,
                linewidths=0,
                rasterized=True,
                label="Sensitive",
            )
            self.hdf5_position_artists.append({"artist": artist, "base_size": base_size})
        if res_xy.size:
            base_size = 1.1
            artist = ax.scatter(
                res_xy[:, 0],
                res_xy[:, 1],
                s=self.hdf5_marker_size(ax, base_size),
                color="#FFD166",
                alpha=0.85,
                linewidths=0,
                rasterized=True,
                label="Resistant",
            )
            self.hdf5_position_artists.append({"artist": artist, "base_size": base_size})

        shape = self.cell_position_frame_coarse_shape(timepoint)
        if shape is not None:
            height, width = shape
            ax.set_xlim(-0.5, float(width) - 0.5)
            ax.set_ylim(float(height) - 0.5, -0.5)
        ax.set_aspect("equal", adjustable="box")
        self.update_hdf5_marker_sizes(ax)
        if sus_xy.size or res_xy.size:
            ax.legend(loc="upper right", fontsize=7, frameon=True)
        return int(sus_xy.shape[0]), int(res_xy.shape[0])

    def load_image(self) -> None:
        entry = self.selected_entry()
        if entry is None:
            self.image_ax.clear()
            self.image_canvas.draw_idle()
            self.clear_reference_map("Select an interval to preview a density map.")
            return
        try:
            self.image = self.read_cached_image(entry.path)
            bg_path = self.background_path(entry)
            self.display_image = self.read_cached_image(bg_path) if bg_path.exists() else self.image
        except Exception as exc:
            messagebox.showerror("Could not load image", str(exc))
            return

        self.current_change = entry
        kept_line = self.keep_line_var.get() and len(self.click_points) == 2
        if not kept_line:
            self.click_points.clear()
            self.current_profile = None
        self.redraw_image()
        if kept_line:
            self.update_profile(redraw_overlay=False)
            self.status_var.set(f"Loaded {entry.path.name}. Kept existing profile line.")
        else:
            if self.profile_colorbar is not None:
                try:
                    self.profile_colorbar.remove()
                except Exception:
                    pass
                self.profile_colorbar = None
            self.profile_fig.clear()
            self.profile_ax = self.profile_fig.add_subplot(111)
            self.profile_canvas.draw_idle()
            self.clear_scatter_preview("Draw a line to preview scatterplots.")
            if self.center_to_edge_var.get():
                self.status_var.set(f"Loaded {entry.path.name}. Click or drag for center-to-edge direction.")
            else:
                self.status_var.set(f"Loaded {entry.path.name}. Click two points or drag to draw the profile line.")

    def redraw_image(self) -> None:
        self.clear_lut_textboxes(self.image_fig)
        if self.image_colorbar is not None:
            try:
                self.image_colorbar.remove()
            except Exception:
                pass
            self.image_colorbar = None
        self.clear_res_density_gate_overlays(self.image_ax)
        was_syncing = self._syncing_map_limits
        self._syncing_map_limits = True
        self._map_limit_sync_axes.discard(self.image_ax)
        self.image_ax.clear()
        if self.display_image is None:
            self._syncing_map_limits = was_syncing
            self.image_canvas.draw_idle()
            return
        map_view_context = self.current_map_view_context()
        if map_view_context != self._map_view_context:
            # A zoom window is useful while moving through intervals in one well,
            # but it is invalid for another run/well or a differently sized map.
            # In particular, reusing 320-um limits on a 40-um map displays only
            # a small fraction of the colony.
            self.reset_map_zoom()
            self._map_view_context = map_view_context
        image_artist = None
        lut_key = None
        try:
            shared_limits = (
                self.robust_limits_for_paths(self.background_limit_paths(self.current_change))
                if self.current_change is not None
                else None
            )
            auto_limits = shared_limits if shared_limits is not None else robust_limits(self.display_image)
            lut_key = self.main_map_lut_key()
            vmin, vmax = self.effective_lut_limits(lut_key, auto_limits)
            cmap = "viridis" if BACKGROUND_CHOICES.get(self.background_var.get()) != "change" else "coolwarm"
            image_artist = self.image_ax.imshow(
                self.display_image,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="upper",
            )
            self.image_colorbar = self.image_fig.colorbar(
                image_artist,
                ax=self.image_ax,
                fraction=0.046,
                pad=0.04,
            )
            self.image_colorbar.set_label(self.image_colorbar_label(self.current_change))
            self.image_ax.set_title(
                f"{self.plate_var.get()} / {self.well_var.get()} / {self.interval_var.get()}\n"
                f"{self.change_type_var.get()}"
            )
            self.image_ax.set_xlabel("x [coarse px]")
            self.image_ax.set_ylabel("y [coarse px]")
            self.line_artist = None
            self.width_artists = []
            if len(self.click_points) == 2:
                self.draw_line_overlay()
            self.draw_map_roi_on_axis(self.image_ax)
            self.draw_res_density_gate_overlay_on_axis(self.image_ax, self.current_change)
            self.draw_selected_points_on_axis(self.image_ax)
            if self._stored_map_xlim is None or self._stored_map_ylim is None:
                self.store_map_limits(self.image_ax)
            else:
                self.apply_stored_map_limits(self.image_ax)
        finally:
            self._syncing_map_limits = was_syncing
        self.connect_map_limit_sync(self.image_ax)
        self.image_fig.tight_layout()
        if self.image_colorbar is not None and image_artist is not None and lut_key is not None:
            self.add_lut_textboxes(self.image_colorbar, image_artist, lut_key, auto_limits, self.image_canvas)
        self.image_canvas.draw_idle()
        self.update_reference_map()

    def draw_line_overlay(self) -> None:
        if len(self.click_points) != 2:
            return
        (x0, y0), (x1, y1) = self.click_points
        self.line_artist = self.image_ax.plot([x0, x1], [y0, y1], color="white", linewidth=1.2)[0]
        self.image_ax.scatter([x0, x1], [y0, y1], color="white", edgecolor="black", s=30, zorder=4)

        width = self.width_var.get()
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length > 0 and width > 1:
            nx = -dy / length
            ny = dx / length
            half = width / 2.0
            for sign in (-1, 1):
                ox = sign * half * nx
                oy = sign * half * ny
                artist = self.image_ax.plot(
                    [x0 + ox, x1 + ox],
                    [y0 + oy, y1 + oy],
                    color="white",
                    linewidth=0.7,
                    alpha=0.6,
                    linestyle="--",
                )[0]
                self.width_artists.append(artist)

    def draw_line_overlay_on_axis(self, ax) -> None:
        if len(self.click_points) != 2:
            return
        (x0, y0), (x1, y1) = self.click_points
        ax.plot([x0, x1], [y0, y1], color="white", linewidth=1.1)
        ax.scatter([x0, x1], [y0, y1], color="white", edgecolor="black", s=20, zorder=4)

        width = self.width_var.get()
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length > 0 and width > 1:
            nx = -dy / length
            ny = dx / length
            half = width / 2.0
            for sign in (-1, 1):
                ox = sign * half * nx
                oy = sign * half * ny
                ax.plot(
                    [x0 + ox, x1 + ox],
                    [y0 + oy, y1 + oy],
                    color="white",
                    linewidth=0.6,
                    alpha=0.6,
                    linestyle="--",
                )

    def clear_reference_map(self, message: str) -> None:
        self.clear_lut_textboxes(self.reference_map_fig)
        self.clear_res_density_gate_overlays(self.reference_map_ax)
        if self.reference_map_colorbar is not None:
            try:
                self.reference_map_colorbar.remove()
            except Exception:
                pass
            self.reference_map_colorbar = None
        self.clear_res_density_gate_overlays(self.reference_map_ax)
        self.reference_map_fig.clear()
        ax = self.reference_map_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        self.reference_map_canvas.draw_idle()

    def update_reference_map(self) -> None:
        entry = self.current_change or self.selected_entry()
        if entry is None:
            self.clear_reference_map("Select an interval to preview a density map.")
            return

        path, channel, timepoint = self.reference_map_path_and_label(entry)
        is_cell_position_frame = channel == "cell_positions"
        if not path.exists():
            missing_label = "cell-position HDF5" if is_cell_position_frame else "density map"
            self.clear_reference_map(f"Missing {missing_label}:\n{path.name}")
            return
        if not is_cell_position_frame:
            try:
                image = self.read_cached_image(path)
            except Exception as exc:
                self.clear_reference_map(f"Could not load extra map:\n{exc}")
                return

        self.clear_lut_textboxes(self.reference_map_fig)
        if self.reference_map_colorbar is not None:
            try:
                self.reference_map_colorbar.remove()
            except Exception:
                pass
            self.reference_map_colorbar = None
        self.reference_map_fig.clear()
        self.reference_map_ax = self.reference_map_fig.add_subplot(111)
        reference_artist = None
        reference_lut_key = None
        if is_cell_position_frame:
            try:
                n_sus, n_res = self.plot_hdf5_cell_positions(
                    self.reference_map_ax,
                    path,
                    timepoint,
                )
            except Exception as exc:
                self.clear_reference_map(f"Could not plot HDF5 cell positions:\n{exc}")
                return
        else:
            shared_limits = self.robust_limits_for_paths(self.reference_map_limit_paths(entry))
            auto_limits = shared_limits if shared_limits is not None else robust_limits(image)
            reference_lut_key = self.reference_map_lut_key(channel)
            vmin, vmax = self.effective_lut_limits(reference_lut_key, auto_limits)
            reference_artist = self.reference_map_ax.imshow(
                image,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                origin="upper",
            )
            self.reference_map_colorbar = self.reference_map_fig.colorbar(
                reference_artist,
                ax=self.reference_map_ax,
                fraction=0.046,
                pad=0.04,
            )
            self.reference_map_colorbar.set_label(self.density_channel_label(channel))
        self.reference_map_ax.set_title(
            f"{self.reference_map_var.get()} day {timepoint}\n"
            f"{self.plate_var.get()} / {self.well_var.get()}"
        )
        if is_cell_position_frame:
            self.reference_map_ax.set_title(
                f"{self.reference_map_var.get()} day {timepoint}\n"
                f"{self.plate_var.get()} / {self.well_var.get()} "
                f"({n_sus:,} sensitive, {n_res:,} resistant)"
            )
        self.reference_map_ax.set_xlabel("x [coarse px]")
        self.reference_map_ax.set_ylabel("y [coarse px]")
        self.draw_line_overlay_on_axis(self.reference_map_ax)
        self.draw_map_roi_on_axis(self.reference_map_ax)
        self.draw_res_density_gate_overlay_on_axis(self.reference_map_ax, entry)
        self.draw_selected_points_on_axis(self.reference_map_ax)
        self.apply_stored_map_limits(self.reference_map_ax)
        self.update_hdf5_marker_sizes(self.reference_map_ax)
        self.connect_map_limit_sync(self.reference_map_ax)
        self.reference_map_fig.tight_layout()
        if (
            self.reference_map_colorbar is not None
            and reference_artist is not None
            and reference_lut_key is not None
        ):
            self.add_lut_textboxes(
                self.reference_map_colorbar,
                reference_artist,
                reference_lut_key,
                auto_limits,
                self.reference_map_canvas,
            )
        self.reference_map_canvas.draw_idle()

    def image_toolbar_active(self) -> bool:
        mode = str(getattr(getattr(self, "image_toolbar", None), "mode", "")).strip().lower()
        return bool(mode) and mode not in {"none", "_mode.none"}

    def event_map_axis_and_canvas(self, event):
        if event.inaxes == self.image_ax:
            return self.image_ax, self.image_canvas
        if event.inaxes == self.reference_map_ax:
            return self.reference_map_ax, self.reference_map_canvas
        return None, None

    def image_event_xy(self, event) -> tuple[float, float] | None:
        axis, _canvas = self.event_map_axis_and_canvas(event)
        if axis is None or event.xdata is None or event.ydata is None:
            return None
        if self.image is None:
            return None
        return float(event.xdata), float(event.ydata)

    def is_left_mouse_button(self, button) -> bool:
        return button in (1, None) or str(button) == "MouseButton.LEFT"

    def event_canvas(self, event):
        _axis, canvas = self.event_map_axis_and_canvas(event)
        return canvas or self.image_canvas

    def remove_drag_preview(self) -> None:
        for artist in self._drag_preview_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._drag_preview_artists = []

    def draw_roi_preview(self) -> None:
        self.clear_roi_preview()
        axis = self._roi_drag_axis or self.image_ax
        canvas = self._roi_drag_canvas or self.image_canvas
        if self.map_roi_shape_var.get() == "Circle":
            if self._roi_drag_start_xy is None or len(self._roi_points) < 2:
                return
            center = self._roi_drag_start_xy
            target = self._roi_points[-1]
            radius = math.hypot(target[0] - center[0], target[1] - center[1])
            artist = Circle(
                center,
                radius,
                fill=False,
                edgecolor="#ff2d75",
                linewidth=1.1,
                linestyle="--",
                zorder=8,
            )
            axis.add_patch(artist)
            self._roi_preview_artists.append(artist)
        else:
            if len(self._roi_points) < 2:
                return
            points = np.asarray(self._roi_points, dtype=float)
            artist = axis.plot(
                points[:, 0],
                points[:, 1],
                color="#ff2d75",
                linewidth=1.1,
                linestyle="--",
                zorder=8,
            )[0]
            self._roi_preview_artists.append(artist)
        canvas.draw_idle()

    def on_map_roi_press(self, event) -> None:
        clicked = self.image_event_xy(event)
        if clicked is None:
            return
        if not self.is_left_mouse_button(getattr(event, "button", 1)):
            return
        axis, canvas = self.event_map_axis_and_canvas(event)
        self._roi_drag_start_xy = clicked
        self._roi_drag_axis = axis
        self._roi_drag_canvas = canvas
        self._roi_points = [clicked]
        self.clear_roi_preview()

    def on_map_roi_motion(self, event) -> None:
        if self._roi_drag_start_xy is None:
            return
        target = self.image_event_xy(event)
        if target is None:
            return
        if self.map_roi_shape_var.get() == "Circle":
            self._roi_points = [self._roi_drag_start_xy, target]
        else:
            if not self._roi_points or math.hypot(
                target[0] - self._roi_points[-1][0],
                target[1] - self._roi_points[-1][1],
            ) >= 0.75:
                self._roi_points.append(target)
        self.draw_roi_preview()

    def on_map_roi_release(self, event) -> None:
        if self._roi_drag_start_xy is None:
            return
        target = self.image_event_xy(event)
        canvas = self._roi_drag_canvas
        start_xy = self._roi_drag_start_xy
        points = list(self._roi_points)
        self._roi_drag_start_xy = None
        self._roi_drag_axis = None
        self._roi_drag_canvas = None
        self._roi_points = []
        self.clear_roi_preview()

        if target is not None:
            points.append(target)

        if self.map_roi_shape_var.get() == "Circle":
            if target is None:
                return
            radius = math.hypot(target[0] - start_xy[0], target[1] - start_xy[1])
            if radius < 1:
                self.status_var.set("Drag at least one pixel to draw a circular ROI.")
                if canvas is not None:
                    canvas.draw_idle()
                return
            self.map_roi = {"type": "circle", "center": start_xy, "radius": float(radius)}
        else:
            polygon_points = np.asarray(points, dtype=float)
            if polygon_points.shape[0] < 3:
                self.status_var.set("Drag a closed area with at least three points for a polygon ROI.")
                if canvas is not None:
                    canvas.draw_idle()
                return
            self.map_roi = {"type": "polygon", "points": polygon_points}

        self.redraw_image()
        self.update_scatter_preview()
        self.status_var.set("Spatial ROI applied to scatter preview.")

    def line_points_for_target(
        self,
        start_xy: tuple[float, float],
        target_xy: tuple[float, float],
    ) -> list[tuple[float, float]]:
        if self.center_to_edge_var.get():
            if self.image is None:
                return []
            height, width = self.image.shape
            center = ((width - 1) / 2.0, (height - 1) / 2.0)
            edge = line_to_image_edge(self.image.shape, center, target_xy)
            return [center, edge]
        return [start_xy, target_xy]

    def nearest_line_endpoint_index(
        self,
        clicked: tuple[float, float],
        event_xy: tuple[float, float],
        *,
        axis=None,
        max_distance_px: float = 12.0,
    ) -> int | None:
        if len(self.click_points) != 2:
            return None
        axis = axis or self.image_ax
        endpoint_distances = []
        for index, point in enumerate(self.click_points):
            screen_x, screen_y = axis.transData.transform(point)
            endpoint_distances.append((math.hypot(event_xy[0] - screen_x, event_xy[1] - screen_y), index))
        distance, index = min(endpoint_distances, key=lambda item: item[0])
        if distance <= max_distance_px:
            return index

        data_distances = [
            (math.hypot(clicked[0] - point[0], clicked[1] - point[1]), index)
            for index, point in enumerate(self.click_points)
        ]
        data_distance, data_index = min(data_distances, key=lambda item: item[0])
        return data_index if data_distance <= 2.0 else None

    def line_points_for_endpoint_drag(
        self,
        endpoint_index: int,
        target_xy: tuple[float, float],
    ) -> list[tuple[float, float]]:
        points = list(self.click_points)
        if len(points) != 2 or endpoint_index not in (0, 1):
            return points
        points[endpoint_index] = target_xy
        return points

    def draw_drag_preview(self, points: list[tuple[float, float]], *, axis=None, canvas=None) -> None:
        self.remove_drag_preview()
        if len(points) != 2:
            return
        axis = axis or self.image_ax
        canvas = canvas or self.image_canvas
        (x0, y0), (x1, y1) = points
        self._drag_preview_artists.append(
            axis.plot([x0, x1], [y0, y1], color="#ffea00", linewidth=1.4, zorder=5)[0]
        )
        self._drag_preview_artists.append(
            axis.scatter([x0, x1], [y0, y1], color="#ffea00", edgecolor="black", s=32, zorder=6)
        )

        width = self.width_var.get()
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length > 0 and width > 1:
            nx = -dy / length
            ny = dx / length
            half = width / 2.0
            for sign in (-1, 1):
                ox = sign * half * nx
                oy = sign * half * ny
                self._drag_preview_artists.append(
                    axis.plot(
                        [x0 + ox, x1 + ox],
                        [y0 + oy, y1 + oy],
                        color="#ffea00",
                        linewidth=0.7,
                        alpha=0.75,
                        linestyle="--",
                        zorder=5,
                    )[0]
                )
        canvas.draw_idle()

    def handle_image_click(self, clicked: tuple[float, float]) -> None:
        if self.center_to_edge_var.get():
            height, width = self.image.shape
            center = ((width - 1) / 2.0, (height - 1) / 2.0)
            try:
                edge = line_to_image_edge(self.image.shape, center, clicked)
            except ValueError as exc:
                messagebox.showerror("Invalid direction", str(exc))
                return
            self.click_points = [center, edge]
        else:
            if len(self.click_points) >= 2:
                self.click_points.clear()
            self.click_points.append(clicked)
        self.redraw_image()
        if len(self.click_points) == 2:
            self.update_profile(redraw_overlay=False)
        else:
            self.status_var.set("First point selected. Click an endpoint or drag to draw the profile line.")

    def on_image_click(self, event) -> None:
        if self.image_toolbar_active():
            return
        clicked = self.image_event_xy(event)
        if clicked is None:
            return
        self.handle_image_click(clicked)

    def on_image_press(self, event) -> None:
        if self.image_toolbar_active():
            return
        if self.map_roi_enabled_var.get():
            self.on_map_roi_press(event)
            return
        clicked = self.image_event_xy(event)
        if clicked is None:
            return
        axis, canvas = self.event_map_axis_and_canvas(event)
        button = getattr(event, "button", 1)
        if not self.is_left_mouse_button(button):
            return
        self._drag_start_xy = clicked
        self._drag_start_canvas_xy = (float(event.x), float(event.y))
        self._drag_axis = axis
        self._drag_canvas = canvas
        self._drag_endpoint_index = self.nearest_line_endpoint_index(
            clicked,
            self._drag_start_canvas_xy,
            axis=axis,
        )
        self._drag_moved = False
        self.remove_drag_preview()

    def on_image_motion(self, event) -> None:
        if self.map_roi_enabled_var.get() and self._roi_drag_start_xy is not None:
            self.on_map_roi_motion(event)
            return
        if self._drag_start_xy is None or self.image_toolbar_active():
            return
        target = self.image_event_xy(event)
        if target is None:
            return
        if self._drag_start_canvas_xy is not None:
            dx = float(event.x) - self._drag_start_canvas_xy[0]
            dy = float(event.y) - self._drag_start_canvas_xy[1]
            if math.hypot(dx, dy) >= 4:
                self._drag_moved = True
        if not self._drag_moved:
            return
        if self._drag_endpoint_index is not None:
            points = self.line_points_for_endpoint_drag(self._drag_endpoint_index, target)
        else:
            try:
                points = self.line_points_for_target(self._drag_start_xy, target)
            except ValueError:
                return
        self.draw_drag_preview(points, axis=self._drag_axis, canvas=self._drag_canvas)

    def on_image_release(self, event) -> None:
        if self.map_roi_enabled_var.get() and self._roi_drag_start_xy is not None:
            self.on_map_roi_release(event)
            return
        if self._drag_start_xy is None or self.image_toolbar_active():
            self._drag_start_xy = None
            self._drag_start_canvas_xy = None
            self._drag_axis = None
            self._drag_canvas = None
            self._drag_endpoint_index = None
            self._drag_moved = False
            self.remove_drag_preview()
            return
        target = self.image_event_xy(event)
        start_xy = self._drag_start_xy
        canvas = self._drag_canvas or self.event_canvas(event)
        endpoint_index = self._drag_endpoint_index
        moved = self._drag_moved
        self._drag_start_xy = None
        self._drag_start_canvas_xy = None
        self._drag_axis = None
        self._drag_canvas = None
        self._drag_endpoint_index = None
        self._drag_moved = False
        self.remove_drag_preview()

        if target is None:
            canvas.draw_idle()
            return
        if not moved:
            if endpoint_index is not None:
                self.status_var.set("Endpoint selected. Drag it to shift only this point.")
                canvas.draw_idle()
                return
            self.handle_image_click(target)
            return

        if endpoint_index is not None:
            points = self.line_points_for_endpoint_drag(endpoint_index, target)
        else:
            try:
                points = self.line_points_for_target(start_xy, target)
            except ValueError as exc:
                messagebox.showerror("Invalid direction", str(exc))
                canvas.draw_idle()
                return
        if len(points) != 2:
            return
        if math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]) < 1:
            self.status_var.set("Drag at least one pixel to draw a profile line.")
            canvas.draw_idle()
            return
        self.click_points = points
        self.redraw_image()
        self.update_profile(redraw_overlay=False)

    def clear_line(self) -> None:
        self.remove_drag_preview()
        self._drag_start_xy = None
        self._drag_start_canvas_xy = None
        self._drag_axis = None
        self._drag_canvas = None
        self._drag_endpoint_index = None
        self._drag_moved = False
        self.click_points.clear()
        self.current_profile = None
        self.redraw_image()
        if self.profile_colorbar is not None:
            try:
                self.profile_colorbar.remove()
            except Exception:
                pass
            self.profile_colorbar = None
        self.profile_fig.clear()
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_canvas.draw_idle()
        self.clear_scatter_preview("Draw a line to preview scatterplots.")
        self.status_var.set("Line cleared. Click two points or drag to define a new profile.")

    def clear_scatter_preview(self, message: str) -> None:
        self.clear_scatter_selectors()
        self.clear_selected_scatter_points(redraw_maps=False)
        self.scatter_fig.clear()
        ax = self.scatter_fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        self.scatter_canvas.draw_idle()

    def update_scatter_preview(self) -> None:
        self.clear_scatter_selectors()
        self.clear_selected_scatter_points(redraw_maps=False, clear_rectangle=False)
        line = self.current_line_points()
        if line is None:
            self.clear_scatter_preview("Draw a line to preview scatterplots.")
            return
        scatter_entry = self.selected_entry()
        if scatter_entry is None:
            self.clear_scatter_preview("Select a change map for this interval.")
            return

        start_xy, end_xy = line
        try:
            line_data = self.line_scatter_data(scatter_entry, start_xy, end_xy)
            full_well_data = self.full_well_scatter_data(scatter_entry)
        except Exception as exc:
            self.clear_scatter_preview(f"Could not build scatter preview:\n{exc}")
            return
        if line_data is None and full_well_data is None:
            self.clear_scatter_preview("Missing density maps for scatter preview.")
            return

        self.scatter_fig.clear()
        axes = self.scatter_fig.subplots(1, 2)
        interval_label = f"{scatter_entry.first_tp}->{scatter_entry.next_tp}"
        n_full = 0
        n_line = 0
        if full_well_data is not None:
            n_full = self.draw_scatter_on_axis(
                self.scatter_fig,
                axes[0],
                scatter_entry,
                full_well_data,
                f"Full well {interval_label}",
                point_size=2,
                register_selection=True,
                selection_panel="full",
                apply_map_roi=True,
            )
        else:
            axes[0].text(0.5, 0.5, "No full-well scatter data", ha="center", va="center")
            axes[0].set_axis_off()

        if line_data is not None:
            n_line = self.draw_scatter_on_axis(
                self.scatter_fig,
                axes[1],
                scatter_entry,
                line_data,
                self.selected_line_scatter_title(interval_label),
                point_size=4,
                register_selection=True,
                selection_panel="line",
                apply_map_roi=True,
            )
        else:
            axes[1].text(0.5, 0.5, "No selected-line scatter data", ha="center", va="center")
            axes[1].set_axis_off()

        if n_full == 0 and n_line == 0:
            self.clear_scatter_preview("No points pass the gate for the selected change map.")
            return
        self.apply_persistent_scatter_selection(redraw_maps=True)
        self.scatter_fig.tight_layout()
        self.scatter_canvas.draw_idle()

    def limits_from_arrays(
        self,
        arrays: list[np.ndarray],
        *,
        pad_fraction: float = 0.0,
    ) -> tuple[float, float] | None:
        finite_arrays = [array[np.isfinite(array)] for array in arrays if np.isfinite(array).any()]
        if not finite_arrays:
            return None
        values = np.concatenate(finite_arrays)
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        if lower == upper:
            upper = lower + 1.0
        if pad_fraction > 0:
            pad = pad_fraction * (upper - lower)
            lower -= pad
            upper += pad
        return lower, upper

    def profile_plot_limits_for_line(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if self.current_change is None:
            return None, None
        entries = [
            entry
            for entry in self._available_changes()
            if entry.kind == self.current_change.kind
        ]
        profile_values: list[np.ndarray] = []
        total_values: list[np.ndarray] = []
        for entry in entries:
            try:
                image = self.read_cached_image(entry.path)
                profile = sample_line_profile(
                    image,
                    start_xy,
                    end_xy,
                    self.width_var.get(),
                    self.pixel_size_um,
                )
                profile_values.append(profile.values)
                total_profile = self.total_density_profile_for_entry(
                    entry,
                    start_xy,
                    end_xy,
                )
                if total_profile is not None:
                    total_values.append(total_profile.values)
            except Exception as exc:
                print(f"[WARN] Skipping profile limit data for {entry.path}: {exc}")
                continue
        return (
            self.limits_from_arrays(profile_values, pad_fraction=0.05),
            self.limits_from_arrays(total_values),
        )

    def update_profile(self, *, redraw_overlay: bool = True) -> None:
        if self.image is None or len(self.click_points) != 2:
            return
        try:
            profile = sample_line_profile(
                self.image,
                self.click_points[0],
                self.click_points[1],
                self.width_var.get(),
                self.pixel_size_um,
            )
        except Exception as exc:
            messagebox.showerror("Could not sample profile", str(exc))
            return
        self.current_profile = profile
        if self.profile_colorbar is not None:
            try:
                self.profile_colorbar.remove()
            except Exception:
                pass
            self.profile_colorbar = None
        self.profile_fig.clear()
        self.profile_ax = self.profile_fig.add_subplot(111)
        total_profile = (
            self.total_density_profile_for_entry(
                self.current_change,
                profile.start_xy,
                profile.end_xy,
            )
            if self.current_change is not None
            else None
        )
        resistant_fraction = (
            self.resistant_fraction_profile_for_entry(
                self.current_change,
                profile.start_xy,
                profile.end_xy,
            )
            if self.current_change is not None
            else None
        )
        res_profile = (
            self.density_profile_for_entry(
                self.current_change,
                "res",
                self.current_change.first_tp,
                profile.start_xy,
                profile.end_xy,
            )
            if self.current_change is not None
            else None
        )
        shared_ylim, shared_total_ylim = self.profile_plot_limits_for_line(
            profile.start_xy,
            profile.end_xy,
        )
        self.profile_ax.plot(
            profile.distance_um,
            profile.values,
            color="#111111",
            linewidth=1.25,
        )
        if (
            total_profile is not None
            and total_profile.values.shape == profile.values.shape
            and resistant_fraction is not None
            and resistant_fraction.shape == total_profile.values.shape
        ):
            total_ax = self.profile_ax.twinx()
            line = colored_profile_line(
                total_ax,
                profile.distance_um,
                total_profile.values,
                resistant_fraction,
                cmap=COMPOSITION_CMAP,
                linewidth=1.8,
                color_limits=(0.0, 1.0),
            )
            cax = self.profile_fig.add_axes([0.88, 0.20, 0.025, 0.62])
            self.profile_colorbar = self.profile_fig.colorbar(
                line,
                cax=cax,
            )
            self.profile_colorbar.set_label("Resistant fraction of density (sensitive = 1 - fraction)")
            total_ax.set_ylabel("Total density [cells/ÂµmÂ²]")
            total_ax.set_xlim(float(profile.distance_um[0]), float(profile.distance_um[-1]))
            if shared_total_ylim is not None:
                total_ax.set_ylim(*shared_total_ylim)
            elif np.isfinite(total_profile.values).any():
                total_ymin = float(np.nanmin(total_profile.values))
                total_ymax = float(np.nanmax(total_profile.values))
                if total_ymin == total_ymax:
                    total_ymax = total_ymin + 1.0
                total_pad = 0.05 * (total_ymax - total_ymin)
                total_ax.set_ylim(total_ymin - total_pad, total_ymax + total_pad)
        elif total_profile is not None and total_profile.values.shape == profile.values.shape:
            total_ax = self.profile_ax.twinx()
            total_ax.plot(
                profile.distance_um,
                total_profile.values,
                color="#4f6d7a",
                linewidth=1.5,
                alpha=0.9,
            )
            total_ax.set_ylabel("Total density [cells/ÂµmÂ²]")
            total_ax.set_xlim(float(profile.distance_um[0]), float(profile.distance_um[-1]))
            if shared_total_ylim is not None:
                total_ax.set_ylim(*shared_total_ylim)
        total_profile = None
        if total_profile is not None and total_profile.values.shape == profile.values.shape:
            line = colored_profile_line(
                self.profile_ax,
                profile.distance_um,
                profile.values,
                total_profile.values,
                color_limits=shared_color_limits,
            )
            self.profile_colorbar = self.profile_fig.colorbar(
                line,
                ax=self.profile_ax,
                fraction=0.046,
                pad=0.04,
            )
            self.profile_colorbar.set_label("Total density [cells/µm²]")
        else:
            pass
        line_length_um = float(profile.distance_um[-1]) if profile.distance_um.size else 0.0
        self.profile_ax.set_xlabel("Distance along selected line [µm]")
        self.profile_ax.set_ylabel(self.y_axis_label(self.current_change))
        self.profile_ax.set_title(f"Line length: {line_length_um:.1f} µm")
        self.profile_ax.grid(True, color="0.9", linewidth=0.6)
        self.profile_ax.set_xlim(float(profile.distance_um[0]), float(profile.distance_um[-1]))
        if shared_ylim is not None:
            self.profile_ax.set_ylim(*shared_ylim)
        elif np.isfinite(profile.values).any():
            ymin = float(np.nanmin(profile.values))
            ymax = float(np.nanmax(profile.values))
            if ymin == ymax:
                ymax = ymin + 1.0
            pad = 0.05 * (ymax - ymin)
            self.profile_ax.set_ylim(ymin - pad, ymax + pad)
        if (
            self.res_density_gate_active()
            and res_profile is not None
            and res_profile.values.shape == profile.values.shape
        ):
            gate_mask = self.res_density_gate_mask(res_profile.values)
            gate_mask &= np.isfinite(profile.values)
            if np.any(gate_mask):
                xlim = self.profile_ax.get_xlim()
                ylim = self.profile_ax.get_ylim()
                self.profile_ax.scatter(
                    profile.distance_um[gate_mask],
                    profile.values[gate_mask],
                    s=12,
                    color="#16a34a",
                    alpha=0.7,
                    linewidths=0,
                    zorder=5,
                )
                self.profile_ax.set_xlim(xlim)
                self.profile_ax.set_ylim(ylim)
        self.profile_fig.subplots_adjust(left=0.14, right=0.68, bottom=0.20, top=0.82)
        self.profile_canvas.draw_idle()
        self.update_scatter_preview()
        if redraw_overlay:
            self.redraw_image()
        self.status_var.set(
            f"Profile sampled over {profile.width_px} px "
            f"({profile.width_px * self.pixel_size_um:.1f} um) width."
        )

    def default_save_stem(self) -> str:
        entry = self.current_change
        if entry is None:
            return "density_change_profile"
        return (
            f"{self.plate_var.get()}_{self.well_var.get()}_"
            f"{entry.first_tp}_to_{entry.next_tp}_{entry.kind}_profile"
        )

    def all_timepoints_save_stem(self) -> str:
        kind = self._selected_kind() or "density_change"
        return f"{self.plate_var.get()}_{self.well_var.get()}_{kind}_all_timepoints_profile"

    def current_line_points(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if len(self.click_points) != 2:
            return None
        return self.click_points[0], self.click_points[1]

    def set_profile_width(self, width: int) -> None:
        width = max(1, int(round(width)))
        if width % 2 == 0:
            width += 1
        self.width_var.set(width)
        if hasattr(self, "width_slider"):
            self.width_slider.set(width)
        self.width_label.configure(text=f"{width} px ({width * self.pixel_size_um:.1f} um)")

    def entry_metadata(self, entry: ChangeEntry) -> dict[str, object]:
        return {
            "first_timepoint": entry.first_tp,
            "next_timepoint": entry.next_tp,
            "interval_label": entry.label,
            "change_type_kind": entry.kind,
            "change_type_label": CHANGE_TYPE_LABELS.get(entry.kind, entry.kind),
            "source_tif": os.fspath(entry.path),
        }

    def line_metadata(
        self,
        output_path: Path,
        export_type: str,
        *,
        entries: list[ChangeEntry] | None = None,
    ) -> dict[str, object] | None:
        line = self.current_line_points()
        if line is None:
            return None
        start_xy, end_xy = line
        dx = end_xy[0] - start_xy[0]
        dy = end_xy[1] - start_xy[1]
        length_px = math.hypot(dx, dy)
        entry = self.current_change or self.selected_entry()
        return {
            "format": "density_change_profile_line",
            "format_version": 1,
            "export_type": export_type,
            "output_path": os.fspath(output_path),
            "run_dir": os.fspath(self.run_dir),
            "config_path": os.fspath(self.config_path),
            "plate": self.plate_var.get(),
            "well": self.well_var.get(),
            "density_maps_dir_name": self.density_maps_name,
            "density_changes_dir_name": self.density_changes_name,
            "pixel_size_um": self.pixel_size_um,
            "coarsening_factor": self.coarsening_factor,
            "background": self.background_var.get(),
            "extra_map": self.reference_map_var.get(),
            "extra_map_time": self.reference_time_var.get(),
            "center_to_edge_mode": bool(self.center_to_edge_var.get()),
            "resistant_density_gate": {
                "enabled": bool(self.res_density_gate_active()),
                "lower_cells_per_um2": float(self.res_density_gate_limits()[0]),
                "upper_cells_per_um2": float(self.res_density_gate_limits()[1]),
                "slider_min_cells_per_um2": float(self.res_density_gate_slider_min),
                "slider_max_cells_per_um2": float(self.res_density_gate_slider_max),
                "step_cells_per_um2": float(self.res_density_gate_step),
            },
            "line": {
                "start_xy_px": [float(start_xy[0]), float(start_xy[1])],
                "end_xy_px": [float(end_xy[0]), float(end_xy[1])],
                "length_px": float(length_px),
                "length_um": float(length_px * self.pixel_size_um),
                "width_px": int(self.width_var.get()),
                "width_um": float(self.width_var.get() * self.pixel_size_um),
            },
            "selected_entry": self.entry_metadata(entry) if entry is not None else None,
            "entries": [self.entry_metadata(item) for item in entries] if entries is not None else None,
        }

    def line_metadata_path(self, output_path: Path) -> Path:
        return output_path.with_suffix(output_path.suffix + ".line.json")

    def save_line_metadata(
        self,
        output_path: str | Path,
        export_type: str,
        *,
        entries: list[ChangeEntry] | None = None,
    ) -> Path | None:
        output = Path(output_path)
        metadata = self.line_metadata(output, export_type, entries=entries)
        if metadata is None:
            return None
        sidecar_path = self.line_metadata_path(output)
        try:
            with open(sidecar_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
                handle.write("\n")
        except Exception as exc:
            messagebox.showwarning("Could not save line metadata", str(exc))
            return None
        return sidecar_path

    def load_line_metadata(self) -> None:
        path = filedialog.askopenfilename(
            title="Load saved profile line",
            filetypes=[("Line metadata", "*.line.json"), ("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception as exc:
            messagebox.showerror("Could not load line metadata", str(exc))
            return

        line = metadata.get("line") if isinstance(metadata, dict) else None
        if not isinstance(line, dict):
            messagebox.showerror("Invalid line metadata", "The selected file does not contain saved line coordinates.")
            return
        try:
            start_xy = tuple(float(value) for value in line["start_xy_px"])
            end_xy = tuple(float(value) for value in line["end_xy_px"])
        except Exception:
            messagebox.showerror("Invalid line metadata", "The saved line coordinates are incomplete or invalid.")
            return
        if len(start_xy) != 2 or len(end_xy) != 2:
            messagebox.showerror("Invalid line metadata", "The saved line coordinates must contain x and y values.")
            return

        plate = str(metadata.get("plate", ""))
        well = str(metadata.get("well", ""))
        selected_entry = metadata.get("selected_entry") or {}
        if plate and plate in list(self.plate_box["values"]):
            self.plate_var.set(plate)
            self._on_plate_changed()
        if well:
            if well in list(self.well_box["values"]):
                self.well_var.set(well)
                self._on_well_changed()
            elif not self.plate_var.get():
                self.well_var.set(well)
                self._on_well_changed()

        kind = selected_entry.get("change_type_kind") if isinstance(selected_entry, dict) else None
        if kind:
            label = CHANGE_TYPE_LABELS.get(str(kind), str(kind))
            if label in list(self.change_type_box["values"]):
                self.change_type_var.set(label)
                self._refresh_intervals()

        interval_label = selected_entry.get("interval_label") if isinstance(selected_entry, dict) else None
        if interval_label and interval_label in getattr(self, "_entry_by_label", {}):
            labels = list(getattr(self, "_entry_by_label", {}).keys())
            self.interval_var.set(str(interval_label))
            self._set_interval_index(labels.index(str(interval_label)), labels=labels, move_slider=True, load=True)

        width_px = line.get("width_px")
        if width_px is not None:
            self.set_profile_width(int(round(float(width_px))))

        self.click_points = [(start_xy[0], start_xy[1]), (end_xy[0], end_xy[1])]
        if self.image is None:
            self.load_image()
        self.redraw_image()
        self.update_profile(redraw_overlay=False)
        self.status_var.set(f"Loaded line metadata: {path}")

    def save_csv(self) -> None:
        if self.current_profile is None:
            messagebox.showinfo("No profile", "Draw a line first.")
            return
        if self.current_change is None:
            messagebox.showinfo("No change map", "Select a change map first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save profile CSV",
            defaultextension=".csv",
            initialfile=f"{self.default_save_stem()}.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        profile = self.current_profile
        total_profile = self.total_density_profile_for_entry(
            self.current_change,
            profile.start_xy,
            profile.end_xy,
        )
        total_values = (
            total_profile.values
            if total_profile is not None and total_profile.values.shape == profile.values.shape
            else np.full(profile.values.shape, np.nan, dtype=float)
        )
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "distance_px",
                    "distance_um",
                    "value",
                    "total_density_cells_per_um2",
                    "start_x",
                    "start_y",
                    "end_x",
                    "end_y",
                    "width_px",
                    "width_um",
                    "value_label",
                ]
            )
            for distance_px, distance_um, value, total_density in zip(
                profile.distance_px,
                profile.distance_um,
                profile.values,
                total_values,
            ):
                writer.writerow(
                    [
                        distance_px,
                        distance_um,
                        value,
                        total_density,
                        profile.start_xy[0],
                        profile.start_xy[1],
                        profile.end_xy[0],
                        profile.end_xy[1],
                        profile.width_px,
                        profile.width_px * self.pixel_size_um,
                        self.y_axis_label(self.current_change),
                    ]
                )
        metadata_path = self.save_line_metadata(path, "profile_csv")
        suffix = f" and line metadata: {metadata_path}" if metadata_path is not None else ""
        self.status_var.set(f"Saved CSV: {path}{suffix}")

    def save_all_timepoints_csv(self) -> None:
        line = self.current_line_points()
        if line is None:
            messagebox.showinfo("No profile line", "Draw or define a line first.")
            return
        kind = self._selected_kind()
        if kind is None:
            messagebox.showinfo("No change type", "Select a change type first.")
            return

        entries = [entry for entry in self._available_changes() if entry.kind == kind]
        if not entries:
            messagebox.showinfo("No timepoints", "No change maps are available for this selection.")
            return

        path = filedialog.asksaveasfilename(
            title="Save all-timepoints profile CSV",
            defaultextension=".csv",
            initialfile=f"{self.all_timepoints_save_stem()}.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        start_xy, end_xy = line
        saved_rows = 0
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "plate",
                    "well",
                    "first_timepoint",
                    "next_timepoint",
                    "change_type",
                    "distance_px",
                    "distance_um",
                    "value",
                    "total_density_cells_per_um2",
                    "start_x",
                    "start_y",
                    "end_x",
                    "end_y",
                    "width_px",
                    "width_um",
                    "value_label",
                    "source_tif",
                ]
            )
            for entry in entries:
                try:
                    image = self.read_image(entry.path, use_cache=False)
                    profile = sample_line_profile(
                        image,
                        start_xy,
                        end_xy,
                        self.width_var.get(),
                        self.pixel_size_um,
                    )
                    total_profile = self.total_density_profile_for_entry(
                        entry,
                        start_xy,
                        end_xy,
                        use_cache=False,
                    )
                except Exception as exc:
                    print(f"[WARN] Skipping {entry.path}: {exc}")
                    continue
                total_values = (
                    total_profile.values
                    if total_profile is not None and total_profile.values.shape == profile.values.shape
                    else np.full(profile.values.shape, np.nan, dtype=float)
                )
                for distance_px, distance_um, value, total_density in zip(
                    profile.distance_px,
                    profile.distance_um,
                    profile.values,
                    total_values,
                ):
                    writer.writerow(
                        [
                            self.plate_var.get(),
                            self.well_var.get(),
                            entry.first_tp,
                            entry.next_tp,
                            entry.kind,
                            distance_px,
                            distance_um,
                            value,
                            total_density,
                            profile.start_xy[0],
                            profile.start_xy[1],
                            profile.end_xy[0],
                            profile.end_xy[1],
                            profile.width_px,
                            profile.width_px * self.pixel_size_um,
                            self.y_axis_label(entry),
                            entry.path,
                        ]
                    )
                    saved_rows += 1

        metadata_path = self.save_line_metadata(path, "all_timepoints_profile_csv", entries=entries)
        suffix = f" and line metadata: {metadata_path}" if metadata_path is not None else ""
        self.status_var.set(
            f"Saved {saved_rows:,} profile rows for {len(entries)} intervals: {path}{suffix}"
        )

    def save_all_timepoints_plots(self) -> None:
        line = self.current_line_points()
        if line is None:
            messagebox.showinfo("No profile line", "Draw or define a line first.")
            return
        kind = self._selected_kind()
        if kind is None:
            messagebox.showinfo("No change type", "Select a change type first.")
            return

        entries = [entry for entry in self._available_changes() if entry.kind == kind]
        if not entries:
            messagebox.showinfo("No timepoints", "No change maps are available for this selection.")
            return

        path = filedialog.asksaveasfilename(
            title="Save all-timepoints profile plots",
            defaultextension=".pdf",
            initialfile=f"{self.all_timepoints_save_stem()}_plots.pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        start_xy, end_xy = line
        plot_items: list[
            tuple[
                ChangeEntry,
                ProfileResult,
                ProfileResult | None,
                np.ndarray | None,
                ProfileResult | None,
            ]
        ] = []
        for entry in entries:
            try:
                image = self.read_image(entry.path, use_cache=False)
                profile = sample_line_profile(
                    image,
                    start_xy,
                    end_xy,
                    self.width_var.get(),
                    self.pixel_size_um,
                )
                total_profile = self.total_density_profile_for_entry(
                    entry,
                    start_xy,
                    end_xy,
                    use_cache=False,
                )
                resistant_fraction = self.resistant_fraction_profile_for_entry(
                    entry,
                    start_xy,
                    end_xy,
                    use_cache=False,
                )
                res_profile = self.density_profile_for_entry(
                    entry,
                    "res",
                    entry.first_tp,
                    start_xy,
                    end_xy,
                    use_cache=False,
                )
            except Exception as exc:
                print(f"[WARN] Skipping {entry.path}: {exc}")
                continue
            plot_items.append((entry, profile, total_profile, resistant_fraction, res_profile))

        if not plot_items:
            messagebox.showinfo("No plots", "No valid profiles could be sampled for export.")
            return

        finite_value_arrays = [
            profile.values[np.isfinite(profile.values)]
            for _, profile, _, _, _ in plot_items
            if np.isfinite(profile.values).any()
        ]
        if finite_value_arrays:
            all_values = np.concatenate(finite_value_arrays)
            global_ymin = float(np.nanmin(all_values))
            global_ymax = float(np.nanmax(all_values))
            if global_ymin == global_ymax:
                global_ymax = global_ymin + 1.0
            ypad = 0.05 * (global_ymax - global_ymin)
            shared_ylim = (global_ymin - ypad, global_ymax + ypad)
        else:
            shared_ylim = None

        shared_total_ylim = self.limits_from_arrays(
            [
                total_profile.values
                for _, _, total_profile, _, _ in plot_items
                if total_profile is not None
            ]
        )

        first_profile = plot_items[0][1]
        shared_xlim = (
            float(first_profile.distance_um[0]),
            float(first_profile.distance_um[-1]),
        )

        saved_pages = 0
        with PdfPages(path) as pdf:
            for entry, profile, total_profile, resistant_fraction, res_profile in plot_items:
                fig = Figure(figsize=(7.0, 3.5), dpi=150)
                ax = fig.add_subplot(111)
                ax.plot(
                    profile.distance_um,
                    profile.values,
                    color="#111111",
                    linewidth=1.25,
                )
                if (
                    total_profile is not None
                    and total_profile.values.shape == profile.values.shape
                    and resistant_fraction is not None
                    and resistant_fraction.shape == total_profile.values.shape
                ):
                    total_ax = ax.twinx()
                    line = colored_profile_line(
                        total_ax,
                        profile.distance_um,
                        total_profile.values,
                        resistant_fraction,
                        cmap=COMPOSITION_CMAP,
                        linewidth=1.8,
                        color_limits=(0.0, 1.0),
                    )
                    cax = fig.add_axes([0.86, 0.18, 0.025, 0.64])
                    colorbar = fig.colorbar(line, cax=cax)
                    colorbar.set_label("Resistant fraction of density (sensitive = 1 - fraction)")
                    total_ax.set_ylabel("Total density [cells/ÂµmÂ²]")
                    total_ax.set_xlim(*shared_xlim)
                    if shared_total_ylim is not None:
                        total_ax.set_ylim(*shared_total_ylim)
                elif total_profile is not None and total_profile.values.shape == profile.values.shape:
                    total_ax = ax.twinx()
                    total_ax.plot(
                        profile.distance_um,
                        total_profile.values,
                        color="#4f6d7a",
                        linewidth=1.5,
                        alpha=0.9,
                    )
                    total_ax.set_ylabel("Total density [cells/ÂµmÂ²]")
                    total_ax.set_xlim(*shared_xlim)
                    if shared_total_ylim is not None:
                        total_ax.set_ylim(*shared_total_ylim)
                total_profile = None
                if total_profile is not None and total_profile.values.shape == profile.values.shape:
                    line = colored_profile_line(
                        ax,
                        profile.distance_um,
                        profile.values,
                        total_profile.values,
                        color_limits=shared_color_limits,
                    )
                    colorbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
                    colorbar.set_label("Total density [cells/µm²]")
                else:
                    pass
                ax.set_xlabel("Distance along selected line [µm]")
                ax.set_ylabel(self.y_axis_label(entry))
                ax.set_xlim(*shared_xlim)
                if shared_ylim is not None:
                    ax.set_ylim(*shared_ylim)
                if (
                    self.res_density_gate_active()
                    and res_profile is not None
                    and res_profile.values.shape == profile.values.shape
                ):
                    gate_mask = self.res_density_gate_mask(res_profile.values)
                    gate_mask &= np.isfinite(profile.values)
                    if np.any(gate_mask):
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        ax.scatter(
                            profile.distance_um[gate_mask],
                            profile.values[gate_mask],
                            s=12,
                            color="#16a34a",
                            alpha=0.7,
                            linewidths=0,
                            zorder=5,
                        )
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)
                line_length_um = float(profile.distance_um[-1]) if profile.distance_um.size else 0.0
                ax.set_title(
                    f"{self.plate_var.get()} / {self.well_var.get()} / "
                    f"{entry.first_tp}->{entry.next_tp}\n"
                    f"Line length: {line_length_um:.1f} µm; "
                    f"width: {profile.width_px * self.pixel_size_um:.1f} µm"
                )
                ax.grid(True, color="0.9", linewidth=0.6)
                fig.subplots_adjust(left=0.11, right=0.70, bottom=0.16, top=0.82)
                pdf.savefig(fig)
                saved_pages += 1

        metadata_path = self.save_line_metadata(path, "all_timepoints_profile_plots", entries=entries)
        suffix = f" and line metadata: {metadata_path}" if metadata_path is not None else ""
        self.status_var.set(f"Saved {saved_pages} profile plot pages: {path}{suffix}")

    def save_line_scatterplot(self) -> None:
        line = self.current_line_points()
        if line is None:
            messagebox.showinfo("No profile line", "Draw or define a line first.")
            return
        scatter_entry = self.selected_entry()
        if scatter_entry is None:
            messagebox.showinfo("No change map", "Select a change map first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save line scatterplot",
            defaultextension=".png",
            initialfile=(
                f"{self.plate_var.get()}_{self.well_var.get()}_"
                f"{scatter_entry.first_tp}_to_{scatter_entry.next_tp}_"
                f"line_{scatter_entry.kind}_vs_total_density.png"
            ),
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            scatter_data = self.line_scatter_data(scatter_entry, line[0], line[1])
        except Exception as exc:
            messagebox.showerror("Could not build scatterplot", str(exc))
            return
        if scatter_data is None:
            messagebox.showerror(
                "Missing density maps",
                "The scatterplot requires total density, initial resistant density, "
                "and next resistant density maps.",
            )
            return

        fig = Figure(figsize=(6.4, 5.0), dpi=200)
        n_points = self.draw_line_scatterplot(fig, scatter_entry, scatter_data)
        if n_points == 0:
            messagebox.showinfo(
                "No gated points",
                "No sampled line-band pixels pass the gate for the selected change map.",
            )
            return
        fig.savefig(path, dpi=300, bbox_inches="tight")
        metadata_path = self.save_line_metadata(path, "line_scatterplot", entries=[scatter_entry])
        suffix = f" and line metadata: {metadata_path}" if metadata_path is not None else ""
        self.status_var.set(f"Saved line scatterplot: {path}{suffix}")
        return

        zero_threshold = self.zero_initial_threshold()
        xscale = self.config_axis_scale("relative_density_change_scatter_xscale")
        yscale = self.config_axis_scale("relative_density_change_scatter_yscale")

        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & np.isfinite(marker_color)
            & np.isfinite(next_res)
            & (marker_color > zero_threshold)
            & (next_res > zero_threshold)
        )
        if xscale == "log":
            valid &= x > 0
        if yscale == "log":
            valid &= y > 0

        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            messagebox.showinfo(
                "No gated points",
                "No sampled line points pass the same relative-growth gate "
                "used by the pipeline scatterplot.",
            )
            return
        order = valid_idx[np.argsort(marker_color[valid_idx])]

        color_max = self.scatter_colorbar_vmax(marker_color)
        color_norm = Normalize(vmin=0.0, vmax=color_max, clip=True)
        color_map = LinearSegmentedColormap.from_list(
            "black_to_goldenrod",
            ["#000000", "#DAA520"],
        )

        fig = Figure(figsize=(6.4, 5.0), dpi=200)
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            x[order],
            y[order],
            c=marker_color[order],
            cmap=color_map,
            norm=color_norm,
            s=14,
            alpha=1.0,
            linewidths=0,
        )
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        xlim = self.config_axis_limits("relative_density_change_scatter_xlim")
        ylim = self.scatter_y_axis_limits(rel_entry)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(f"Total density at day {rel_entry.first_tp} [cells/µm²]")
        ax.set_ylabel(self.current_profile_y_axis_label(rel_entry))
        ax.set_title(
            f"{self.plate_var.get()} {self.well_var.get()}: "
            f"selected-line scatter {rel_entry.first_tp}->{rel_entry.next_tp}\n"
            f"n={valid_idx.size:,} gated points; width={self.width_var.get() * self.pixel_size_um:.1f} µm"
        )
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.text(
            0.02,
            0.98,
            f"plotted where initial and next res > {zero_threshold:g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.35",
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
        colorbar.set_label(
            f"Resistant density at day {rel_entry.first_tp} [cells/µm²]; "
            f"vmax={color_max:.3g}"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        self.status_var.set(f"Saved line scatterplot: {path}")

    def save_all_line_scatterplots(self) -> None:
        line = self.current_line_points()
        if line is None:
            messagebox.showinfo("No profile line", "Draw or define a line first.")
            return

        selected = self.selected_entry()
        selected_kind = selected.kind if selected is not None else self._selected_kind()
        entries = [
            entry
            for entry in self._available_changes()
            if entry.kind == selected_kind
        ]
        if not entries:
            messagebox.showinfo("No change maps", "No maps are available for the selected change type.")
            return

        path = filedialog.asksaveasfilename(
            title="Save all line scatterplots",
            defaultextension=".pdf",
            initialfile=(
                f"{self.plate_var.get()}_{self.well_var.get()}_"
                f"all_timepoints_line_{selected_kind}_vs_total_density.pdf"
            ),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        start_xy, end_xy = line
        scatter_items: list[
            tuple[
                ChangeEntry,
                dict[str, np.ndarray] | None,
                dict[str, np.ndarray] | None,
            ]
        ] = []
        for entry in entries:
            try:
                line_data = self.line_scatter_data(
                    entry,
                    start_xy,
                    end_xy,
                    use_cache=False,
                )
                full_data = self.full_well_scatter_data(entry)
            except Exception as exc:
                print(f"[WARN] Skipping {entry.path}: {exc}")
                continue
            if line_data is not None or full_data is not None:
                scatter_items.append((entry, line_data, full_data))

        if not scatter_items:
            messagebox.showinfo("No scatterplots", "No valid line scatterplots could be sampled.")
            return

        saved_pages = 0
        skipped_empty = 0
        with PdfPages(path) as pdf:
            for entry, line_data, full_data in scatter_items:
                fig = Figure(figsize=(12.0, 5.0), dpi=200)
                axes = fig.subplots(1, 2)
                interval_label = f"{entry.first_tp}->{entry.next_tp}"
                n_full = 0
                n_line = 0
                if full_data is not None:
                    n_full = self.draw_scatter_on_axis(
                        fig,
                        axes[0],
                        entry,
                        full_data,
                        f"Full well {interval_label}",
                        point_size=2,
                    )
                else:
                    axes[0].text(0.5, 0.5, "No full-well scatter data", ha="center", va="center")
                    axes[0].set_axis_off()

                if line_data is not None:
                    n_line = self.draw_scatter_on_axis(
                        fig,
                        axes[1],
                        entry,
                        line_data,
                        self.selected_line_scatter_title(interval_label),
                        point_size=4,
                    )
                else:
                    axes[1].text(0.5, 0.5, "No selected-line scatter data", ha="center", va="center")
                    axes[1].set_axis_off()

                if n_full == 0 and n_line == 0:
                    skipped_empty += 1
                    continue
                fig.tight_layout()
                pdf.savefig(fig, dpi=300, bbox_inches="tight")
                saved_pages += 1

        metadata_path = self.save_line_metadata(path, "all_timepoints_line_scatterplots", entries=entries)
        suffix = f" and line metadata: {metadata_path}" if metadata_path is not None else ""
        self.status_var.set(
            f"Saved {saved_pages} line scatterplot pages"
            f"{f' ({skipped_empty} empty intervals skipped)' if skipped_empty else ''}: {path}{suffix}"
        )

    def save_plot(self) -> None:
        if self.current_profile is None:
            messagebox.showinfo("No profile", "Draw a line first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save profile plot",
            defaultextension=".png",
            initialfile=f"{self.default_save_stem()}.png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        self.profile_fig.savefig(path, dpi=300, bbox_inches="tight")
        metadata_path = self.save_line_metadata(path, "profile_plot")
        suffix = f" and line metadata: {metadata_path}" if metadata_path is not None else ""
        self.status_var.set(f"Saved plot: {path}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the density-change line-profile GUI."
    )
    parser.add_argument(
        "--config",
        default=os.fspath(DEFAULT_CONFIG_PATH),
        help=f"YAML config path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--run-dir",
        default=os.fspath(DEFAULT_RUN_DIR),
        help=f"Pipeline run directory. Default: {DEFAULT_RUN_DIR}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    run_dir = Path(args.run_dir).expanduser()
    root = tk.Tk()
    root.geometry("1280x860")
    DensityChangeProfileGUI(root, run_dir=run_dir, config_path=config_path)
    root.mainloop()


if __name__ == "__main__":
    main()
