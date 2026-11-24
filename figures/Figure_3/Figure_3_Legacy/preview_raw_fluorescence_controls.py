#!/usr/bin/env python3
"""Interactive, low-resolution preview for Figure 3 raw-fluorescence settings.

Run this file directly.  Select d1/d12/d26 at left and move the sliders; the
preview updates without rebuilding any density maps or figure caches.  Click
"Print settings" to copy the displayed values into plot_at50_density_maps.py.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import matplotlib
# PyCharm's inline/SciView backend does not provide usable widget sliders.
# Tk ships with the local Windows Python installation, unlike Qt bindings.
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.widgets import Button, RadioButtons, Slider
import tifffile as tiff

import plot_at50_density_maps as maps


# Previewing every fourth pixel keeps slider updates responsive while retaining
# the tile/background structure needed for display tuning.
PREVIEW_DOWNSAMPLE = 4
DAYS = ("01", "12", "26")


def normalise(channel, low_percentile, high_percentile, gamma):
    low = np.nanpercentile(channel, low_percentile)
    high = np.nanpercentile(channel, high_percentile)
    if high <= low:
        return np.zeros_like(channel, dtype=float)
    return np.clip((channel - low) / (high - low), 0, 1) ** gamma


class RawPreview:
    def __init__(self):
        self.images: dict[str, np.ndarray] = {}
        self.day = DAYS[0]
        self.sliders = {}
        self.figure, self.axis = plt.subplots(figsize=(8.3, 7.0))
        self.figure.subplots_adjust(left=0.34, bottom=0.42, right=0.80, top=0.94)
        self.axis.set_axis_off()
        self.artist = self.axis.imshow(self.composite(), origin="lower")
        self.axis.set_title("Figure 3 raw fluorescence preview — d1")

        slider_specs = (
            ("blue background %", 50, 98, maps.RAW_BACKGROUND_PERCENTILE, 0.5),
            ("blue upper %", 90, 100, maps.BLUE_SIGNAL_PERCENTILE, 0.1),
            ("blue gamma", 0.10, 2.00, maps.BLUE_SIGNAL_GAMMA, 0.05),
            ("blue floor", 0.00, 0.60, maps.BLUE_DISPLAY_FLOOR, 0.01),
            ("yellow background %", 50, 98, maps.YELLOW_BACKGROUND_PERCENTILE, 0.5),
            ("yellow upper %", 90, 100, maps.RAW_SIGNAL_PERCENTILE, 0.1),
            ("yellow gamma", 0.10, 2.00, maps.YELLOW_SIGNAL_GAMMA, 0.05),
            ("yellow floor", 0.00, 0.60, maps.YELLOW_DISPLAY_FLOOR, 0.01),
        )
        for index, (label, minimum, maximum, initial, step) in enumerate(slider_specs):
            axis = self.figure.add_axes((0.08, 0.35 - index * 0.038, 0.21, 0.018))
            slider = Slider(axis, label, minimum, maximum, valinit=initial, valstep=step)
            slider.on_changed(self.update)
            self.sliders[label] = slider

        day_axis = self.figure.add_axes((0.84, 0.44, 0.12, 0.12))
        self.day_selector = RadioButtons(day_axis, ("d1", "d12", "d26"), active=0)
        self.day_selector.on_clicked(self.change_day)
        print_axis = self.figure.add_axes((0.08, 0.02, 0.20, 0.035))
        self.print_button = Button(print_axis, "Print settings")
        self.print_button.on_clicked(self.print_settings)
        save_axis = self.figure.add_axes((0.08, 0.07, 0.20, 0.035))
        self.save_button = Button(save_axis, "Save to plot script")
        self.save_button.on_clicked(self.save_settings)

    def raw_image(self):
        if self.day not in self.images:
            path = maps.raw_experiment_path(self.day)
            raw = tiff.imread(path).astype(np.float32)
            self.images[self.day] = raw[:, ::PREVIEW_DOWNSAMPLE, ::PREVIEW_DOWNSAMPLE]
        return self.images[self.day]

    def composite(self):
        raw = self.raw_image()
        value = {label: slider.val for label, slider in self.sliders.items()} if self.sliders else {
            "blue background %": maps.RAW_BACKGROUND_PERCENTILE,
            "blue upper %": maps.BLUE_SIGNAL_PERCENTILE,
            "blue gamma": maps.BLUE_SIGNAL_GAMMA,
            "blue floor": maps.BLUE_DISPLAY_FLOOR,
            "yellow background %": maps.YELLOW_BACKGROUND_PERCENTILE,
            "yellow upper %": maps.RAW_SIGNAL_PERCENTILE,
            "yellow gamma": maps.YELLOW_SIGNAL_GAMMA,
            "yellow floor": maps.YELLOW_DISPLAY_FLOOR,
        }
        blue = normalise(raw[0], value["blue background %"], value["blue upper %"], value["blue gamma"])
        yellow = normalise(raw[1], value["yellow background %"], value["yellow upper %"], value["yellow gamma"])
        blue[blue < value["blue floor"]] = 0
        yellow[yellow < value["yellow floor"]] = 0
        return np.clip(
            blue[..., None] * np.asarray(to_rgb(maps.FIGURE_COLOR_MAP["blue"]))
            + yellow[..., None] * np.asarray(to_rgb(maps.FIGURE_COLOR_MAP["yellow"])),
            0, 1,
        )

    def update(self, _=None):
        self.artist.set_data(self.composite())
        self.figure.canvas.draw_idle()

    def change_day(self, label):
        self.day = label.removeprefix("d").zfill(2)
        self.axis.set_title(f"Figure 3 raw fluorescence preview — {label}")
        self.update()

    def print_settings(self, _=None):
        value = {label: slider.val for label, slider in self.sliders.items()}
        print("\n# Copy into plot_at50_density_maps.py")
        print(f"RAW_BACKGROUND_PERCENTILE = {value['blue background %']:.1f}")
        print(f"BLUE_SIGNAL_PERCENTILE = {value['blue upper %']:.1f}")
        print(f"BLUE_SIGNAL_GAMMA = {value['blue gamma']:.2f}")
        print(f"BLUE_DISPLAY_FLOOR = {value['blue floor']:.2f}")
        print(f"YELLOW_BACKGROUND_PERCENTILE = {value['yellow background %']:.1f}")
        print(f"RAW_SIGNAL_PERCENTILE = {value['yellow upper %']:.1f}")
        print(f"YELLOW_SIGNAL_GAMMA = {value['yellow gamma']:.2f}")
        print(f"YELLOW_DISPLAY_FLOOR = {value['yellow floor']:.2f}")

    def save_settings(self, _=None):
        """Persist the active slider values in the figure plotting script."""
        value = {label: slider.val for label, slider in self.sliders.items()}
        updates = {
            "RAW_BACKGROUND_PERCENTILE": f"{value['blue background %']:.1f}",
            "BLUE_SIGNAL_PERCENTILE": f"{value['blue upper %']:.1f}",
            "BLUE_SIGNAL_GAMMA": f"{value['blue gamma']:.2f}",
            "BLUE_DISPLAY_FLOOR": f"{value['blue floor']:.2f}",
            "YELLOW_BACKGROUND_PERCENTILE": f"{value['yellow background %']:.1f}",
            "RAW_SIGNAL_PERCENTILE": f"{value['yellow upper %']:.1f}",
            "YELLOW_SIGNAL_GAMMA": f"{value['yellow gamma']:.2f}",
            "YELLOW_DISPLAY_FLOOR": f"{value['yellow floor']:.2f}",
        }
        path = Path(maps.__file__).resolve()
        text = path.read_text(encoding="utf-8")
        for name, formatted_value in updates.items():
            text, replacements = re.subn(
                rf"^{name}\s*=\s*.*$", f"{name} = {formatted_value}",
                text, flags=re.MULTILINE,
            )
            if replacements != 1:
                raise RuntimeError(f"Could not save {name} in {path}")
        path.write_text(text, encoding="utf-8")
        print(f"Saved raw-image display settings to {path}")


if __name__ == "__main__":
    preview = RawPreview()
    plt.show()
