"""Export selected microscopy movie frames with the Figure 3 channel display."""

from pathlib import Path
import os
import sys

import numpy as np
from PIL import Image
import tifffile

ROOT = Path(__file__).resolve().parents[2]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from shared.data_paths import figure_plot_dir

MOVIE_DIRECTORY = Path(
    os.environ.get(
        "EICHE_FISCHER_MOVIE_DIR",
        figure_plot_dir("Figure_1", "Movies", create=False),
    )
)
OUTPUT_DIRECTORY = MOVIE_DIRECTORY / "fluorescence_composites_days_01_10_19_33"
MOVIES = (
    "D3_AT50-D3_zoom.tif",
    "A2_AT100-A2.tif",
    "A1_CT-A1.tif",
)
# TIFF timesteps are one-based in the requested export; NumPy page positions are zero-based.
TIMESTEP_TO_DAY = {1: 1, 5: 10, 9: 19, 15: 33}

# Display settings, ordered identically for sensitive/blue and resistant/yellow.
SEN_BACKGROUND_PERCENTILE = 68.0
SEN_SIGNAL_PERCENTILE = 98.5
SEN_SIGNAL_GAMMA = 0.10
SEN_DISPLAY_MIN_BY_FRAME = {
    ("D3_AT50-D3_zoom", 1): 0.0,
    ("D3_AT50-D3_zoom", 10): 150.0,
    ("D3_AT50-D3_zoom", 19): 97.0,
    ("D3_AT50-D3_zoom", 33): 182.0,
    ("A2_AT100-A2", 1): 0.0,
    ("A2_AT100-A2", 10): 164.0,
    ("A2_AT100-A2", 19): 164.0,
    ("A2_AT100-A2", 33): 164.0,
    ("A1_CT-A1", 1): 0.0,
    ("A1_CT-A1", 10): 117.0,
    ("A1_CT-A1", 19): 195.0,
    ("A1_CT-A1", 33): 180.0,
}
SEN_DISPLAY_MAX = 235
SEN_RGB = np.asarray((0x41, 0x69, 0xE1), dtype=float) / 255.0

RES_BACKGROUND_PERCENTILE = 65.0
RES_SIGNAL_PERCENTILE = 100.0
RES_SIGNAL_GAMMA = 0.20
RES_DISPLAY_MIN = 113.0
RES_DISPLAY_MAX = 180
RES_RGB = np.asarray((0xDA, 0xA5, 0x20), dtype=float) / 255.0


def scale(channel: np.ndarray, *, low: float, high: float) -> np.ndarray:
    """Scale a channel with shared limits, keeping all frames comparable."""
    if high <= low:
        return np.zeros_like(channel, dtype=float)
    return np.clip((channel - low) / (high - low), 0, 1)


def apply_gamma(channel: np.ndarray, *, gamma: float) -> np.ndarray:
    """Apply gamma correction to a normalized channel."""
    return channel ** gamma


def apply_display_min(channel: np.ndarray, *, display_min: float) -> np.ndarray:
    """Apply an 8-bit display minimum after gamma correction."""
    return np.clip(
        (channel * 255.0 - display_min) / (255.0 - display_min),
        0,
        1,
    )


def apply_display_max(channel: np.ndarray, *, display_max: float) -> np.ndarray:
    """Apply the final ImageJ-like display maximum after gamma correction."""
    if not 0 < display_max <= 255:
        raise ValueError(f"Display maximum must be in (0, 255], got {display_max}")
    return np.clip(channel * 255.0 / display_max, 0, 1)


def shared_channel_limits(movie_paths: tuple[Path, ...]) -> dict[str, tuple[float, float]]:
    """Calculate one fluorescence range for every requested output frame."""
    samples = {"sen": [], "res": []}
    maxima = {"sen": -np.inf, "res": -np.inf}
    sample_size = 250_000
    for movie_path in movie_paths:
        with tifffile.TiffFile(movie_path) as tif:
            series = tif.series[0]
            channel_count = series.shape[1]
            for timestep in TIMESTEP_TO_DAY:
                time_index = timestep - 1
                for name, channel_index in (("res", 0), ("sen", 1)):
                    channel = series.pages[
                        time_index * channel_count + channel_index
                    ].asarray()
                    flattened = channel.ravel()
                    indices = np.linspace(
                        0, flattened.size - 1,
                        num=min(sample_size, flattened.size), dtype=int,
                    )
                    samples[name].append(flattened[indices])
                    maxima[name] = max(maxima[name], float(np.nanmax(channel)))
    return {
        "sen": (
            float(np.nanpercentile(np.concatenate(samples["sen"]), SEN_BACKGROUND_PERCENTILE)),
            float(np.nanpercentile(np.concatenate(samples["sen"]), SEN_SIGNAL_PERCENTILE)),
        ),
        "res": (
            float(np.nanpercentile(np.concatenate(samples["res"]), RES_BACKGROUND_PERCENTILE)),
            maxima["res"],
        ),
    }


def export_movie(movie_path: Path, limits: dict[str, tuple[float, float]]) -> list[Path]:
    outputs = []
    two_channel_frames = []
    with tifffile.TiffFile(movie_path) as tif:
        series = tif.series[0]
        if series.axes != "TCYX" or series.shape[1] < 2:
            raise ValueError(f"Expected a TCYX TIFF with at least two channels: {movie_path}")
        channel_count = series.shape[1]
        for timestep, day in TIMESTEP_TO_DAY.items():
            time_index = timestep - 1
            if time_index >= series.shape[0]:
                raise ValueError(f"Timestep {timestep} is unavailable in {movie_path.name}")
            # These movie TIFFs store resistant/yellow in channel 1 and
            # sensitive/blue in channel 2 (the page indices are zero-based).
            res = series.pages[time_index * channel_count].asarray().astype(float)
            sen = series.pages[time_index * channel_count + 1].asarray().astype(float)
            sen = scale(
                sen,
                low=limits["sen"][0], high=limits["sen"][1],
            )
            res = scale(
                res,
                low=limits["res"][0], high=limits["res"][1],
            )
            sen = apply_gamma(sen, gamma=SEN_SIGNAL_GAMMA)
            res = apply_gamma(res, gamma=RES_SIGNAL_GAMMA)
            sen_display_min = SEN_DISPLAY_MIN_BY_FRAME[(movie_path.stem, day)]
            sen = apply_display_min(sen, display_min=sen_display_min)
            res = apply_display_min(res, display_min=RES_DISPLAY_MIN)
            sen = apply_display_max(sen, display_max=SEN_DISPLAY_MAX)
            res = apply_display_max(res, display_max=RES_DISPLAY_MAX)
            composite = np.clip(
                sen[..., np.newaxis] * SEN_RGB
                + res[..., np.newaxis] * RES_RGB,
                0,
                1,
            )
            output_path = OUTPUT_DIRECTORY / f"{movie_path.stem}_day_{day:02d}.png"
            Image.fromarray(np.round(composite * 255).astype(np.uint8), mode="RGB").save(
                output_path
            )
            outputs.append(output_path)
            two_channel_frames.append(np.stack((
                np.round(res * 255).astype(np.uint8),
                np.round(sen * 255).astype(np.uint8),
            )))
    two_channel_output = OUTPUT_DIRECTORY / (
        f"{movie_path.stem}_days_01_10_19_33_two_channel.tif"
    )
    res_lut = np.round(
        np.asarray(RES_RGB)[:, np.newaxis] * np.arange(256)
    ).astype(np.uint8)
    sen_lut = np.round(
        np.asarray(SEN_RGB)[:, np.newaxis] * np.arange(256)
    ).astype(np.uint8)
    tifffile.imwrite(
        two_channel_output,
        np.stack(two_channel_frames),
        imagej=True,
        metadata={"axes": "TCYX", "LUTs": [res_lut, sen_lut]},
    )
    outputs.append(two_channel_output)
    return outputs


def export_gamma_intermediate_stack(
    movie_paths: tuple[Path, ...], limits: dict[str, tuple[float, float]]
) -> Path:
    """Save all requested frames after gamma but before the display limits."""
    frames = []
    labels = []
    for movie_path in movie_paths:
        with tifffile.TiffFile(movie_path) as tif:
            series = tif.series[0]
            channel_count = series.shape[1]
            for timestep, day in TIMESTEP_TO_DAY.items():
                time_index = timestep - 1
                res = series.pages[time_index * channel_count].asarray().astype(float)
                sen = series.pages[time_index * channel_count + 1].asarray().astype(float)
                res = apply_gamma(
                    scale(res, low=limits["res"][0], high=limits["res"][1]),
                    gamma=RES_SIGNAL_GAMMA,
                )
                sen = apply_gamma(
                    scale(sen, low=limits["sen"][0], high=limits["sen"][1]),
                    gamma=SEN_SIGNAL_GAMMA,
                )
                frames.append(np.stack((
                    np.round(res * 255).astype(np.uint8),
                    np.round(sen * 255).astype(np.uint8),
                )))
                label_prefix = f"{movie_path.stem}, day {day:02d}"
                labels.extend((f"{label_prefix}, res", f"{label_prefix}, sen"))
    output_path = OUTPUT_DIRECTORY / (
        "all_wells_days_01_10_19_33_shared_scaled_after_gamma_before_min.tif"
    )
    res_lut = np.round(
        np.asarray(RES_RGB)[:, np.newaxis] * np.arange(256)
    ).astype(np.uint8)
    sen_lut = np.round(
        np.asarray(SEN_RGB)[:, np.newaxis] * np.arange(256)
    ).astype(np.uint8)
    tifffile.imwrite(
        output_path,
        np.stack(frames),
        imagej=True,
        metadata={"axes": "TCYX", "LUTs": [res_lut, sen_lut], "Labels": labels},
    )
    return output_path


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    movie_paths = tuple(MOVIE_DIRECTORY / movie for movie in MOVIES)
    limits = shared_channel_limits(movie_paths)
    outputs = [output for movie in movie_paths for output in export_movie(movie, limits)]
    intermediate_stack = export_gamma_intermediate_stack(movie_paths, limits)
    print(
        "Shared normalized limits: "
        f"sen={limits['sen']}, res={limits['res']}"
    )
    print(f"Exported {len(outputs)} files to {OUTPUT_DIRECTORY}")
    for output in outputs:
        print(output.name)
    print(intermediate_stack.name)


if __name__ == "__main__":
    main()
