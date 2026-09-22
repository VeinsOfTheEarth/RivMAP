"""Run RivMAPy's 1984-2015 temporal-change workflow on bundled masks.

From the repository root::

    python examples/multiyear_change.py

The script extracts every annual centerline, measures consecutive migration,
erosion, accretion, and neck cutoffs, then saves a compact review figure.  It
does not modify ``data/riv.mat``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / "examples" / "output" / ".matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np

from rivmapy.change import migration_centerlines, migration_mask
from rivmapy.io import RiverRecord, load_rivmap_mat
from rivmapy.mask import centerline_from_mask

@dataclass(frozen=True)
class TemporalSummary:
    years: np.ndarray
    centerlines: tuple[np.ndarray, ...]
    reach_width_pixels: np.ndarray
    interval_years: np.ndarray
    migration_pixels: np.ndarray
    erosion_pixels: np.ndarray
    accretion_pixels: np.ndarray
    centerline_cutoffs: np.ndarray
    mask_cutoff_pixels: np.ndarray
    migration_frequency: np.ndarray
    cutoff_frequency: np.ndarray


def _line_length(line: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def _post_1995_widening(
    years: np.ndarray, widths: np.ndarray
) -> tuple[float, int | None, int | None]:
    """Return low-to-later-peak widening, or NaN for an earlier subset."""

    eligible = np.flatnonzero(years >= 1995)
    if not len(eligible):
        return float("nan"), None, None
    low_index = int(eligible[np.argmin(widths[eligible])])
    peak_index = int(low_index + np.argmax(widths[low_index:]))
    return (
        float(widths[peak_index] - widths[low_index]),
        int(years[low_index]),
        int(years[peak_index]),
    )


def analyze_records(records: tuple[RiverRecord, ...]) -> TemporalSummary:
    """Compute all annual and consecutive-interval products for the example."""

    if len(records) < 2:
        raise ValueError("the temporal example requires at least two records")
    shape = records[0].single_thread.shape
    if any(record.single_thread.shape != shape for record in records):
        raise ValueError("all records must use a common raster grid")

    centerlines: list[np.ndarray] = []
    lengths: list[float] = []
    reach_widths: list[float] = []
    for record in records:
        result = centerline_from_mask(
            record.single_thread,
            record.exit_sides,
            record.nominal_width,
        )
        centerlines.append(result.coordinates)
        length = _line_length(result.coordinates)
        lengths.append(length)
        reach_widths.append(float(record.single_thread.sum() / length))

    interval_count = len(records) - 1
    migration_pixels = np.zeros(interval_count, dtype=float)
    erosion_pixels = np.zeros(interval_count, dtype=float)
    accretion_pixels = np.zeros(interval_count, dtype=float)
    centerline_cutoffs = np.zeros(interval_count, dtype=int)
    mask_cutoff_pixels = np.zeros(interval_count, dtype=float)
    migration_frequency = np.zeros(shape, dtype=np.uint16)
    cutoff_frequency = np.zeros(shape, dtype=np.uint16)

    for index in range(interval_count):
        first = records[index]
        second = records[index + 1]
        centerline_change = migration_centerlines(
            centerlines[index],
            centerlines[index + 1],
            shape,
            first.nominal_width,
            exit_sides_t1=first.exit_sides,
            exit_sides_t2=second.exit_sides,
        )
        mask_change = migration_mask(
            first.single_thread,
            second.single_thread,
            first.nominal_width,
        )
        migration_pixels[index] = centerline_change.migrated.sum()
        erosion_pixels[index] = mask_change.erosion.sum()
        accretion_pixels[index] = mask_change.accretion.sum()
        centerline_cutoffs[index] = len(centerline_change.cutoff_area)
        mask_cutoff_pixels[index] = mask_change.cutoffs.sum()
        migration_frequency += centerline_change.migrated.astype(np.uint16)
        cutoff_frequency += centerline_change.cutoffs.astype(np.uint16)

    return TemporalSummary(
        years=np.asarray([record.year for record in records], dtype=int),
        centerlines=tuple(centerlines),
        reach_width_pixels=np.asarray(reach_widths),
        interval_years=np.diff([record.year for record in records]).astype(float),
        migration_pixels=migration_pixels,
        erosion_pixels=erosion_pixels,
        accretion_pixels=accretion_pixels,
        centerline_cutoffs=centerline_cutoffs,
        mask_cutoff_pixels=mask_cutoff_pixels,
        migration_frequency=migration_frequency,
        cutoff_frequency=cutoff_frequency,
    )


def create_figure(
    records: tuple[RiverRecord, ...],
    summary: TemporalSummary,
    *,
    pixel_size_metres: float,
):
    """Create the temporal review figure and derived physical-unit series."""

    interval_years = summary.years[1:]
    old_length = np.asarray([_line_length(line) for line in summary.centerlines[:-1]])
    migration_metres_per_year = (
        summary.migration_pixels / old_length * pixel_size_metres / summary.interval_years
    )
    square_kilometres_per_pixel = pixel_size_metres**2 / 1_000_000
    erosion_km2 = summary.erosion_pixels * square_kilometres_per_pixel
    accretion_km2 = summary.accretion_pixels * square_kilometres_per_pixel
    width_metres = summary.reach_width_pixels * pixel_size_metres
    post_1995_widening, low_year, peak_year = _post_1995_widening(
        summary.years, width_metres
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    first = records[0].single_thread
    last = records[-1].single_thread
    planform_rgb = np.ones((*first.shape, 3), dtype=float)
    planform_rgb[first] = np.array([0.45, 0.72, 0.87])
    planform_rgb[last] = np.array([0.99, 0.60, 0.33])
    planform_rgb[first & last] = np.array([0.45, 0.45, 0.45])
    planform_rgb[summary.cutoff_frequency > 0] = np.array([0.75, 0.10, 0.15])
    axes[0, 0].imshow(planform_rgb, interpolation="nearest")
    axes[0, 0].set(
        title=f"Planforms: {records[0].year} (blue), {records[-1].year} (orange), cutoffs (red)",
        xlabel="x (pixel)",
        ylabel="y (pixel)",
    )

    axes[0, 1].plot(summary.years, width_metres, marker="o", ms=3, color="#238b45")
    axes[0, 1].set(
        title="Reach-average width",
        xlabel="year",
        ylabel="width (m)",
    )

    migration_axis = axes[1, 0]
    migration_axis.plot(
        interval_years,
        migration_metres_per_year,
        marker="o",
        ms=3,
        color="#2c7fb8",
        label="migration",
    )
    migration_axis.set(
        title="Annual centerline migration and detected cutoffs",
        xlabel="ending year",
        ylabel="area/length migration (m/yr)",
    )
    cutoff_axis = migration_axis.twinx()
    cutoff_axis.bar(
        interval_years,
        summary.centerline_cutoffs,
        width=0.65,
        color="#cb181d",
        alpha=0.35,
        label="cutoffs",
    )
    cutoff_axis.set_ylabel("cutoff count")
    lines = migration_axis.lines + cutoff_axis.containers
    migration_axis.legend(
        [migration_axis.lines[0], cutoff_axis.containers[0]],
        ["migration", "cutoffs"],
        loc="upper right",
    )

    balance_axis = axes[1, 1]
    balance_axis.plot(interval_years, erosion_km2, color="#d95f0e", label="erosion")
    balance_axis.plot(interval_years, accretion_km2, color="#756bb1", label="accretion")
    balance_axis.plot(
        interval_years,
        np.cumsum(erosion_km2 - accretion_km2),
        color="#252525",
        ls="--",
        label="cumulative erosion - accretion",
    )
    balance_axis.axhline(0, color="0.7", lw=0.8)
    balance_axis.set(
        title="Mask-derived erosion/accretion",
        xlabel="ending year",
        ylabel="area (km²)",
    )
    balance_axis.legend(fontsize=8)

    figure.suptitle("RivMAPy multiyear-change demonstration", fontsize=16)
    diagnostics = {
        "cutoff_count": int(summary.centerline_cutoffs.sum()),
        "width_change_metres": float(width_metres[-1] - width_metres[0]),
        "post_1995_widening_metres": post_1995_widening,
        "post_1995_low_year": low_year,
        "post_1995_peak_year": peak_year,
        "erosion_accretion_balance_km2": float(np.sum(erosion_km2 - accretion_km2)),
        "mean_migration_metres_per_year": float(np.mean(migration_metres_per_year)),
    }
    return figure, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY_ROOT / "data" / "riv.mat")
    parser.add_argument("--pixel-size", type=float, default=30.0, help="metres per pixel")
    parser.add_argument(
        "--maximum-records",
        type=int,
        default=None,
        help="optional quick-run limit; the demonstrated result uses all records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "examples" / "output" / "multiyear_change.png",
    )
    parser.add_argument("--show", action="store_true", help="open the figure after saving")
    arguments = parser.parse_args()
    if not np.isfinite(arguments.pixel_size) or arguments.pixel_size <= 0:
        parser.error("--pixel-size must be finite and positive")

    records = load_rivmap_mat(arguments.data)
    if arguments.maximum_records is not None:
        if arguments.maximum_records < 2:
            parser.error("--maximum-records must be at least 2")
        records = records[: arguments.maximum_records]
    summary = analyze_records(records)
    figure, diagnostics = create_figure(
        records,
        summary,
        pixel_size_metres=arguments.pixel_size,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"years: {summary.years[0]}-{summary.years[-1]}")
    print(f"centerline cutoffs: {diagnostics['cutoff_count']}")
    print(f"width change: {diagnostics['width_change_metres']:.1f} m")
    if np.isfinite(diagnostics["post_1995_widening_metres"]):
        print(
            "post-1995 low-to-peak widening: "
            f"{diagnostics['post_1995_widening_metres']:.1f} m "
            f"({diagnostics['post_1995_low_year']}-"
            f"{diagnostics['post_1995_peak_year']})"
        )
    print(
        "cumulative erosion - accretion: "
        f"{diagnostics['erosion_accretion_balance_km2']:.2f} km^2"
    )
    print(
        "mean centerline migration: "
        f"{diagnostics['mean_migration_metres_per_year']:.2f} m/yr"
    )
    print(f"figure: {arguments.output}")
    if arguments.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
