"""Run RivMAPy's single-year geometry workflow on the bundled Ucayali data.

From the repository root::

    python examples/static_geometry.py

The script reads ``data/riv.mat`` without modifying it, prints compact
diagnostics, and writes ``examples/output/static_geometry.png``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / "examples" / "output" / ".matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np

from rivmapy.geometry import angles, curvatures, smooth_polyline
from rivmapy.io import load_rivmap_mat
from rivmapy.mask import (
    banklines_from_mask,
    centerline_from_mask,
    width_from_banklines,
    width_from_mask,
)

def analyze_year(data_file: Path, year: int, mask_kind: str = "single_thread"):
    """Return the record and all static products used by the example."""

    records = load_rivmap_mat(data_file)
    try:
        record = next(record for record in records if record.year == year)
    except StopIteration as exc:
        available = f"{records[0].year}-{records[-1].year}"
        raise ValueError(f"year {year} is unavailable; fixture covers {available}") from exc
    mask = record.mask(mask_kind)
    centerline = centerline_from_mask(
        mask,
        record.exit_sides,
        record.nominal_width,
    )
    banks = banklines_from_mask(mask, record.exit_sides)
    bank_width = width_from_banklines(
        centerline.coordinates,
        banks.left,
        banks.right,
        record.nominal_width,
    )
    mask_width = width_from_mask(
        mask,
        centerline.coordinates,
        record.nominal_width / 2,
    )
    smoothed = smooth_polyline(centerline.coordinates, record.nominal_width)
    heading = angles(smoothed)
    curvature = curvatures(smoothed)
    return record, mask, centerline, banks, bank_width, mask_width, heading, curvature


def create_figure(
    record,
    mask,
    centerline,
    banks,
    bank_width,
    mask_width,
    heading,
    curvature,
):
    """Create the review figure and return it with scalar diagnostics."""

    increments = np.linalg.norm(np.diff(centerline.coordinates, axis=0), axis=1)
    distance = np.r_[0.0, np.cumsum(increments)]
    reach_width = float(mask.sum() / distance[-1])
    mean_bank_width = float(np.nanmean(bank_width))

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    planform = axes[0, 0]
    planform.imshow(mask, cmap="gray_r", interpolation="nearest")
    planform.plot(banks.left[:, 0], banks.left[:, 1], color="#2c7fb8", lw=0.8, label="left bank")
    planform.plot(banks.right[:, 0], banks.right[:, 1], color="#7fcdbb", lw=0.8, label="right bank")
    planform.plot(
        centerline.coordinates[:, 0],
        centerline.coordinates[:, 1],
        color="#d95f0e",
        lw=1.0,
        label="centerline",
    )
    planform.set_title(f"{record.year} single-thread mask and extracted geometry")
    planform.set_xlabel("x (pixel)")
    planform.set_ylabel("y (pixel)")
    planform.legend(loc="lower right", fontsize=8)

    axes[0, 1].plot(distance, bank_width, color="#2c7fb8", lw=0.9)
    axes[0, 1].axhline(mean_bank_width, color="#d95f0e", ls="--", lw=1)
    axes[0, 1].set(
        title=f"Normal bank width (mean {mean_bank_width:.2f} px)",
        xlabel="distance along centerline (pixel)",
        ylabel="width (pixel)",
    )

    axes[1, 0].plot(mask_width.distances, mask_width.widths, color="#41ab5d", lw=1)
    axes[1, 0].axhline(reach_width, color="#d95f0e", ls="--", lw=1)
    axes[1, 0].set(
        title=f"Mask area/length width (reach mean {reach_width:.2f} px)",
        xlabel="distance along centerline (pixel)",
        ylabel="width (pixel)",
    )

    diagnostics = axes[1, 1]
    diagnostics.plot(distance, np.rad2deg(heading), color="#756bb1", lw=0.8, label="heading")
    diagnostics.set(
        title="Smoothed centerline direction and curvature",
        xlabel="distance along centerline (pixel)",
        ylabel="heading (degree)",
    )
    curvature_axis = diagnostics.twinx()
    curvature_axis.plot(distance, curvature, color="#e6550d", lw=0.7, alpha=0.8, label="curvature")
    curvature_axis.set_ylabel("curvature (1/pixel)")
    lines = diagnostics.lines + curvature_axis.lines
    diagnostics.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)

    figure.suptitle("RivMAPy static-geometry demonstration", fontsize=16)
    return figure, {
        "reach_width": reach_width,
        "mean_bank_width": mean_bank_width,
        "centerline_length": float(distance[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPOSITORY_ROOT / "data" / "riv.mat")
    parser.add_argument("--year", type=int, default=1984)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "examples" / "output" / "static_geometry.png",
    )
    parser.add_argument("--show", action="store_true", help="open the figure after saving")
    arguments = parser.parse_args()

    products = analyze_year(arguments.data, arguments.year)
    figure, diagnostics = create_figure(*products)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180)
    print(f"year: {arguments.year}")
    print(f"centerline length: {diagnostics['centerline_length']:.2f} px")
    print(f"reach mask-area width: {diagnostics['reach_width']:.2f} px")
    print(f"mean bank-intersection width: {diagnostics['mean_bank_width']:.2f} px")
    print(f"figure: {arguments.output}")
    if arguments.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
