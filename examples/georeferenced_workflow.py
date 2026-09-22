"""Demonstrate metadata preservation from GeoTIFF masks to GIS outputs.

Run from the repository root after ``pip install -e '.[geospatial]'``::

    python examples/georeferenced_workflow.py

The default creates two SYNTHETIC masks on an explicitly chosen projected
test grid. These coordinates are not georeferencing for the Ucayali fixture.
Use --mask-a and --mask-b for real, aligned GeoTIFF channel masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from affine import Affine

from rivmapy import (
    GeoRaster, RasterGrid, banklines_from_mask, centerline_from_mask,
    channel_belt_from_masks, crop_to_mask, migration_centerlines, migration_mask,
    read_geopackage, read_geotiff, read_mask, spatial_changes,
    width_from_banklines, width_from_mask,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_inputs(directory: Path) -> tuple[Path, Path]:
    """Create test observations with known map coordinates and a nodata margin."""
    shape = (100, 180)
    rows, columns = np.indices(shape)
    middle = 50 + 14 * np.sin(columns / 35)
    valid = np.ones(shape, dtype=bool)
    valid[:4] = False
    grid = RasterGrid(Affine(30, 0, 500000, 0, -30, 9000000), "EPSG:32718", shape, valid)
    paths = []
    for index, displacement in enumerate((0, 3)):
        mask = np.abs(rows - middle - displacement) <= 5
        path = directory / f"synthetic_input_{index + 1}.tif"
        GeoRaster(mask, grid).to_geotiff(path, overwrite=True)
        paths.append(path)
    return tuple(paths)


def run_workflow(
    mask_a: Path, mask_b: Path, output: Path, *, exit_sides: str = "WE", nominal_width: float = 10
) -> dict:
    """Run static, temporal, and spatial analysis and verify written products."""
    output.mkdir(parents=True, exist_ok=True)
    first, second = read_mask(mask_a), read_mask(mask_b)
    first.grid.assert_aligned(second.grid)
    cl_a = centerline_from_mask(first, exit_sides, nominal_width)
    cl_b = centerline_from_mask(second, exit_sides, nominal_width)
    banks = banklines_from_mask(first, exit_sides)
    widths = width_from_banklines(cl_a, banks, nominal_width=nominal_width)
    area_widths = width_from_mask(first, cl_a, nominal_width)
    change = migration_mask(first, second, nominal_width)
    swept = migration_centerlines(
        cl_a, cl_b, first.grid.shape, nominal_width,
        exit_sides_t1=exit_sides, exit_sides_t2=exit_sides,
    )
    cropped = crop_to_mask(first, exit_sides)

    cl_a.to_geotiff(output / "centerline.tif", overwrite=True)
    cropped.to_geotiff(output / "cropped_mask.tif", overwrite=True)
    for field in ("erosion", "accretion", "unchanged", "cutoffs"):
        change.to_geotiff(output / f"{field}.tif", field, overwrite=True)
    swept.to_geotiff(output / "migrated.tif", "migrated", overwrite=True)

    vector_path = output / "results.gpkg"
    cl_a.line().to_geopackage(vector_path, layer="centerline", overwrite=True)
    for field in ("left", "right"):
        banks.line(field).to_geopackage(vector_path, layer=f"{field}_bank", overwrite=True)
    width_properties = [
        {"width_px": float(w) if np.isfinite(w) else None, "distance_px": float(s)}
        for w, s in zip(widths.widths, widths.distances, strict=True)
    ]
    widths.points(properties=width_properties).to_geopackage(vector_path, layer="bank_widths", overwrite=True)
    segment_properties = [
        {"width_px": float(w) if np.isfinite(w) else None, "distance_px": float(s)}
        for w, s in zip(area_widths.widths, area_widths.distances, strict=True)
    ]
    area_widths.points("sample_coordinates", properties=segment_properties).to_geopackage(
        vector_path, layer="area_widths", overwrite=True
    )
    swept.polygons("cutoffs").to_geopackage(vector_path, layer="cutoffs", overwrite=True)

    # Small demonstration belt. These are deliberately explicit tuning values,
    # not a change to RivMAP's legacy defaults for real reaches.
    belt = channel_belt_from_masks(
        [first, second], exit_sides, nominal_width, 2 * nominal_width,
        dilation_factor=1.5, smoothing_factor=4, padding_factor=4,
    )
    belt.to_geotiff(output / "belt_envelope.tif", "envelope", overwrite=True)
    belt.polygons("cell_masks").to_geopackage(vector_path, layer="belt_cells", overwrite=True)
    spatial = spatial_changes(belt, [first, second], [cl_a, cl_b])

    reread = read_geotiff(output / "erosion.tif")
    first.grid.assert_aligned(reread.grid)
    np.testing.assert_array_equal(reread.valid_mask, change.grid.valid)
    np.testing.assert_array_equal(reread.array.astype(bool), change.erosion)
    vectors = read_geopackage(vector_path, layer="centerline")
    assert vectors.crs == first.crs
    np.testing.assert_allclose(vectors.geometries[0].coords, cl_a.world_coordinates(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        cropped.grid.world_coordinates([[0, 0]]),
        first.grid.world_coordinates([[cropped.left, cropped.top]]), rtol=0, atol=1e-8,
    )
    return {
        "crs": str(first.crs), "shape": first.grid.shape,
        "valid_pixels": int(change.grid.valid.sum()),
        "centerline_vertices": len(cl_a.coordinates),
        "belt_cells": len(belt.cell_masks),
        "spatial_shape": spatial.channel_area.shape,
        "geopackage": str(vector_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-a", type=Path)
    parser.add_argument("--mask-b", type=Path)
    parser.add_argument("--exit-sides", default="WE")
    parser.add_argument("--nominal-width", type=float, default=10.0, help="pixels")
    parser.add_argument("--output", type=Path, default=ROOT / "examples" / "output" / "georeferenced")
    args = parser.parse_args()
    if (args.mask_a is None) != (args.mask_b is None):
        parser.error("supply both --mask-a and --mask-b, or neither for synthetic inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mask_a is None:
        print("Synthetic test grid; these are not Ucayali map coordinates.")
        first, second = synthetic_inputs(args.output)
    else:
        first, second = args.mask_a, args.mask_b
    result = run_workflow(first, second, args.output, exit_sides=args.exit_sides, nominal_width=args.nominal_width)
    for key, value in result.items():
        print(f"{key}: {value}")
    print("Verified: native CRS, raster transform, valid zeros/nodata, crop offset, and vector map coordinates.")


if __name__ == "__main__":
    main()
