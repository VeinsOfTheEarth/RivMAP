"""Exercise the documented geospatial workflow against real files."""

from pathlib import Path
import runpy

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyogrio")

from rivmapy import read_geopackage, read_geotiff, read_mask


def test_documented_workflow_preserves_grid_and_native_crs(tmp_path):
    example = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "georeferenced_workflow.py")
    )
    first_path, second_path = example["synthetic_inputs"](tmp_path)
    output = tmp_path / "products"
    result = example["run_workflow"](first_path, second_path, output)
    first = read_mask(first_path)

    assert result["crs"] == "EPSG:32718"
    assert result["belt_cells"] == 10
    assert result["spatial_shape"] == (10, 2)
    assert result["valid_pixels"] == 96 * 180
    for field in ("centerline", "erosion", "accretion", "unchanged", "cutoffs", "migrated", "belt_envelope"):
        raster = read_geotiff(output / f"{field}.tif")
        first.grid.assert_aligned(raster.grid)
        np.testing.assert_array_equal(raster.valid_mask, first.valid_mask)

    vector_path = output / "results.gpkg"
    for layer in ("centerline", "left_bank", "right_bank", "bank_widths", "area_widths", "cutoffs", "belt_cells"):
        vectors = read_geopackage(vector_path, layer=layer)
        assert vectors.crs == first.crs
    cells = read_geopackage(vector_path, layer="belt_cells")
    assert [p["cell_index"] for p in cells.properties] == list(range(10))
    envelope = read_geotiff(output / "belt_envelope.tif")
    assert sum(g.area for g in cells.geometries) == pytest.approx(
        envelope.array.sum() * first.grid.pixel_area
    )
