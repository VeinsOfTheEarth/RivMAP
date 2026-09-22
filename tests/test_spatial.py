from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("rasterio")
from affine import Affine
from shapely.geometry import Point

from rivmapy.spatial import GeoRaster, GeoVector, RasterGrid, SpatialResultMixin, mosaic_georasters


def test_crop_preserves_world_location_with_rotation_and_unequal_pixels():
    transform = Affine(30, 2, 500000, 3, -20, 9000000)
    grid = RasterGrid(transform, "EPSG:32718", (20, 30))
    data = GeoRaster(np.arange(600).reshape(20, 30), grid)
    cropped = data.crop(3, 16, 7, 25)
    np.testing.assert_array_equal(cropped.array, data.array[3:16, 7:25])
    np.testing.assert_allclose(
        cropped.grid.world_coordinates([[0, 0], [4, 5]]),
        grid.world_coordinates([[7, 3], [11, 8]]),
    )
    assert cropped.crs == data.crs


def test_grid_comparison_is_not_loose_at_large_world_coordinates():
    first = RasterGrid(Affine(30, 0, 5e8, 0, -30, 9e8), "EPSG:32718", (10, 20))
    shifted = RasterGrid(Affine(30, 0, 5e8 + 0.3, 0, -30, 9e8), "EPSG:32718", (10, 20))
    with pytest.raises(ValueError, match="alignment"):
        first.assert_aligned(shifted)
    with pytest.raises(ValueError, match="CRS"):
        first.assert_aligned(RasterGrid(first.transform, "EPSG:32719", first.shape))


def test_nodata_is_distinct_from_valid_false_pixels():
    valid = np.array([[True, True], [False, True]])
    grid = RasterGrid(Affine(30, 0, 500000, 0, -30, 9000000), "EPSG:32718", (2, 2), valid)
    raster = GeoRaster(np.array([[0, 1], [255, 0]]), grid, nodata=255)
    np.testing.assert_array_equal(raster.binary_array(), [[False, True], [False, False]])
    assert raster.valid_mask[0, 0]
    assert not raster.valid_mask[1, 0]
    derived = raster.with_array(np.zeros((2, 2), dtype=bool))
    assert derived.nodata is None
    np.testing.assert_array_equal(derived.valid_mask, valid)


def test_direct_nodata_constructor_marks_sentinel_unobserved():
    grid = RasterGrid(Affine.scale(30, -30), "EPSG:32718", (2, 2))
    raster = GeoRaster(np.array([[0, 1], [255, 0]], dtype=np.uint8), grid, nodata=255)
    np.testing.assert_array_equal(raster.binary_array(), [[False, True], [False, False]])
    assert not raster.valid_mask[1, 0]
    assert raster.valid_mask[0, 0]


def test_grid_scale_handles_projected_units_and_refuses_degree_lengths():
    grid = RasterGrid(Affine(30, 0, 500000, 0, -30, 9000000), "EPSG:32718", (2, 2))
    assert grid.pixel_size == 30
    assert grid.pixel_area == 900
    rotated = RasterGrid(Affine.rotation(20) * Affine.scale(30, -30), grid.crs, grid.shape)
    assert rotated.pixel_size == pytest.approx(30)
    geographic = RasterGrid(Affine.scale(0.1, -0.1), "EPSG:4326", grid.shape)
    with pytest.raises(ValueError, match="projected"):
        _ = geographic.pixel_area
    with pytest.raises(ValueError, match="projected"):
        _ = geographic.pixel_size
    anisotropic = RasterGrid(Affine.scale(10, -20), grid.crs, grid.shape)
    with pytest.raises(ValueError, match="square"):
        _ = anisotropic.pixel_size


def test_line_pixel_centers_and_polygon_pixel_edges_are_distinct():
    @dataclass
    class Result(SpatialResultMixin):
        coordinates: np.ndarray
        cutoffs: np.ndarray
        grid: RasterGrid

    grid = RasterGrid(Affine(30, 0, 500000, 0, -30, 9000000), "EPSG:32718", (2, 2))
    result = Result(np.array([[0, 0], [1, 1]]), np.array([[True, False], [False, False]]), grid)
    line = result.line()
    np.testing.assert_allclose(line.geometries[0].coords[0], [500015, 8999985])
    polygon = result.polygons().geometries[0]
    assert polygon.bounds == (500000, 8999970, 500030, 9000000)
    assert polygon.area == 900
    assert polygon.contains(Point(line.geometries[0].coords[0]))


def test_spatial_containers_do_not_modify_callers():
    valid = np.ones((2, 2), dtype=bool)
    data = np.zeros((2, 2), dtype=np.uint8)
    grid = RasterGrid(Affine.scale(30, -30), "EPSG:32718", (2, 2), valid)
    raster = GeoRaster(data, grid)
    valid[0, 0] = False
    data[0, 0] = 1
    assert grid.valid[0, 0]
    assert raster.array[0, 0] == 0


def test_invalid_spatial_containers_fail_explicitly():
    with pytest.raises(ValueError, match="CRS"):
        RasterGrid(Affine.identity(), None, (2, 2))
    with pytest.raises(ValueError, match="invertible"):
        RasterGrid(Affine.scale(0, 1), "EPSG:32718", (2, 2))
    with pytest.raises(ValueError, match="non-empty"):
        GeoVector([Point()], "EPSG:32718")


def test_mosaic_carries_crs_and_only_overwrites_with_valid_pixels():
    first = GeoRaster(np.array([[1, 0], [1, 0]], dtype=np.uint8),
        RasterGrid(Affine(30, 0, 500000, 0, -30, 9000000), "EPSG:32718", (2, 2)))
    second = GeoRaster(np.array([[1, 1], [1, 1]], dtype=np.uint8),
        RasterGrid(Affine(30, 0, 500030, 0, -30, 9000000), "EPSG:32718", (2, 2),
        np.array([[False, True], [True, False]])))
    result = mosaic_georasters([first, second])
    np.testing.assert_array_equal(result.array, [[1, 0, 1], [1, 1, 0]])
    np.testing.assert_array_equal(result.valid_mask, [[True, True, True], [True, True, False]])
    assert result.crs == first.crs
    assert result.transform == first.transform
    wrong_crs = GeoRaster(second.array, RasterGrid(second.transform, "EPSG:32719", (2, 2)))
    with pytest.raises(ValueError, match="CRS"):
        mosaic_georasters([first, wrong_crs])
