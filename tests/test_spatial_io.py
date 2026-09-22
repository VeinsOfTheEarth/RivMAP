"""Round trips exercise georeferencing and validity, not only file creation."""

import numpy as np
import pytest
import shapely

rasterio = pytest.importorskip("rasterio")
pyogrio = pytest.importorskip("pyogrio")
from affine import Affine
from shapely.geometry import LineString, Point

from rivmapy.spatial import GeoRaster, GeoVector, RasterGrid
from rivmapy.spatial_io import read_geopackage, read_geotiff, read_mask, write_geopackage, write_geotiff


@pytest.mark.parametrize("transform", [
    Affine(12, 0, 381123, 0, -23, 7954321),
    Affine(12, 2, 381123, 3, -23, 7954321),
])
def test_geotiff_roundtrip_preserves_grid_values_and_valid_zeros(tmp_path, transform):
    array = np.asarray([[0, 1, 7], [0, 2, 3]], dtype=np.uint8)
    valid = np.asarray([[True, True, False], [True, False, True]])
    original = GeoRaster(array, RasterGrid(transform, "EPSG:32718", array.shape, valid))
    path = write_geotiff(tmp_path / "mask.tif", original)
    actual = read_geotiff(path)
    assert actual.crs == original.crs
    assert actual.transform == original.transform
    np.testing.assert_array_equal(actual.array, array)
    np.testing.assert_array_equal(actual.valid_mask, valid)
    assert actual.nodata is None
    assert not path.with_suffix(".tif.msk").exists()


def test_boolean_mask_roundtrip_and_crop_grid(tmp_path):
    transform = Affine(12, 0, 381123, 0, -23, 7954321)
    # This affine is the parent raster's column=2,row=1 crop: its first
    # pixel center must still occupy the parent's original pixel center.
    cropped_transform = Affine(12, 0, 381147, 0, -23, 7954298)
    array = np.asarray([[True, False], [False, True]])
    original = GeoRaster(array, RasterGrid(cropped_transform, "EPSG:32718", array.shape))
    path = write_geotiff(tmp_path / "crop.tif", original)
    actual = read_mask(path)
    np.testing.assert_array_equal(actual.array, array)
    assert actual.array.dtype == bool
    assert rasterio.transform.xy(actual.transform, 0, 0) == rasterio.transform.xy(transform, 1, 2)


def test_geotiff_nodata_read_mask_and_overwrite(tmp_path):
    array = np.asarray([[0, 255], [1, 0]], dtype=np.uint8)
    valid = array != 255
    original = GeoRaster(array, RasterGrid(Affine(5, 0, 800, 0, -5, 1200), "EPSG:32618", array.shape, valid), nodata=255)
    path = write_geotiff(tmp_path / "nodata.tif", original)
    with pytest.raises(FileExistsError):
        write_geotiff(path, original)
    write_geotiff(path, original, overwrite=True)
    actual = read_geotiff(path)
    assert actual.nodata == 255
    np.testing.assert_array_equal(actual.valid_mask, valid)
    binary = read_mask(path)
    np.testing.assert_array_equal(binary.array, [[False, False], [True, False]])
    np.testing.assert_array_equal(binary.valid_mask, valid)
    assert binary.nodata is None


@pytest.mark.parametrize("nodata", [256, -1, 0.5, np.nan])
def test_geotiff_rejects_colliding_or_unrepresentable_nodata(tmp_path, nodata):
    array = np.zeros((2, 2), dtype=np.uint8)
    original = GeoRaster(array, RasterGrid(Affine(5, 0, 800, 0, -5, 1200), "EPSG:32618", array.shape), nodata=nodata)
    with pytest.raises(ValueError, match="nodata"):
        write_geotiff(tmp_path / "bad.tif", original)


def test_direct_nodata_sentinel_excludes_pixels(tmp_path):
    array = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    original = GeoRaster(array, RasterGrid(Affine(5, 0, 800, 0, -5, 1200), "EPSG:32618", array.shape), nodata=0)
    actual = read_geotiff(write_geotiff(tmp_path / "zero_nodata.tif", original))
    np.testing.assert_array_equal(actual.valid_mask, array != 0)
    assert actual.nodata == 0


@pytest.mark.parametrize("values", [[[0, 1]], [[0, 255]], [[0, 0]], [[255, 255]]])
def test_read_mask_standard_encodings(tmp_path, values):
    array = np.asarray(values, dtype=np.uint8)
    original = GeoRaster(array, RasterGrid(Affine(5, 0, 800, 0, -5, 1200), "EPSG:32618", array.shape))
    path = write_geotiff(tmp_path / "binary.tif", original)
    np.testing.assert_array_equal(read_mask(path).array, array != 0)


def test_read_mask_requires_explicit_class(tmp_path):
    array = np.asarray([[0, 2, 3]], dtype=np.uint8)
    original = GeoRaster(array, RasterGrid(Affine(5, 0, 800, 0, -5, 1200), "EPSG:32618", array.shape))
    path = write_geotiff(tmp_path / "classes.tif", original)
    with pytest.raises(ValueError, match="channel_value"):
        read_mask(path)
    np.testing.assert_array_equal(read_mask(path, channel_value=2).array, [[False, True, False]])
    np.testing.assert_array_equal(read_mask(path, channel_value=0).array, [[True, False, False]])


def test_read_geotiff_requires_crs_and_valid_band(tmp_path):
    path = tmp_path / "no_crs.tif"
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint8", transform=Affine(5, 0, 800, 0, -5, 1200)) as target:
        target.write(np.ones((2, 2), dtype=np.uint8), 1)
    with pytest.raises(ValueError, match="CRS"):
        read_geotiff(path)
    for band in [0, -1, 2, True]:
        with pytest.raises(ValueError, match="band"):
            read_geotiff(path, band=band)


def test_read_geotiff_rejects_gcps_without_affine(tmp_path):
    path = tmp_path / "gcps.tif"
    from rasterio.control import GroundControlPoint
    gcps = [GroundControlPoint(row=0, col=0, x=100, y=500),
            GroundControlPoint(row=0, col=2, x=120, y=500),
            GroundControlPoint(row=2, col=0, x=100, y=480)]
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint8", gcps=gcps, crs="EPSG:32618") as target:
        target.write(np.ones((2, 2), dtype=np.uint8), 1)
    with pytest.raises(ValueError, match="GCP/RPC"):
        read_geotiff(path)


def test_geotiff_float_nodata_roundtrip(tmp_path):
    array = np.asarray([[0.0, np.nan], [1.25, 2.5]], dtype=np.float32)
    valid = np.isfinite(array)
    original = GeoRaster(array, RasterGrid(Affine(5, 0, 800, 0, -5, 1200), "EPSG:32618", array.shape, valid), nodata=np.nan)
    actual = read_geotiff(write_geotiff(tmp_path / "float.tif", original))
    np.testing.assert_array_equal(actual.array, array)
    np.testing.assert_array_equal(actual.valid_mask, valid)
    assert np.isnan(actual.nodata)


def test_geotiff_overwrite_replaces_old_external_mask(tmp_path):
    path = tmp_path / "old_external.tif"
    transform = Affine(5, 0, 800, 0, -5, 1200)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                           dtype="uint8", transform=transform, crs="EPSG:32618") as target:
            target.write(np.ones((2, 2), dtype=np.uint8), 1)
            target.write_mask(np.zeros((2, 2), dtype=np.uint8))
    original = GeoRaster(np.zeros((2, 2), dtype=bool), RasterGrid(transform, "EPSG:32618", (2, 2)))
    write_geotiff(path, original, overwrite=True)
    actual = read_geotiff(path)
    assert actual.valid_mask.all()
    assert not actual.array.any()


def test_geopackage_roundtrip_native_projected_coordinates_and_properties(tmp_path):
    geometries = (
        LineString([(381123.25, 7954321.75), (381145.5, 7954300.0)]),
        LineString([(381110.0, 7954250.0), (381190.0, 7954178.0)]),
    )
    properties = (
        {"year": 1984, "distance": 17.25, "name": "Ucayali", "nullable": None, "count": 3, "flag": True},
        {"year": 2015, "distance": None, "name": "Río", "nullable": None, "count": None, "flag": False},
    )
    original = GeoVector(geometries, "EPSG:32718", properties)
    path = write_geopackage(tmp_path / "centerlines.gpkg", original, layer="centerlines")
    actual = read_geopackage(path)
    assert actual.crs == original.crs
    assert actual.properties == original.properties
    assert all(shapely.equals_exact(a, b, tolerance=0) for a, b in zip(actual.geometries, geometries))
    assert isinstance(actual.properties[0]["count"], int)


def test_geopackage_layer_overwrite_preserves_other_layers(tmp_path):
    vector = GeoVector((Point(4, 5),), "EPSG:4326", ({"id": 1},))
    updated = GeoVector((Point(6, 7),), "EPSG:4326", ({"id": 2},))
    path = write_geopackage(tmp_path / "layers.gpkg", vector, layer="first")
    with pytest.raises(FileExistsError):
        write_geopackage(path, vector, layer="first")
    write_geopackage(path, vector, layer="second")
    write_geopackage(path, updated, layer="first", overwrite=True)
    assert read_geopackage(path, layer="first").properties == updated.properties
    assert read_geopackage(path, layer="second").properties == vector.properties
    with pytest.raises(ValueError, match="specify layer"):
        read_geopackage(path)


def test_geopackage_large_identifiers_are_exact_or_rejected(tmp_path):
    vector = GeoVector((Point(4, 5),), "EPSG:4326", ({"id": 2**60 + 1},))
    path = write_geopackage(tmp_path / "large.gpkg", vector)
    assert read_geopackage(path).properties == vector.properties
    nullable = GeoVector((Point(4, 5), Point(6, 7)), "EPSG:4326", ({"id": 2**60 + 1}, {"id": None}))
    with pytest.raises(ValueError, match="identifiers as strings"):
        write_geopackage(tmp_path / "nullable.gpkg", nullable)


def test_geopackage_property_fid_name_is_not_lost(tmp_path):
    vector = GeoVector((Point(4, 5),), "EPSG:4326", ({"fid": 42, "_rivmapy_fid": 9},))
    path = write_geopackage(tmp_path / "fid.gpkg", vector)
    assert read_geopackage(path).properties == vector.properties


def test_geopackage_empty_result_roundtrip(tmp_path):
    vector = GeoVector((), "EPSG:32718")
    path = write_geopackage(tmp_path / "no_cutoffs.gpkg", vector)
    actual = read_geopackage(path)
    assert actual.crs == vector.crs
    assert actual.geometries == ()
    assert actual.properties == ()
