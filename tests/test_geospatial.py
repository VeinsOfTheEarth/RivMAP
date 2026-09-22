import numpy as np
import pytest

from affine import Affine

from rivmapy.geospatial import (
    combine_georeferenced_images,
    mosaic_rasters,
    pixel_to_world,
    stitch_polylines,
)


def test_mosaic_adjacent_rasters_and_transform():
    left = np.ones((2, 3), dtype=np.uint8)
    right = np.full((2, 2), 2, dtype=np.uint8)
    result = mosaic_rasters(
        [left, right],
        [Affine(10, 0, 100, 0, -10, 200), Affine(10, 0, 130, 0, -10, 200)],
        nodata=0,
    )
    np.testing.assert_array_equal(
        result.array,
        np.array([[1, 1, 1, 2, 2], [1, 1, 1, 2, 2]]),
    )
    assert result.transform == Affine(10, 0, 100, 0, -10, 200)
    assert result.coverage.all()


def test_mosaic_overlap_policy_and_nodata():
    transform = Affine(1, 0, 0, 0, -1, 2)
    first = np.array([[1.0, 1.0], [1.0, 1.0]])
    second = np.array([[np.nan, 2.0], [2.0, 2.0]])
    keep_first = mosaic_rasters(
        [first, second],
        [transform, transform],
        overlap="first",
        source_nodata=np.nan,
    )
    np.testing.assert_array_equal(keep_first.array, first)
    keep_last = mosaic_rasters(
        [first, second],
        [transform, transform],
        overlap="last",
        source_nodata=np.nan,
    )
    np.testing.assert_array_equal(keep_last.array, [[1, 2], [2, 2]])


def test_mosaic_gap_uses_coverage_with_no_nodata_value():
    first = np.ones((1, 1), dtype=np.uint8)
    second = np.full((1, 1), 2, dtype=np.uint8)
    result = mosaic_rasters(
        [first, second],
        [Affine(1, 0, 0, 0, -1, 1), Affine(1, 0, 2, 0, -1, 1)],
        nodata=None,
    )
    np.testing.assert_array_equal(result.array, [[1, 0, 2]])
    np.testing.assert_array_equal(result.coverage, [[True, False, True]])
    assert result.nodata is None


def test_mosaic_promotes_unsigned_input_for_negative_nodata():
    result = mosaic_rasters(
        [np.ones((1, 1), dtype=np.uint8)],
        [Affine(1, 0, 0, 0, -1, 1)],
        nodata=-9999,
    )
    assert np.issubdtype(result.array.dtype, np.signedinteger)


def test_mosaic_rejects_subpixel_offset():
    with pytest.raises(ValueError, match="sub-pixel"):
        mosaic_rasters(
            [np.ones((2, 2)), np.ones((2, 2))],
            [Affine(1, 0, 0, 0, -1, 2), Affine(1, 0, 1.5, 0, -1, 2)],
        )


def test_pixel_to_world_uses_pixel_centres_by_default():
    result = pixel_to_world([[0, 0], [2, 1]], Affine(10, 0, 100, 0, -10, 200))
    np.testing.assert_allclose(result, [[105, 195], [125, 185]])


def test_pixel_to_world_accepts_one_coordinate():
    result = pixel_to_world([[2, 1]], Affine(10, 0, 100, 0, -10, 200))
    np.testing.assert_allclose(result, [[125, 185]])


def test_stitch_polylines_orients_and_deduplicates():
    pieces = [
        np.array([[0, 0], [1, 0], [2, 0]]),
        np.array([[4, 0], [3, 0], [2, 0]]),
        np.array([[-2, 0], [-1, 0], [0, 0]]),
    ]
    combined = stitch_polylines(pieces, deduplicate_tolerance=0)
    np.testing.assert_array_equal(combined[:, 0], np.arange(-2, 5))


def test_compatibility_wrapper_transforms_lines_before_stitching():
    line_a = np.array([[0, 0], [1, 0]])
    line_b = np.array([[0, 0], [1, 0]])
    result = combine_georeferenced_images(
        [line_a, line_b],
        [Affine.translation(0, 0), Affine.translation(2, 0)],
        kind="polyline",
        deduplicate_tolerance=0,
    )
    np.testing.assert_allclose(result[:, 0], [0.5, 1.5, 2.5, 3.5])
