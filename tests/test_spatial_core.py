"""Georeferencing remains attached across the scientific analysis functions."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("rasterio")
from affine import Affine

from rivmapy.belt import ChannelBelt, channel_belt_from_masks
from rivmapy.change import migration_centerlines, migration_mask, spatial_changes
from rivmapy.mask import (
    CenterlineResult, WidthSamples, banklines_from_mask, centerline_from_mask,
    crop_to_mask, width_from_banklines, width_from_mask,
)
from rivmapy.spatial import GeoRaster, RasterGrid


def grid(shape=(81, 161), *, valid=None):
    return RasterGrid(Affine(30, 0, 400000, 0, -30, 8000000), "EPSG:32718", shape, valid)


def channel(*, offset=0, cropped=False):
    mask = np.zeros((81, 161), dtype=bool)
    mask[37 + offset:44 + offset, 10 if cropped else 0:151 if cropped else 161] = True
    return GeoRaster(mask, grid(mask.shape))


def line_result(coordinates, source_grid):
    return CenterlineResult(np.asarray(coordinates, dtype=float), np.zeros(source_grid.shape, dtype=bool), source_grid)


def test_crop_offsets_and_static_products_retain_their_grid():
    mask = channel(cropped=True)
    cropped = crop_to_mask(mask, "WE")
    assert cropped.left == 10
    assert cropped.right == 151
    assert cropped.grid.shape == cropped.mask.shape
    assert cropped.grid.transform == Affine(30, 0, 400300, 0, -30, 8000000)
    np.testing.assert_allclose(
        cropped.grid.world_coordinates([[0, 40]]), mask.grid.world_coordinates([[10, 40]])
    )
    centerline = centerline_from_mask(mask, "WE", 7)
    banks = banklines_from_mask(mask, "WE")
    assert centerline.grid is mask.grid
    assert banks.grid is mask.grid
    assert centerline.coordinates[:, 0].min() >= cropped.left
    np.testing.assert_allclose(
        centerline.world_coordinates(), mask.grid.world_coordinates(centerline.coordinates)
    )
    widths = width_from_banklines(centerline, banks, nominal_width=7)
    assert isinstance(widths, WidthSamples)
    mask.grid.assert_aligned(widths.grid)
    assert len(widths.widths) == len(centerline.coordinates)
    np.testing.assert_array_equal(widths.coordinates, centerline.coordinates)
    profile = width_from_mask(mask, centerline, spacing=30)
    mask.grid.assert_aligned(profile.grid)
    assert len(profile.breakpoints) == len(profile.widths) + 1
    assert profile.sample_coordinates.shape == (len(profile.widths), 2)


def test_crop_hole_filling_does_not_turn_nodata_into_channel():
    source = channel(cropped=True)
    valid = np.ones(source.grid.shape, dtype=bool)
    valid[39:42, 70:74] = False
    data = source.array.astype(np.uint8)
    data[~valid] = 255
    mask = GeoRaster(data, source.grid.with_valid_mask(valid), nodata=255)
    result = crop_to_mask(mask, "WE")
    assert not np.any(result.mask[~result.grid.valid])
    with pytest.raises(ValueError, match="nodata"):
        centerline_from_mask(mask, "WE", 7)


def test_mask_changes_use_only_jointly_observed_pixels():
    first = channel()
    valid = np.ones(first.grid.shape, dtype=bool)
    valid[:, 60:70] = False
    second_array = first.array.astype(np.uint8)
    second_array[~valid] = 255
    second = GeoRaster(second_array, first.grid.with_valid_mask(valid), nodata=255)
    result = migration_mask(first, second, nominal_width=7)
    assert not result.erosion.any()
    assert not result.accretion.any()
    assert not result.cutoffs.any()
    np.testing.assert_array_equal(result.unchanged, valid)
    np.testing.assert_array_equal(result.grid.valid, valid)
    assert result.raster("erosion").grid is result.grid


def test_unknown_bank_is_rejected_instead_of_returning_a_narrower_channel():
    source = channel()
    valid = np.ones(source.grid.shape, dtype=bool)
    valid[35:40, :] = False
    observed = GeoRaster(source.array, source.grid.with_valid_mask(valid))
    full_centerline = centerline_from_mask(source, "WE", 7)
    full_banks = banklines_from_mask(source, "WE")
    with pytest.raises(ValueError, match="nodata touches"):
        centerline_from_mask(observed, "WE", 7)
    with pytest.raises(ValueError, match="nodata touches"):
        banklines_from_mask(observed, "WE")
    with pytest.raises(ValueError, match="nodata touches"):
        width_from_mask(observed, full_centerline, 30)
    with pytest.raises(ValueError, match="nodata touches"):
        width_from_banklines(
            replace(full_centerline, grid=observed.grid),
            replace(full_banks, grid=observed.grid), nominal_width=7,
        )
    with pytest.raises(ValueError, match="nodata touches"):
        channel_belt_from_masks([source, observed], "WE", 7, 20)


def test_reflected_grid_is_rejected_before_left_and_right_can_be_mislabeled():
    source = channel()
    reflected_grid = replace(source.grid, transform=Affine(30, 0, 400000, 0, 30, 8000000))
    reflected = GeoRaster(source.array, reflected_grid)
    with pytest.raises(ValueError, match="handedness"):
        banklines_from_mask(reflected, "WE")
    with pytest.raises(ValueError, match="handedness"):
        centerline_from_mask(reflected, "WE", 7)
    # Pixel classifications do not depend on left/right orientation.
    result = migration_mask(reflected, reflected, 7)
    assert result.grid.transform == reflected_grid.transform


def test_explicit_grid_cannot_override_unknown_bank_in_centerline_metadata():
    source = channel()
    centerline = centerline_from_mask(source, "WE", 7)
    banks = banklines_from_mask(source, "WE")
    valid = np.ones(source.grid.shape, dtype=bool)
    valid[35:40, :] = False
    centerline = replace(centerline, grid=source.grid.with_valid_mask(valid))
    with pytest.raises(ValueError, match="nodata touches"):
        width_from_banklines(centerline, banks, nominal_width=7, grid=source.grid)


def test_incompletely_observed_abandoned_component_is_not_an_automatic_cutoff():
    source_grid = grid((15, 15))
    first = np.zeros(source_grid.shape, dtype=bool)
    first[4:11, 4:11] = True
    second = np.zeros_like(first)
    valid = np.ones_like(first)
    valid[:, 8:] = False
    first_raster = GeoRaster(first, source_grid)
    second_raster = GeoRaster(second, source_grid.with_valid_mask(valid))
    result = migration_mask(first_raster, second_raster, nominal_width=1)
    assert not result.cutoffs.any()
    np.testing.assert_array_equal(result.accretion, first & valid)
    seed = np.zeros_like(first)
    seed[6, 6] = True
    seeded = migration_mask(first_raster, second_raster, 1, GeoRaster(seed, source_grid))
    np.testing.assert_array_equal(seeded.cutoffs, first & valid)


@pytest.mark.parametrize("kind", ["shift", "crs", "shape"])
def test_temporal_operations_reject_same_array_with_incompatible_spatial_grid(kind):
    first = channel()
    if kind == "shift":
        other_grid = replace(first.grid, transform=Affine(30, 0, 400007.5, 0, -30, 8000000))
        match = "alignment"
    elif kind == "crs":
        other_grid = replace(first.grid, crs="EPSG:32618")
        match = "CRS"
    else:
        other_grid = replace(first.grid, shape=(80, 161))
        match = "shape"
    other = GeoRaster(np.zeros(other_grid.shape, dtype=bool), other_grid)
    with pytest.raises(ValueError, match=match):
        migration_mask(first, other, 7)
    with pytest.raises(ValueError, match=match):
        channel_belt_from_masks([first, other], "WE", 7, 20)


def test_mixed_temporal_inputs_and_misaligned_cutoff_seeds_are_rejected():
    mask = channel()
    with pytest.raises(ValueError, match="both masks"):
        migration_mask(mask, mask.array, 7)
    with pytest.raises(ValueError, match="all channel masks"):
        channel_belt_from_masks([mask, mask.array], "WE", 7, 20)
    with pytest.raises(ValueError, match="alignment"):
        seed_grid = replace(mask.grid, transform=Affine(30, 0, 400030, 0, -30, 8000000))
        migration_mask(mask, mask, 7, GeoRaster(mask.array, seed_grid))


def test_centerline_swept_area_excludes_nodata():
    source_grid = grid((12, 12))
    valid = np.ones(source_grid.shape, dtype=bool)
    valid[3:8, 4:6] = False
    first = line_result([[3, 0], [3, 11]], source_grid)
    second = line_result([[5, 0], [5, 11]], source_grid.with_valid_mask(valid))
    result = migration_centerlines(first, second, source_grid.shape, 2)
    assert result.migrated.any()
    assert not np.any(result.migrated[~valid])
    np.testing.assert_array_equal(result.grid.valid, valid)
    with pytest.raises(ValueError, match="both centerlines"):
        migration_centerlines(first, second.coordinates, source_grid.shape, 2)


def test_cutoff_area_counts_only_observed_pixels_and_retains_index_rows():
    source_grid = grid((25, 21))
    first_xy = [[0, 10], [5, 10], [5, 20], [15, 20], [15, 10], [20, 10]]
    second_xy = [[0, 8], [5, 10], [15, 10], [20, 8]]
    valid = np.ones(source_grid.shape, dtype=bool)
    valid[12:16, 7:12] = False
    first = line_result(first_xy, source_grid)
    second = line_result(second_xy, source_grid.with_valid_mask(valid))
    result = migration_centerlines(
        first, second, source_grid.shape, 3, exit_sides_t1="WE", exit_sides_t2="WE"
    )
    assert len(result.cutoff_area) == len(result.cutoff_indices) == 1
    assert result.cutoff_area.sum() == np.count_nonzero(result.cutoffs)
    assert not np.any(result.cutoffs[~valid])
    np.testing.assert_array_equal(result.cutoff_indices, [[1, 4]])


def test_spatial_belt_and_summaries_share_common_valid_extent():
    first = channel()
    second = channel(offset=3)
    valid = np.ones(first.grid.shape, dtype=bool)
    valid[25:28, 50:55] = False
    second = GeoRaster(second.array, second.grid.with_valid_mask(valid))
    belt = channel_belt_from_masks(
        [first, second], "WE", 7, 20, dilation_factor=1.5,
        smoothing_factor=4, padding_factor=4,
    )
    assert isinstance(belt, ChannelBelt)
    np.testing.assert_array_equal(belt.grid.valid, valid)
    assert not np.any(belt.envelope[~valid])
    np.testing.assert_array_equal(belt.cell_masks.any(axis=0), belt.envelope)
    lines = [centerline_from_mask(mask, "WE", 7) for mask in (first, second)]
    migration = migration_mask(first, second, 7)
    summaries = spatial_changes(
        belt, [first, second], lines,
        {"erosion": [migration.raster("erosion"), migration.raster("erosion")]},
    )
    np.testing.assert_array_equal(summaries.grid.valid, valid)
    np.testing.assert_array_equal(summaries.cell_area, belt.cell_masks.sum(axis=(1, 2)))
    assert summaries.channel_area[:, 0].sum() == first.array.sum()
    assert summaries.analyzed_area["erosion"][:, 0].sum() == migration.erosion.sum()
    with pytest.raises(ValueError, match="all channel masks"):
        spatial_changes(belt, [first.array], [lines[0]])


def test_spatial_summary_lengths_skip_unobserved_pixel_footprints():
    source_grid = grid((7, 6))
    valid = np.ones(source_grid.shape, dtype=bool)
    valid[2:4, :] = False
    mask = GeoRaster(np.ones(source_grid.shape, dtype=bool), source_grid.with_valid_mask(valid))
    centerline = line_result([[2, 0], [2, 6]], mask.grid)
    cells = np.ones((1, 7, 6), dtype=bool)
    result = spatial_changes(cells, [mask], [centerline])
    assert result.channel_area[0, 0] == 30
    assert result.cell_area[0] == 30
    assert result.centerline_length[0, 0] == pytest.approx(4)
