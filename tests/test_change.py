"""Tests for temporal change and spatial aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from rivmapy.change import (
    migration_centerlines,
    migration_mask,
    spatial_cells_from_edges,
    spatial_changes,
)


def test_migration_mask_one_pixel_translation_is_exact() -> None:
    first = np.zeros((7, 9), dtype=bool)
    second = np.zeros_like(first)
    first[2:5, 2:6] = True
    second[2:5, 3:7] = True

    result = migration_mask(first, second, nominal_width=10)

    expected_erosion = np.zeros_like(first)
    expected_erosion[2:5, 6] = True
    expected_accretion = np.zeros_like(first)
    expected_accretion[2:5, 2] = True
    np.testing.assert_array_equal(result.erosion, expected_erosion)
    np.testing.assert_array_equal(result.accretion, expected_accretion)
    np.testing.assert_array_equal(result.unchanged, first == second)
    assert not result.cutoffs.any()


def test_migration_mask_area_threshold_is_strict_and_configurable() -> None:
    first = np.zeros((8, 12), dtype=bool)
    second = np.zeros_like(first)
    first[1:3, 1:5] = True  # area 8: not greater than the legacy threshold
    first[4:7, 7:10] = True  # area 9: cutoff

    result = migration_mask(first, second, nominal_width=2)

    assert result.accretion[1:3, 1:5].all()
    assert not result.cutoffs[1:3, 1:5].any()
    assert result.cutoffs[4:7, 7:10].all()
    assert not result.accretion[4:7, 7:10].any()


def test_supplied_cutoff_seed_selects_and_removes_whole_component() -> None:
    first = np.zeros((8, 12), dtype=bool)
    second = np.zeros_like(first)
    first[1:4, 1:4] = True
    first[4:7, 8:11] = True
    seed = np.zeros_like(first)
    seed[2, 2] = True

    result = migration_mask(first, second, nominal_width=100, cutoff_mask=seed)

    assert result.cutoffs[1:4, 1:4].all()
    assert not result.accretion[1:4, 1:4].any()
    assert result.accretion[4:7, 8:11].all()
    assert not result.cutoffs[4:7, 8:11].any()


def test_parallel_centerlines_produce_a_consistent_swept_strip() -> None:
    first = np.array([[3.0, 0.0], [3.0, 11.0]])
    second = np.array([[5.0, 0.0], [5.0, 11.0]])

    result = migration_centerlines(first, second, (12, 12), nominal_width=2)

    assert result.migrated.dtype == np.bool_
    assert result.migrated.shape == (12, 12)
    assert not result.migrated[:, 3].any()  # time-1 line is excluded
    assert result.migrated[:, 5].all()  # time-2 line is included
    assert 20 <= np.count_nonzero(result.migrated) <= 26
    assert not result.cutoffs.any()
    assert result.cutoff_indices.shape == (0, 2)


def test_simple_neck_cutoff_has_area_and_expected_lengths() -> None:
    # The old centerline takes a 30-pixel U-shaped route between the two
    # crossings; the new chute is a 10-pixel straight connection.
    first = np.array(
        [[0, 10], [5, 10], [5, 20], [15, 20], [15, 10], [20, 10]],
        dtype=float,
    )
    second = np.array([[0, 8], [5, 10], [15, 10], [20, 8]], dtype=float)

    result = migration_centerlines(
        first,
        second,
        (25, 21),
        nominal_width=3,
        exit_sides_t1="WE",
        exit_sides_t2="WE",
    )

    assert len(result.cutoff_area) == 1
    np.testing.assert_array_equal(result.cutoff_indices, [[1, 4]])
    assert result.cutoff_length[0] == pytest.approx(30.0)
    assert result.chute_length[0] == pytest.approx(10.0)
    assert result.cutoff_area[0] > 50
    assert not np.any(result.migrated & result.cutoffs)
    assert np.count_nonzero(result.migrated | result.cutoffs) > 80


def test_spatial_cells_and_zonal_core_count_area_and_length() -> None:
    left = np.array([[0, 0], [0, 3], [0, 6]], dtype=float)
    right = np.array([[5, 0], [5, 3], [5, 6]], dtype=float)
    cells = spatial_cells_from_edges(left, right, (7, 6))
    assert not np.any(cells[0] & cells[1])

    channel = np.ones((7, 6), dtype=bool)
    changed = np.zeros_like(channel)
    changed[1, 2] = True
    changed[5, 2] = True
    centerline = np.array([[2, 0], [2, 6]], dtype=float)

    result = spatial_changes(
        cells,
        [channel],
        [centerline],
        {"migration": [changed]},
    )

    np.testing.assert_array_equal(result.channel_area[:, 0], result.cell_area)
    np.testing.assert_array_equal(result.analyzed_area["migration"][:, 0], [1, 1])
    assert result.centerline_length[:, 0].sum() == pytest.approx(6.0)
    assert np.all(result.centerline_length[:, 0] > 0)


def test_temporal_inputs_are_not_mutated_and_shapes_are_checked() -> None:
    first = np.eye(4, dtype=bool)
    second = first.copy()
    before = first.copy()
    migration_mask(first, second, nominal_width=1)
    np.testing.assert_array_equal(first, before)

    with pytest.raises(ValueError, match="matching shapes"):
        migration_mask(first, np.zeros((3, 4), dtype=bool), nominal_width=1)

