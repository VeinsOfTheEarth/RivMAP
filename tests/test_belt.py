"""Synthetic tests for automatic channel-belt construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import line

from rivmapy.belt import ChannelBelt, channel_belt_from_masks


def _channel_mask(
    shape: tuple[int, int], center_y: np.ndarray, half_width: float = 3.0
) -> np.ndarray:
    rows = np.arange(shape[0], dtype=float)[:, None]
    return np.abs(rows - center_y[None, :]) <= half_width


def _straight_series() -> list[np.ndarray]:
    shape = (81, 161)
    x = np.arange(shape[1], dtype=float)
    return [
        _channel_mask(shape, np.full_like(x, 37.0)),
        _channel_mask(shape, np.full_like(x, 43.0)),
    ]


def _curved_series() -> list[np.ndarray]:
    shape = (101, 181)
    x = np.arange(shape[1], dtype=float)
    phase = 2.0 * np.pi * x / (shape[1] - 1)
    return [
        _channel_mask(shape, 50.0 + 15.0 * np.sin(phase)),
        _channel_mask(shape, 50.0 + 15.0 * np.sin(phase + 0.18)),
        _channel_mask(shape, 50.0 + 13.0 * np.sin(phase - 0.15)),
    ]


def _corner_series() -> list[np.ndarray]:
    shape = (100, 120)
    masks: list[np.ndarray] = []
    for offset in (0, 4):
        mask = np.zeros(shape, dtype=bool)
        rows, columns = line(20 + offset, 0, shape[0] - 1, 75 + offset)
        mask[rows, columns] = True
        masks.append(ndi.binary_dilation(mask, iterations=3))
    return masks


def _build(
    masks: list[np.ndarray], exit_sides: str = "WE"
) -> ChannelBelt:
    return channel_belt_from_masks(
        masks,
        exit_sides,
        nominal_width=7.0,
        spacing=20.0,
        dilation_factor=1.5,
        smoothing_factor=4.0,
        dilation_increment_factor=0.5,
        padding_factor=4.0,
        max_iterations=8,
    )


def _assert_partition_and_coverage(
    result: ChannelBelt, masks: list[np.ndarray]
) -> None:
    union = np.logical_or.reduce(masks)
    assert np.all(result.envelope[union])
    assert np.array_equal(np.any(result.cell_masks, axis=0), result.envelope)
    assert np.all(np.count_nonzero(result.cell_masks, axis=0) <= 1)
    assert np.all(np.count_nonzero(result.cell_masks, axis=(1, 2)) > 0)


def test_straight_belt_is_oriented_partitioned_and_nearly_even() -> None:
    masks = _straight_series()

    result = _build(masks)

    _assert_partition_and_coverage(result, masks)
    assert result.left.shape == result.right.shape == result.centerline.shape
    assert len(result.left) == len(result.cell_masks) + 1
    assert len(result.midpoint_distances) == len(result.cell_masks)
    assert np.all(np.diff(result.centerline[:, 0]) > 0)
    assert np.all(result.left[:, 1] < result.right[:, 1])
    station_lengths = np.linalg.norm(np.diff(result.centerline, axis=0), axis=1)
    assert np.all((station_lengths > 0.75 * 20.0) & (station_lengths < 1.25 * 20.0))
    assert np.all(np.diff(result.midpoint_distances) > 0)


def test_gently_curved_time_series_has_complete_disjoint_cells() -> None:
    masks = _curved_series()

    result = _build(masks)

    _assert_partition_and_coverage(result, masks)
    assert np.ptp(result.centerline[:, 1]) > 10.0
    widths = np.linalg.norm(result.right - result.left, axis=1)
    assert np.all(widths > 2.0)
    assert np.all(np.isfinite(result.midpoint_distances))


def test_reversing_flow_reverses_axis_and_swaps_banks() -> None:
    masks = _straight_series()

    west_to_east = _build(masks, "WE")
    east_to_west = _build(masks, "EW")

    np.testing.assert_array_equal(west_to_east.envelope, east_to_west.envelope)
    np.testing.assert_allclose(
        west_to_east.centerline, east_to_west.centerline[::-1], atol=1e-7
    )
    np.testing.assert_allclose(
        west_to_east.left, east_to_west.right[::-1], atol=1e-7
    )
    np.testing.assert_allclose(
        west_to_east.right, east_to_west.left[::-1], atol=1e-7
    )


def test_adjacent_exits_use_monotone_bank_pairing_when_normals_are_ambiguous() -> None:
    masks = _corner_series()

    result = _build(masks, "SW")

    _assert_partition_and_coverage(result, masks)
    # South is upstream and west is downstream.
    assert result.centerline[0, 1] > result.centerline[-1, 1]
    assert result.centerline[0, 0] > result.centerline[-1, 0]
    assert np.all(np.diff(result.midpoint_distances) > 0)


def test_result_dataclass_is_frozen() -> None:
    result = _build(_straight_series())

    with pytest.raises(FrozenInstanceError):
        result.envelope = np.zeros_like(result.envelope)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("masks", "exit_sides", "width", "spacing", "match"),
    [
        ([], "WE", 7, 20, "at least one"),
        ([np.ones((5, 5), dtype=bool)], "WW", 7, 20, "distinct"),
        ([np.ones((5, 5), dtype=bool)], "WE", 0, 20, "greater than zero"),
        ([np.ones((5, 5), dtype=bool)], "WE", 7, 0.5, "at least one pixel"),
        ([np.zeros((5, 5), dtype=bool)], "WE", 7, 20, "channel pixels"),
        ([np.ones((5, 5, 1), dtype=bool)], "WE", 7, 20, "2-D"),
        (
            [np.ones((5, 5), dtype=bool), np.ones((6, 5), dtype=bool)],
            "WE",
            7,
            20,
            "same shape",
        ),
    ],
)
def test_input_validation(
    masks: list[np.ndarray],
    exit_sides: str,
    width: float,
    spacing: float,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        channel_belt_from_masks(masks, exit_sides, width, spacing)


def test_each_mask_must_touch_both_declared_exits() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[8:12, 3:-3] = True

    with pytest.raises(ValueError, match="does not touch declared W exit"):
        channel_belt_from_masks(mask[None, ...], "WE", 4, 10)


def test_iteration_and_memory_guards_are_validated() -> None:
    masks = _straight_series()

    with pytest.raises(ValueError, match="positive integer"):
        channel_belt_from_masks(masks, "WE", 7, 20, max_iterations=0)
    with pytest.raises(ValueError, match="max_padded_pixels"):
        channel_belt_from_masks(
            masks,
            "WE",
            7,
            20,
            padding_factor=20,
            max_padded_pixels=100,
        )
