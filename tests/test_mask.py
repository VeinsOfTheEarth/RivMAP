from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rivmapy.mask import (
    BanklineResult,
    CenterlineResult,
    CropResult,
    WidthProfile,
    banklines_from_mask,
    centerline_from_mask,
    crop_to_mask,
    skeleton_coords,
    width_from_banklines,
    width_from_mask,
)


def horizontal_channel(
    *, height: int = 80, width: int = 181, half_width: int = 10
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    middle = height // 2
    mask[middle - half_width : middle + half_width, 5 : width - 5] = True
    return mask


def vertical_channel(
    *, height: int = 181, width: int = 80, half_width: int = 10
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    middle = width // 2
    mask[5 : height - 5, middle - half_width : middle + half_width] = True
    return mask


def curved_channel(
    *, height: int = 100, width: int = 201, half_width: int = 7
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    x = np.arange(width)
    middle = height / 2 + 16 * np.sin(2 * np.pi * x / (width - 1))
    for column, center in enumerate(np.rint(middle).astype(int)):
        lower = max(0, center - half_width)
        upper = min(height, center + half_width + 1)
        mask[lower:upper, column] = True
    return mask


def test_crop_to_mask_uses_only_declared_exit_dimensions_and_fills_holes() -> None:
    mask = horizontal_channel()
    mask[37:43, 80:90] = False
    original = mask.copy()

    result = crop_to_mask(mask, "WE")

    assert isinstance(result, CropResult)
    assert (result.top, result.bottom) == (0, mask.shape[0])
    assert (result.left, result.right) == (5, mask.shape[1] - 5)
    assert result.mask.dtype == np.bool_
    assert result.mask[40, 80]
    assert_array_equal(mask, original)


def test_skeleton_coords_traces_diagonal_with_eight_connectivity() -> None:
    skeleton = np.eye(25, dtype=bool)

    coordinates = skeleton_coords(skeleton, "NS")

    assert coordinates.shape == (25, 2)
    assert_allclose(coordinates[:, 0], coordinates[:, 1])
    assert tuple(coordinates[0]) == (0.0, 0.0)
    assert tuple(coordinates[-1]) == (24.0, 24.0)


@pytest.mark.parametrize("bad_sides", ["N", "NN", "NX", "NES", ""])
def test_invalid_exit_sides_are_rejected(bad_sides: str) -> None:
    with pytest.raises(ValueError):
        crop_to_mask(horizontal_channel(), bad_sides)


def test_skeleton_coords_rejects_disconnected_and_branched_paths() -> None:
    disconnected = np.zeros((12, 12), dtype=bool)
    disconnected[1:5, 2] = True
    disconnected[7:11, 9] = True
    with pytest.raises(ValueError, match="connected"):
        skeleton_coords(disconnected, "NS")

    branched = np.zeros((15, 15), dtype=bool)
    branched[2:13, 7] = True
    branched[7, 7:12] = True
    with pytest.raises(ValueError, match="branchless"):
        skeleton_coords(branched, "NS")


def test_horizontal_centerline_is_upstream_to_downstream_and_reversible() -> None:
    mask = horizontal_channel()

    forward = centerline_from_mask(mask, "WE", 12)
    reverse = centerline_from_mask(mask, "EW", 12)

    assert isinstance(forward, CenterlineResult)
    assert forward.coordinates[0, 0] < forward.coordinates[-1, 0]
    assert reverse.coordinates[0, 0] > reverse.coordinates[-1, 0]
    assert np.median(forward.coordinates[:, 1]) == pytest.approx(39.5, abs=1.5)
    assert_array_equal(forward.mask, reverse.mask)
    assert_allclose(forward.coordinates, reverse.coordinates[::-1])
    x = np.rint(forward.coordinates[:, 0]).astype(int)
    y = np.rint(forward.coordinates[:, 1]).astype(int)
    assert mask[y, x].all()


def test_vertical_centerline_uses_image_y_direction() -> None:
    mask = vertical_channel()

    result = centerline_from_mask(mask, "NS", 12)

    assert result.coordinates[0, 1] < result.coordinates[-1, 1]
    assert np.median(result.coordinates[:, 0]) == pytest.approx(39.5, abs=1.5)


def test_curved_centerline_is_connected_and_stays_in_channel() -> None:
    mask = curved_channel()

    result = centerline_from_mask(mask, "WE", 10)

    steps = np.abs(np.diff(result.coordinates, axis=0))
    assert np.all(steps.max(axis=1) <= 1)
    assert result.coordinates[0, 0] < result.coordinates[-1, 0]
    x = np.rint(result.coordinates[:, 0]).astype(int)
    y = np.rint(result.coordinates[:, 1]).astype(int)
    assert mask[y, x].all()


def test_centerline_uses_largest_connected_channel_component() -> None:
    mask = horizontal_channel()
    mask[2:5, 2:5] = True

    result = centerline_from_mask(mask, "WE", 12)

    assert not result.mask[2:5, 2:5].any()
    assert np.ptp(result.coordinates[:, 0]) > 100


def test_horizontal_banklines_have_scientific_left_right_orientation() -> None:
    mask = horizontal_channel()

    banks = banklines_from_mask(mask, "WE")
    reversed_banks = banklines_from_mask(mask, "EW")

    assert isinstance(banks, BanklineResult)
    assert banks.left[:, 1].mean() < banks.right[:, 1].mean()
    assert banks.left[0, 0] < banks.left[-1, 0]
    # Reversing flow reverses orientation and swaps physical left/right.
    assert reversed_banks.left[:, 1].mean() > reversed_banks.right[:, 1].mean()
    assert reversed_banks.left[0, 0] > reversed_banks.left[-1, 0]
    assert_allclose(banks.left, reversed_banks.right[::-1])
    assert_allclose(banks.right, reversed_banks.left[::-1])


def test_vertical_banklines_define_left_while_looking_downstream() -> None:
    mask = vertical_channel()

    north_to_south = banklines_from_mask(mask, "NS")
    south_to_north = banklines_from_mask(mask, "SN")

    assert north_to_south.left[:, 0].mean() > north_to_south.right[:, 0].mean()
    assert north_to_south.left[0, 1] < north_to_south.left[-1, 1]
    assert south_to_north.left[:, 0].mean() < south_to_north.right[:, 0].mean()
    assert south_to_north.left[0, 1] > south_to_north.left[-1, 1]


@pytest.mark.parametrize(
    ("direction", "expected_width"),
    [("horizontal", 20.0), ("vertical", 20.0), ("diagonal", 10.0)],
)
def test_width_from_banklines_on_straight_idealized_geometry(
    direction: str, expected_width: float
) -> None:
    parameter = np.linspace(0, 100, 101)
    if direction == "horizontal":
        center = np.column_stack((parameter, np.full_like(parameter, 50)))
        left = np.column_stack((parameter, np.full_like(parameter, 40)))
        right = np.column_stack((parameter, np.full_like(parameter, 60)))
        nominal_width = 20
    elif direction == "vertical":
        center = np.column_stack((np.full_like(parameter, 50), parameter))
        left = np.column_stack((np.full_like(parameter, 60), parameter))
        right = np.column_stack((np.full_like(parameter, 40), parameter))
        nominal_width = 20
    else:
        normal = np.array([-1.0, 1.0]) / np.sqrt(2)
        center = np.column_stack((parameter, parameter))
        left = center + normal * 5
        right = center - normal * 5
        nominal_width = 10

    widths = width_from_banklines(center, left, right, nominal_width)

    assert np.isnan(widths[[0, -1]]).all()
    assert_allclose(widths[5:-5], expected_width, atol=1e-8)


def test_width_from_banklines_returns_nan_when_search_does_not_reach_banks() -> None:
    parameter = np.linspace(0, 100, 101)
    center = np.column_stack((parameter, np.zeros_like(parameter)))
    left = np.column_stack((parameter, np.full_like(parameter, -20)))
    right = np.column_stack((parameter, np.full_like(parameter, 20)))

    widths = width_from_banklines(
        center,
        left,
        right,
        2,
        search_radius_factor=3,
    )

    assert np.isnan(widths).all()


def test_width_from_mask_returns_area_over_length_profile() -> None:
    mask = np.zeros((80, 180), dtype=bool)
    mask[30:50, :] = True
    center = np.column_stack((np.arange(180, dtype=float), np.full(180, 39.5)))

    profile = width_from_mask(mask, center, 40)

    assert isinstance(profile, WidthProfile)
    assert len(profile.widths) == 5
    assert profile.breakpoints[0] == 0
    assert profile.breakpoints[-1] == pytest.approx(179)
    assert np.all(np.diff(profile.distances) > 0)
    assert_allclose(profile.widths, 20, atol=1.1)


def test_width_from_mask_accepts_explicit_distance_breakpoints() -> None:
    mask = np.zeros((80, 180), dtype=bool)
    mask[30:50, :] = True
    center = np.column_stack((np.arange(180, dtype=float), np.full(180, 39.5)))
    breakpoints = np.array([0.0, 60.0, 120.0, 179.0])

    profile = width_from_mask(mask, center, breakpoints)

    assert_array_equal(profile.breakpoints, breakpoints)
    assert_allclose(profile.distances, [30.0, 90.0, 149.5])
    assert_allclose(profile.widths, 20, atol=0.75)


def test_mask_analysis_rejects_invalid_input_contracts() -> None:
    with pytest.raises(TypeError, match="boolean"):
        crop_to_mask(np.ones((10, 10), dtype=np.uint8), "WE")
    with pytest.raises(ValueError, match="two-dimensional"):
        crop_to_mask(np.ones(10, dtype=bool), "WE")
    with pytest.raises(ValueError, match="no foreground"):
        crop_to_mask(np.zeros((10, 10), dtype=bool), "WE")
    with pytest.raises(ValueError, match="nominal_width"):
        centerline_from_mask(horizontal_channel(), "WE", 0)
    with pytest.raises(ValueError, match="foreground pixels"):
        centerline_from_mask(np.eye(10, dtype=bool), "NS", 1)

    valid_center = np.column_stack((np.arange(20, dtype=float), np.zeros(20)))
    with pytest.raises(ValueError, match="spacing"):
        width_from_mask(horizontal_channel(), valid_center, 0)
    with pytest.raises(ValueError, match="strictly increasing"):
        width_from_mask(horizontal_channel(), valid_center, [0, 10, 5])
    with pytest.raises(ValueError, match="shape"):
        width_from_banklines(valid_center, np.arange(5), valid_center, 10)
