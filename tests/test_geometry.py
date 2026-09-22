"""Focused numerical checks for RivMAPy's geometry primitives."""

from __future__ import annotations

import numpy as np
import pytest

from rivmapy.geometry import (
    angles,
    curvatures,
    polyline_intersections,
    resample_polyline,
    smooth_polyline,
)


def test_angles_for_straight_and_diagonal_lines() -> None:
    horizontal = np.array([[0, 0], [1, 0], [3, 0]], dtype=float)
    diagonal = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)

    np.testing.assert_allclose(angles(horizontal)[1:], 0.0)
    np.testing.assert_allclose(angles(diagonal)[1:], np.pi / 4)
    assert np.isnan(angles(horizontal)[0])


def test_angles_are_unwrapped_across_the_branch_cut() -> None:
    headings = np.deg2rad([179.0, -179.0, -178.0])
    points = np.vstack(
        (
            np.zeros(2),
            np.cumsum(np.column_stack((np.cos(headings), np.sin(headings))), axis=0),
        )
    )

    result = angles(points)

    assert np.isnan(result[0])
    assert np.all(np.abs(np.diff(result[1:])) < np.deg2rad(3.0))


def test_curvature_is_zero_for_a_straight_line() -> None:
    points = np.column_stack((np.arange(7.0), np.arange(7.0)))

    result = curvatures(points)

    assert np.all(np.isnan(result[:2]))
    np.testing.assert_allclose(result[2:], 0.0, atol=1e-14)


def test_curvature_of_a_circular_arc_has_image_coordinate_sign() -> None:
    radius = 10.0
    parameter = np.linspace(0.0, np.pi / 2, 41)
    arc = radius * np.column_stack((np.cos(parameter), np.sin(parameter)))

    result = curvatures(arc)

    assert np.all(np.isnan(result[:2]))
    np.testing.assert_allclose(result[2:-1], -1.0 / radius, rtol=2e-3)


def test_smoothing_shrinks_at_ends_and_preserves_endpoints() -> None:
    points = np.column_stack(
        (
            np.arange(7.0),
            np.array([0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0]),
        )
    )

    result = smooth_polyline(points, window=5)

    np.testing.assert_array_equal(result[[0, -1]], points[[0, -1]])
    np.testing.assert_allclose(result[1], np.mean(points[:3], axis=0))
    np.testing.assert_allclose(result[3], np.mean(points[1:6], axis=0))
    np.testing.assert_allclose(result[-2], np.mean(points[-3:], axis=0))


def test_nonodd_smoothing_preserves_a_uniform_straight_line() -> None:
    points = np.column_stack((np.arange(8.0), 2.0 * np.arange(8.0) + 1.0))

    result = smooth_polyline(points, window=4.5)

    np.testing.assert_allclose(result, points)
    assert not np.shares_memory(result, points)


def test_resampling_by_count_follows_arc_length_around_a_corner() -> None:
    points = np.array([[0, 0], [2, 0], [2, 2]], dtype=float)

    result = resample_polyline(points, count=5)

    expected = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]], dtype=float)
    np.testing.assert_allclose(result, expected)


def test_resampling_by_spacing_includes_the_exact_endpoint() -> None:
    points = np.array([[0, 0], [2, 0], [2, 2]], dtype=float)

    result = resample_polyline(points, spacing=1.5)

    expected = np.array([[0, 0], [1.5, 0], [2, 1], [2, 2]], dtype=float)
    np.testing.assert_allclose(result, expected)


def test_resampling_at_distances_preserves_requested_order() -> None:
    points = np.array([[0, 0], [2, 0], [2, 2]], dtype=float)

    result = resample_polyline(points, distances=[4.0, 0.5, 2.0])

    expected = np.array([[2, 2], [0.5, 0], [2, 0]], dtype=float)
    np.testing.assert_allclose(result, expected)


def test_single_segment_intersection_returns_fractional_positions() -> None:
    first = np.array([[0, 0], [2, 2]], dtype=float)
    second = np.array([[0, 2], [2, 0]], dtype=float)

    points, first_positions, second_positions = polyline_intersections(first, second)

    np.testing.assert_allclose(points, [[1.0, 1.0]])
    np.testing.assert_allclose(first_positions, [0.5])
    np.testing.assert_allclose(second_positions, [0.5])


def test_intersections_are_sorted_along_the_first_polyline() -> None:
    first = np.array([[0, 0], [4, 0]], dtype=float)
    second = np.array([[3, -1], [3, 1], [1, 1], [1, -1]], dtype=float)

    points, first_positions, second_positions = polyline_intersections(first, second)

    np.testing.assert_allclose(points, [[1, 0], [3, 0]])
    np.testing.assert_allclose(first_positions, [0.25, 0.75])
    np.testing.assert_allclose(second_positions, [2.5, 0.5])


def test_shared_vertex_intersection_is_deduplicated() -> None:
    first = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    second = np.array([[1, -1], [1, 1]], dtype=float)

    points, first_positions, second_positions = polyline_intersections(first, second)

    np.testing.assert_allclose(points, [[1, 0]])
    np.testing.assert_allclose(first_positions, [1.0])
    np.testing.assert_allclose(second_positions, [0.5])


def test_collinear_overlap_is_represented_by_boundary_points() -> None:
    first = np.array([[0, 0], [3, 0]], dtype=float)
    second = np.array([[1, 0], [2, 0]], dtype=float)

    points, first_positions, second_positions = polyline_intersections(first, second)

    np.testing.assert_allclose(points, [[1, 0], [2, 0]])
    np.testing.assert_allclose(first_positions, [1 / 3, 2 / 3])
    np.testing.assert_allclose(second_positions, [0, 1])


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (angles, (np.zeros((3, 3)),)),
        (angles, (np.array([[0, 0], [np.nan, 1]]),)),
        (curvatures, (np.array([[0, 0], [1, 0]]),)),
        (smooth_polyline, (np.array([[0, 0], [1, 0]]), 1)),
        (resample_polyline, (np.array([[0, 0], [0, 0]]),)),
    ],
)
def test_invalid_geometry_inputs_raise(
    function: object, arguments: tuple[object, ...]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        function(*arguments)  # type: ignore[operator]


def test_resampling_requires_one_valid_sampling_mode() -> None:
    points = np.array([[0, 0], [2, 0]], dtype=float)

    with pytest.raises(ValueError, match="exactly one"):
        resample_polyline(points)
    with pytest.raises(ValueError, match="exactly one"):
        resample_polyline(points, count=3, spacing=1)
    with pytest.raises(ValueError, match="within"):
        resample_polyline(points, distances=[2.1])


def test_intersections_validate_inputs_and_empty_result_shapes() -> None:
    first = np.array([[0, 0], [1, 0]], dtype=float)
    second = np.array([[0, 1], [1, 1]], dtype=float)

    points, first_positions, second_positions = polyline_intersections(first, second)

    assert points.shape == (0, 2)
    assert first_positions.shape == second_positions.shape == (0,)
    with pytest.raises(ValueError, match="duplicate"):
        polyline_intersections([[0, 0], [0, 0]], second)
    with pytest.raises(ValueError, match="nonnegative"):
        polyline_intersections(first, second, atol=-1)
