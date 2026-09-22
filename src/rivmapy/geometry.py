"""Numerical geometry primitives used throughout RivMAPy.

Coordinates are always represented by floating-point arrays with shape
``(n, 2)`` and columns ``(x, y)``.  In particular, this module does not swap
the axes to NumPy's usual ``(row, column)`` image indexing convention.

The angle and curvature definitions intentionally retain RivMAP's original
image-coordinate sign convention: because image ``y`` increases downward, a
counter-clockwise arc in Cartesian coordinates has negative curvature.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_coordinates

__all__ = [
    "angles",
    "curvatures",
    "smooth_polyline",
    "resample_polyline",
    "polyline_intersections",
]


FloatArray = NDArray[np.float64]


def _as_polyline(
    coordinates: ArrayLike,
    *,
    name: str = "coordinates",
    minimum_length: int = 2,
) -> FloatArray:
    """Validate a polyline and reject its undefined zero-length segments."""

    points = as_coordinates(
        coordinates, name=name, minimum_length=minimum_length
    )
    if np.any(np.all(np.diff(points, axis=0) == 0.0, axis=1)):
        raise ValueError(f"{name} must not contain consecutive duplicate points")
    return points


def _cumulative_distances(points: FloatArray) -> FloatArray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate((np.array([0.0]), np.cumsum(lengths)))


def angles(coordinates: ArrayLike) -> FloatArray:
    """Return the unwrapped heading at each polyline vertex, in radians.

    A segment's heading is assigned to its downstream vertex, matching the
    original MATLAB ``angles.m`` routine.  Consequently, the first value is
    ``NaN``.  Heading is measured by ``atan2(dy, dx)`` and is unwrapped to
    avoid artificial jumps at ``-pi``/``pi``.
    """

    points = _as_polyline(coordinates)
    differences = np.diff(points, axis=0)
    headings = np.unwrap(np.arctan2(differences[:, 1], differences[:, 0]))
    return np.concatenate((np.array([np.nan]), headings))


def curvatures(coordinates: ArrayLike) -> FloatArray:
    """Return signed curvature along a polyline in inverse coordinate units.

    This is the nonuniform central-difference expression used by RivMAP's
    ``curvatures.m``.  The first two values are ``NaN`` because no upstream
    heading exists there; the final value uses the legacy one-sided stencil.
    The leading minus sign is retained for image-coordinate compatibility.
    """

    points = _as_polyline(coordinates, minimum_length=3)
    distance = _cumulative_distances(points)
    heading = angles(points)

    distance_down = np.concatenate((distance[2:3], distance[:-1]))
    distance_up = np.concatenate((distance[1:], distance[-3:-2]))
    heading_down = np.concatenate((heading[2:3], heading[:-1]))
    heading_up = np.concatenate((heading[1:], heading[-3:-2]))

    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = -(
            (heading_up - heading)
            / (distance_up - distance)
            * (distance - distance_down)
            + (heading - heading_down)
            / (distance - distance_down)
            * (distance_up - distance)
        ) / (distance_up - distance_down)

    # State this scientifically meaningful boundary condition explicitly;
    # it also protects the public contract from floating-point implementation
    # details in the expression above.
    curvature[:2] = np.nan
    return np.asarray(curvature, dtype=float)


def _odd_window_average(points: FloatArray, window: int) -> FloatArray:
    """Apply RivMAP's odd-width moving average, shrinking at both ends."""

    point_count = len(points)
    largest_odd_window = point_count if point_count % 2 else point_count - 1
    effective_window = min(window, largest_odd_window)
    if effective_window < 3:
        return points.copy()

    half_width = (effective_window - 1) // 2
    cumulative = np.vstack((np.zeros((1, 2)), np.cumsum(points, axis=0)))
    result = np.empty_like(points)

    for index in range(point_count):
        local_half_width = min(half_width, index, point_count - 1 - index)
        start = index - local_half_width
        stop = index + local_half_width + 1
        result[index] = (cumulative[stop] - cumulative[start]) / (stop - start)

    return result


def smooth_polyline(coordinates: ArrayLike, window: float) -> FloatArray:
    """Smooth both coordinate columns with RivMAP's averaging filter.

    Odd integer windows are centered moving averages.  Near each endpoint,
    the window shrinks to the largest centered odd window that fits, leaving
    both endpoints unchanged.  A nonodd window blends the adjacent lower and
    upper odd-window results using the efficiency-preserving weighting from
    ``savfilt.m``.  Windows wider than the input are safely capped.
    """

    points = _as_polyline(coordinates)
    if isinstance(window, (bool, np.bool_)):
        raise TypeError("window must be a real scalar greater than one")
    try:
        numeric_window = float(window)
    except (TypeError, ValueError) as error:
        raise TypeError("window must be a real scalar greater than one") from error
    if not np.isfinite(numeric_window) or numeric_window <= 1.0:
        raise ValueError("window must be finite and greater than one")

    if numeric_window.is_integer() and int(numeric_window) % 2 == 1:
        return _odd_window_average(points, int(numeric_window))

    lower_window = math.floor(numeric_window)
    if lower_window % 2 == 0:
        lower_window -= 1
    upper_window = lower_window + 2

    lower_result = _odd_window_average(points, lower_window)
    upper_result = _odd_window_average(points, upper_window)
    if np.array_equal(lower_result, upper_result):
        return lower_result

    weight = math.sqrt(
        (upper_window / numeric_window - 1.0)
        / (upper_window / lower_window - 1.0)
    )
    weight = min(1.0, max(0.0, weight))
    return lower_result * weight + upper_result * (1.0 - weight)


def resample_polyline(
    coordinates: ArrayLike,
    *,
    count: int | None = None,
    spacing: float | None = None,
    distances: ArrayLike | None = None,
) -> FloatArray:
    """Linearly resample a polyline by cumulative chord length.

    Exactly one sampling mode must be supplied:

    ``count``
        Produce this many evenly spaced points, including both endpoints.
    ``spacing``
        Use this distance between samples and append the exact endpoint when
        the total length is not an integer multiple of the spacing.
    ``distances``
        Evaluate at arbitrary absolute along-line distances.  Their input
        order is preserved and every value must lie in ``[0, total_length]``.
    """

    points = _as_polyline(coordinates)
    supplied_modes = sum(value is not None for value in (count, spacing, distances))
    if supplied_modes != 1:
        raise ValueError("supply exactly one of count, spacing, or distances")

    cumulative = _cumulative_distances(points)
    total_length = float(cumulative[-1])

    if count is not None:
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ):
            raise TypeError("count must be an integer")
        if count < 2:
            raise ValueError("count must be at least two")
        targets = np.linspace(0.0, total_length, int(count))
    elif spacing is not None:
        if isinstance(spacing, (bool, np.bool_)):
            raise TypeError("spacing must be a real scalar")
        try:
            numeric_spacing = float(spacing)
        except (TypeError, ValueError) as error:
            raise TypeError("spacing must be a real scalar") from error
        if not np.isfinite(numeric_spacing) or numeric_spacing <= 0.0:
            raise ValueError("spacing must be finite and greater than zero")
        targets = np.arange(0.0, total_length, numeric_spacing, dtype=float)
        if len(targets) == 0 or not np.isclose(
            targets[-1], total_length, rtol=1e-12, atol=1e-12
        ):
            targets = np.append(targets, total_length)
        else:
            targets[-1] = total_length
    else:
        try:
            targets = np.asarray(distances, dtype=float)
        except (TypeError, ValueError) as error:
            raise TypeError("distances must be a one-dimensional numeric array") from error
        if targets.ndim != 1 or targets.size == 0:
            raise ValueError("distances must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(targets)):
            raise ValueError("distances must contain only finite values")
        tolerance = 1e-12 * max(1.0, total_length)
        if np.any(targets < -tolerance) or np.any(targets > total_length + tolerance):
            raise ValueError("distances must lie within the polyline length")
        targets = np.clip(targets, 0.0, total_length)

    result = np.column_stack(
        (
            np.interp(targets, cumulative, points[:, 0]),
            np.interp(targets, cumulative, points[:, 1]),
        )
    )
    return np.asarray(result, dtype=float)


def _cross_2d(first: FloatArray, second: FloatArray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _segment_intersections(
    first_start: FloatArray,
    first_end: FloatArray,
    second_start: FloatArray,
    second_end: FloatArray,
    *,
    atol: float,
) -> list[tuple[FloatArray, float, float]]:
    """Return point and local parameters for two finite line segments."""

    first_vector = first_end - first_start
    second_vector = second_end - second_start
    offset = second_start - first_start
    denominator = _cross_2d(first_vector, second_vector)
    parallel_scale = max(
        1.0,
        float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector)),
    )

    if abs(denominator) > atol * parallel_scale:
        first_parameter = _cross_2d(offset, second_vector) / denominator
        second_parameter = _cross_2d(offset, first_vector) / denominator
        if (
            -atol <= first_parameter <= 1.0 + atol
            and -atol <= second_parameter <= 1.0 + atol
        ):
            first_parameter = min(1.0, max(0.0, first_parameter))
            second_parameter = min(1.0, max(0.0, second_parameter))
            point = first_start + first_parameter * first_vector
            return [(point, first_parameter, second_parameter)]
        return []

    collinear_scale = max(
        1.0, float(np.linalg.norm(offset) * np.linalg.norm(first_vector))
    )
    if abs(_cross_2d(offset, first_vector)) > atol * collinear_scale:
        return []

    first_squared_length = float(np.dot(first_vector, first_vector))
    second_squared_length = float(np.dot(second_vector, second_vector))
    second_start_on_first = float(
        np.dot(second_start - first_start, first_vector) / first_squared_length
    )
    second_end_on_first = float(
        np.dot(second_end - first_start, first_vector) / first_squared_length
    )
    overlap_start = max(0.0, min(second_start_on_first, second_end_on_first))
    overlap_end = min(1.0, max(second_start_on_first, second_end_on_first))

    if overlap_end < overlap_start - atol:
        return []

    if abs(overlap_end - overlap_start) <= atol:
        first_parameters = [(overlap_start + overlap_end) / 2.0]
    else:
        # A finite representation of a continuum: report both overlap ends.
        first_parameters = [overlap_start, overlap_end]

    intersections: list[tuple[FloatArray, float, float]] = []
    for first_parameter in first_parameters:
        first_parameter = min(1.0, max(0.0, first_parameter))
        point = first_start + first_parameter * first_vector
        second_parameter = float(
            np.dot(point - second_start, second_vector) / second_squared_length
        )
        second_parameter = min(1.0, max(0.0, second_parameter))
        intersections.append((point, first_parameter, second_parameter))
    return intersections


def polyline_intersections(
    first: ArrayLike,
    second: ArrayLike,
    *,
    deduplicate: bool = True,
    atol: float = 1e-9,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ordered intersections and positions along two polylines.

    The return value is ``(points, first_positions, second_positions)``.
    Positions are zero-based fractional segment indices: ``2.25`` denotes a
    point one quarter of the way from vertex 2 to vertex 3.  Results are
    ordered first by position on ``first`` and then by position on ``second``.

    Shared vertices are deduplicated by default.  Distinct visits to the same
    coordinate remain distinct when their along-line positions differ.
    Collinear overlap is represented by its one or two boundary points rather
    than by an indeterminate continuum.
    """

    first_points = _as_polyline(first, name="first")
    second_points = _as_polyline(second, name="second")
    if not isinstance(deduplicate, (bool, np.bool_)):
        raise TypeError("deduplicate must be Boolean")
    if isinstance(atol, (bool, np.bool_)):
        raise TypeError("atol must be a real scalar")
    try:
        numeric_atol = float(atol)
    except (TypeError, ValueError) as error:
        raise TypeError("atol must be a real scalar") from error
    if not np.isfinite(numeric_atol) or numeric_atol < 0.0:
        raise ValueError("atol must be finite and nonnegative")

    records: list[tuple[FloatArray, float, float]] = []
    for first_index, (first_start, first_end) in enumerate(
        zip(first_points[:-1], first_points[1:], strict=True)
    ):
        first_minimum = np.minimum(first_start, first_end)
        first_maximum = np.maximum(first_start, first_end)
        for second_index, (second_start, second_end) in enumerate(
            zip(second_points[:-1], second_points[1:], strict=True)
        ):
            second_minimum = np.minimum(second_start, second_end)
            second_maximum = np.maximum(second_start, second_end)
            if np.any(first_maximum < second_minimum - numeric_atol) or np.any(
                second_maximum < first_minimum - numeric_atol
            ):
                continue

            for point, first_local, second_local in _segment_intersections(
                first_start,
                first_end,
                second_start,
                second_end,
                atol=numeric_atol,
            ):
                records.append(
                    (
                        np.asarray(point, dtype=float),
                        first_index + first_local,
                        second_index + second_local,
                    )
                )

    if not records:
        return (
            np.empty((0, 2), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    records.sort(key=lambda record: (record[1], record[2]))
    if deduplicate:
        unique_records: list[tuple[FloatArray, float, float]] = []
        position_tolerance = max(1e-12, numeric_atol)
        for record in records:
            if unique_records:
                prior = unique_records[-1]
                same_point = np.allclose(
                    record[0], prior[0], rtol=0.0, atol=numeric_atol
                )
                same_positions = (
                    abs(record[1] - prior[1]) <= position_tolerance
                    and abs(record[2] - prior[2]) <= position_tolerance
                )
                if same_point and same_positions:
                    continue
            unique_records.append(record)
        records = unique_records

    points = np.vstack([record[0] for record in records]).astype(float, copy=False)
    first_positions = np.asarray([record[1] for record in records], dtype=float)
    second_positions = np.asarray([record[2] for record in records], dtype=float)
    return points, first_positions, second_positions
