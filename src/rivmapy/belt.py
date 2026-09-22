"""Automatic channel-belt construction from a time series of channel masks.

This module implements the automatic, geometric half of RivMAP's MATLAB
``spatial_changes.m``.  The scientific intent is preserved -- an
oversmoothed envelope around every observed channel position, divided into
roughly equally spaced cross-belt cells -- while replacing the original
unbounded dilation and inflection-point heuristics with bounded, testable
operations.

All vector coordinates are zero-based image ``(x, y)`` pixel-centre
coordinates.  Raster arrays retain NumPy's ``(row, column) == (y, x)``
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from skimage.draw import polygon as draw_polygon

from ._validation import validate_exit_sides
from .geometry import resample_polyline, smooth_polyline
from .mask import _validate_spatial_geometry, banklines_from_mask, centerline_from_mask
from .spatial import GeoRaster, RasterGrid, SpatialResultMixin


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ChannelBelt(SpatialResultMixin):
    """A flow-oriented channel-belt envelope and its spatial cells.

    ``left``, ``right``, and ``centerline`` contain corresponding station
    points oriented upstream to downstream.  ``midpoint_distances`` contains
    the along-belt midpoint of every cell.  ``cell_masks`` has shape
    ``(n_cells, rows, columns)`` and is an exact, disjoint partition of the
    Boolean ``envelope`` mask.
    """

    left: FloatArray
    right: FloatArray
    centerline: FloatArray
    midpoint_distances: FloatArray
    cell_masks: BoolArray
    envelope: BoolArray
    grid: RasterGrid | None = None


def _positive_scalar(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _nonnegative_scalar(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_masks(
    channel_masks: Sequence[ArrayLike] | NDArray[np.generic],
    exit_sides: str,
) -> tuple[list[BoolArray], tuple[int, int]]:
    if isinstance(channel_masks, np.ndarray) and channel_masks.ndim == 2:
        raise ValueError("channel_masks must contain a time series of 2-D masks")
    try:
        raw_masks = list(channel_masks)
    except TypeError as error:
        raise TypeError("channel_masks must be a nonempty sequence") from error
    if not raw_masks:
        raise ValueError("channel_masks must contain at least one mask")

    masks: list[BoolArray] = []
    shape: tuple[int, int] | None = None
    for index, raw_mask in enumerate(raw_masks):
        array = np.asarray(raw_mask)
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"channel_masks[{index}] must be a nonempty 2-D array")
        if np.issubdtype(array.dtype, np.bool_):
            mask = np.array(array, dtype=bool, copy=True)
        elif np.issubdtype(array.dtype, np.number):
            if not np.all(np.isfinite(array)) or not np.all(
                (array == 0) | (array == 1)
            ):
                raise ValueError(
                    f"channel_masks[{index}] must contain only zero and one"
                )
            mask = np.array(array, dtype=bool, copy=True)
        else:
            raise TypeError(
                f"channel_masks[{index}] must contain Boolean or binary numeric values"
            )
        if not np.any(mask):
            raise ValueError(f"channel_masks[{index}] must contain channel pixels")
        if shape is None:
            shape = mask.shape
            if min(shape) < 3:
                raise ValueError("channel masks must be at least 3 by 3 pixels")
        elif mask.shape != shape:
            raise ValueError("all channel masks must have the same shape")

        for side in exit_sides:
            if side == "N":
                touches = np.any(mask[0, :])
            elif side == "S":
                touches = np.any(mask[-1, :])
            elif side == "W":
                touches = np.any(mask[:, 0])
            else:
                touches = np.any(mask[:, -1])
            if not touches:
                raise ValueError(
                    f"channel_masks[{index}] does not touch declared {side} exit"
                )
        masks.append(mask)

    assert shape is not None
    return masks, shape


def _remove_consecutive_duplicates(points: FloatArray) -> FloatArray:
    if len(points) < 2:
        return points.copy()
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]
    return points[keep].copy()


def _smooth_raster_line(points: FloatArray, window: float) -> FloatArray:
    """Resample near one-pixel density before applying a physical window."""

    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    dense = resample_polyline(points, count=max(2, int(math.ceil(length)) + 1))
    if window > 1.0 and len(dense) > 2:
        dense = smooth_polyline(dense, window)
    result = _remove_consecutive_duplicates(dense)
    if len(result) < 2:
        raise RuntimeError("smoothing collapsed a belt line")
    return result


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        parts: list[Polygon] = []
        for child in geometry.geoms:
            parts.extend(_polygon_parts(child))
        return parts
    return []


def _rasterize_geometry(
    geometry: BaseGeometry, shape: tuple[int, int]
) -> BoolArray:
    result = np.zeros(shape, dtype=bool)
    for polygon in _polygon_parts(geometry):
        exterior = np.asarray(polygon.exterior.coords, dtype=float)
        rows, columns = draw_polygon(exterior[:, 1], exterior[:, 0], shape=shape)
        result[rows, columns] = True
        for ring in polygon.interiors:
            interior = np.asarray(ring.coords, dtype=float)
            rows, columns = draw_polygon(
                interior[:, 1], interior[:, 0], shape=shape
            )
            result[rows, columns] = False
    return result


def _build_envelope(
    union: BoolArray,
    exit_sides: str,
    nominal_width: float,
    *,
    dilation_factor: float,
    smoothing_factor: float,
    dilation_increment_factor: float,
    padding_factor: float,
    max_iterations: int,
    max_padded_pixels: int,
) -> tuple[Polygon, BoolArray]:
    height, width = union.shape
    maximum_radius = nominal_width * (
        dilation_factor + (max_iterations - 1) * dilation_increment_factor
    )
    padding = max(
        2,
        int(math.ceil(nominal_width * padding_factor)),
        int(math.ceil(maximum_radius + 2.0)),
    )
    padded_shape = (height + 2 * padding, width + 2 * padding)
    if math.prod(padded_shape) > max_padded_pixels:
        raise ValueError(
            "requested belt padding exceeds max_padded_pixels; reduce the "
            "empirical factors or increase the explicit memory guard"
        )

    padded_union = np.pad(union, padding, mode="constant", constant_values=False)
    distance_from_union = ndi.distance_transform_edt(~padded_union)
    frame = box(
        float(padding),
        float(padding),
        float(padding + width - 1),
        float(padding + height - 1),
    )
    smoothing_window = nominal_width * smoothing_factor
    last_reason = "no candidate was produced"

    for iteration in range(max_iterations):
        radius = nominal_width * (
            dilation_factor + iteration * dilation_increment_factor
        )
        buffered = distance_from_union <= radius
        try:
            banks = banklines_from_mask(
                buffered,
                exit_sides,
                min_mask_pixels=1,
                min_bank_component_pixels=2,
            )
            left = _smooth_raster_line(banks.left, smoothing_window)
            right = _smooth_raster_line(banks.right, smoothing_window)
        except (ValueError, RuntimeError) as error:
            last_reason = str(error)
            continue

        candidate = Polygon(np.vstack((left, right[::-1])))
        if candidate.is_empty or candidate.area <= 0.0 or not candidate.is_valid:
            last_reason = "smoothed banklines produced an invalid polygon"
            continue
        clipped = candidate.intersection(frame)
        parts = [part for part in _polygon_parts(clipped) if part.area > 0.0]
        if len(parts) != 1 or not parts[0].is_valid:
            last_reason = "the clipped belt was not one simple polygon"
            continue

        local_polygon = translate(parts[0], xoff=-padding, yoff=-padding)
        envelope = _rasterize_geometry(local_polygon, union.shape)
        uncovered = union & ~envelope
        if np.any(uncovered):
            last_reason = (
                f"candidate omitted {int(np.count_nonzero(uncovered))} union pixels"
            )
            continue
        return local_polygon, envelope

    raise RuntimeError(
        "channel-belt envelope did not converge within "
        f"{max_iterations} iterations: {last_reason}"
    )


def _point_parts(geometry: BaseGeometry) -> list[FloatArray]:
    if isinstance(geometry, Point):
        return [np.asarray(geometry.coords[0], dtype=float)]
    if isinstance(geometry, MultiPoint):
        return [np.asarray(point.coords[0], dtype=float) for point in geometry.geoms]
    if isinstance(geometry, LineString):
        coordinates = np.asarray(geometry.coords, dtype=float)
        if len(coordinates) == 0:
            return []
        return [coordinates[0], coordinates[-1]]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        points: list[FloatArray] = []
        for child in geometry.geoms:
            points.extend(_point_parts(child))
        return points
    return []


def _line_parts(geometry: BaseGeometry) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry] if geometry.length > 0.0 else []
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        parts: list[LineString] = []
        for child in geometry.geoms:
            parts.extend(_line_parts(child))
        return parts
    return []


def _extend_axis_to_boundary(
    axis: FloatArray, polygon: Polygon, shape: tuple[int, int]
) -> FloatArray:
    result = axis.copy()
    ray_length = 3.0 * math.hypot(*shape)
    for endpoint, neighbor, direction in ((0, 1, -1.0), (-1, -2, 1.0)):
        tangent = result[endpoint] - result[neighbor]
        if endpoint == 0:
            tangent = -tangent
        norm = float(np.linalg.norm(tangent))
        if norm <= 1e-12:
            continue
        tangent /= norm
        start = result[endpoint]
        ray_end = start + direction * ray_length * tangent
        intersections = _point_parts(
            polygon.boundary.intersection(LineString((start, ray_end)))
        )
        progress = [float(np.dot(point - start, tangent)) for point in intersections]
        if endpoint == 0:
            candidates = [
                (abs(value), point)
                for value, point in zip(progress, intersections, strict=True)
                if value < -1e-8
            ]
        else:
            candidates = [
                (abs(value), point)
                for value, point in zip(progress, intersections, strict=True)
                if value > 1e-8
            ]
        if candidates:
            result[endpoint] = min(candidates, key=lambda item: item[0])[1]
    return _remove_consecutive_duplicates(result)


def _cross_section(
    polygon: Polygon,
    center: FloatArray,
    tangent: FloatArray,
    *,
    ray_length: float,
) -> tuple[FloatArray, FloatArray]:
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise RuntimeError("a belt station has an undefined tangent")
    unit_tangent = tangent / tangent_norm
    # With image y increasing down, this is left while looking downstream.
    left_normal = np.array([unit_tangent[1], -unit_tangent[0]])
    crossline = LineString(
        (center - ray_length * left_normal, center + ray_length * left_normal)
    )
    sections = _line_parts(polygon.intersection(crossline))
    if not sections:
        raise RuntimeError("a belt normal did not intersect the envelope")

    center_point = Point(float(center[0]), float(center[1]))
    section = min(sections, key=lambda part: (part.distance(center_point), -part.length))
    coordinates = np.asarray(section.coords, dtype=float)
    projections = (coordinates - center) @ left_normal
    left_index = int(np.argmax(projections))
    right_index = int(np.argmin(projections))
    if projections[left_index] <= 1e-8 or projections[right_index] >= -1e-8:
        raise RuntimeError("a belt normal did not cross both sides of the envelope")
    return coordinates[left_index].copy(), coordinates[right_index].copy()


def _exit_gate_section(
    polygon: Polygon,
    side: str,
    center: FloatArray,
    tangent: FloatArray,
    shape: tuple[int, int],
) -> tuple[FloatArray, FloatArray]:
    """Return left/right endpoints where the belt meets an image exit."""

    height, width = shape
    margin = 2.0 * math.hypot(height, width)
    if side == "W":
        gate = LineString(((0.0, -margin), (0.0, height - 1 + margin)))
    elif side == "E":
        gate = LineString(
            ((float(width - 1), -margin), (float(width - 1), height - 1 + margin))
        )
    elif side == "N":
        gate = LineString(((-margin, 0.0), (width - 1 + margin, 0.0)))
    else:
        gate = LineString(
            ((-margin, float(height - 1)), (width - 1 + margin, float(height - 1)))
        )
    sections = _line_parts(polygon.intersection(gate))
    if not sections:
        raise RuntimeError(f"belt envelope does not intersect declared {side} exit")
    center_point = Point(float(center[0]), float(center[1]))
    section = min(sections, key=lambda part: (part.distance(center_point), -part.length))
    coordinates = np.asarray(section.coords, dtype=float)
    if len(coordinates) < 2:
        raise RuntimeError(f"belt has zero width at declared {side} exit")

    unit_tangent = tangent / np.linalg.norm(tangent)
    left_normal = np.array([unit_tangent[1], -unit_tangent[0]])
    midpoint = (coordinates[0] + coordinates[-1]) / 2.0
    projections = (coordinates - midpoint) @ left_normal
    left_index = int(np.argmax(projections))
    right_index = int(np.argmin(projections))
    if projections[left_index] <= 1e-8 or projections[right_index] >= -1e-8:
        raise RuntimeError(f"could not orient the belt at declared {side} exit")
    return coordinates[left_index].copy(), coordinates[right_index].copy()


def _station_edges(
    polygon: Polygon,
    envelope: BoolArray,
    exit_sides: str,
    nominal_width: float,
    spacing: float,
    smoothing_factor: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    raw_axis = centerline_from_mask(
        envelope,
        exit_sides,
        nominal_width,
        min_mask_pixels=1,
        min_centerline_length_factor=0.0,
    ).coordinates
    smoothing_window = nominal_width * smoothing_factor
    axis = _smooth_raster_line(raw_axis, smoothing_window)
    axis_geometry = LineString(axis)
    # Averaging may cut across a very concave envelope.  Retain the unsmoothed
    # skeleton axis in that case rather than allowing an invalid normal origin.
    if not polygon.buffer(0.75).covers(axis_geometry):
        axis = _remove_consecutive_duplicates(raw_axis)
    axis = _extend_axis_to_boundary(axis, polygon, envelope.shape)

    axis_length = float(np.linalg.norm(np.diff(axis, axis=0), axis=1).sum())
    if axis_length <= 0.0:
        raise RuntimeError("the channel-belt centerline has zero length")
    cell_count = max(1, int(round(axis_length / spacing)))
    target_distances = np.linspace(0.0, axis_length, cell_count + 1)
    stations = resample_polyline(axis, distances=target_distances)
    tangents = np.empty_like(stations)
    tangents[0] = stations[1] - stations[0]
    tangents[-1] = stations[-1] - stations[-2]
    if len(stations) > 2:
        tangents[1:-1] = stations[2:] - stations[:-2]

    ray_length = 3.0 * math.hypot(*envelope.shape)
    left = np.empty_like(stations)
    right = np.empty_like(stations)
    for index, (station, tangent) in enumerate(
        zip(stations, tangents, strict=True)
    ):
        if index == 0:
            left[index], right[index] = _exit_gate_section(
                polygon, exit_sides[0], station, tangent, envelope.shape
            )
        elif index == len(stations) - 1:
            left[index], right[index] = _exit_gate_section(
                polygon, exit_sides[1], station, tangent, envelope.shape
            )
        else:
            left[index], right[index] = _cross_section(
                polygon, station, tangent, ray_length=ray_length
            )
        cross = float(
            tangent[0] * (right[index, 1] - left[index, 1])
            - tangent[1] * (right[index, 0] - left[index, 0])
        )
        if cross <= 0.0:
            raise RuntimeError("left/right belt orientation became inconsistent")

    centerline = (left + right) / 2.0
    centerline_distances = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))
    ]
    midpoint_distances = (
        centerline_distances[:-1] + centerline_distances[1:]
    ) / 2.0
    return left, right, centerline, midpoint_distances, centerline_distances


def _proportional_station_edges(
    envelope: BoolArray,
    exit_sides: str,
    spacing: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Pair monotone positions along both banks when normals are ambiguous.

    Wide concave belts, especially reaches with adjacent image exits, can
    intersect a centerline normal more than once.  Equal normalized progress
    along the two independently flow-oriented banks is less locally
    orthogonal, but it preserves station order and cannot jump between remote
    boundary branches.  The resulting quadrilaterals still pass the same
    strict topology checks as normal-paired cells.
    """

    banks = banklines_from_mask(
        envelope,
        exit_sides,
        min_mask_pixels=1,
        min_bank_component_pixels=2,
    )
    bank_lengths = [
        float(np.linalg.norm(np.diff(bank, axis=0), axis=1).sum())
        for bank in (banks.left, banks.right)
    ]
    dense_count = max(2, int(math.ceil(max(bank_lengths))) + 1)
    dense_left = resample_polyline(banks.left, count=dense_count)
    dense_right = resample_polyline(banks.right, count=dense_count)
    dense_centerline = (dense_left + dense_right) / 2.0
    dense_distances = np.r_[
        0.0,
        np.cumsum(np.linalg.norm(np.diff(dense_centerline, axis=0), axis=1)),
    ]
    keep = np.r_[True, np.diff(dense_distances) > 1e-10]
    dense_left = dense_left[keep]
    dense_right = dense_right[keep]
    dense_distances = dense_distances[keep]
    if len(dense_distances) < 2 or dense_distances[-1] <= 0.0:
        raise RuntimeError("proportional bank pairing produced a zero-length axis")

    cell_count = max(1, int(round(float(dense_distances[-1]) / spacing)))
    targets = np.linspace(0.0, float(dense_distances[-1]), cell_count + 1)

    def interpolate_bank(bank: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                np.interp(targets, dense_distances, bank[:, 0]),
                np.interp(targets, dense_distances, bank[:, 1]),
            )
        )

    left = interpolate_bank(dense_left)
    right = interpolate_bank(dense_right)
    centerline = (left + right) / 2.0
    centerline_distances = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))
    ]
    midpoint_distances = (
        centerline_distances[:-1] + centerline_distances[1:]
    ) / 2.0
    _validated_quadrilaterals(left, right)
    return left, right, centerline, midpoint_distances, centerline_distances


def _validated_quadrilaterals(
    left: FloatArray, right: FloatArray
) -> list[Polygon]:
    quadrilaterals: list[Polygon] = []
    for index in range(len(left) - 1):
        coordinates = np.vstack(
            (left[index], right[index], right[index + 1], left[index + 1])
        )
        polygon = Polygon(coordinates)
        if polygon.is_empty or polygon.area <= 1e-8 or not polygon.is_valid:
            raise RuntimeError(
                f"belt cross sections produced an invalid cell at index {index}"
            )
        quadrilaterals.append(polygon)

    tree = STRtree(quadrilaterals)
    for first_index, first in enumerate(quadrilaterals):
        for second_index_raw in tree.query(first, predicate="intersects"):
            second_index = int(second_index_raw)
            if second_index <= first_index:
                continue
            if first.intersection(quadrilaterals[second_index]).area > 1e-7:
                raise RuntimeError(
                    "belt quadrilaterals overlap; automatic normal pairing is ambiguous"
                )
    return quadrilaterals


def _partition_envelope(
    polygon: Polygon,
    envelope: BoolArray,
    left: FloatArray,
    right: FloatArray,
    guide_axis: FloatArray,
    target_distances: FloatArray,
) -> BoolArray:
    quadrilaterals = _validated_quadrilaterals(left, right)
    cells = np.zeros((len(quadrilaterals), *envelope.shape), dtype=bool)
    assigned = np.zeros_like(envelope)
    for index, quadrilateral in enumerate(quadrilaterals):
        mask = _rasterize_geometry(quadrilateral.intersection(polygon), envelope.shape)
        mask &= envelope & ~assigned
        cells[index] = mask
        assigned |= mask

    # Chordal quadrilaterals can omit a thin raster crescent along a curved
    # vector edge.  Assign those pixels by nearest along-axis station so the
    # public cell masks exactly and disjointly partition the envelope.
    unassigned_yx = np.argwhere(envelope & ~assigned)
    if len(unassigned_yx):
        guide_length = float(
            np.linalg.norm(np.diff(guide_axis, axis=0), axis=1).sum()
        )
        dense_spacing = min(1.0, max(0.25, guide_length / 10_000.0))
        dense_axis = resample_polyline(guide_axis, spacing=dense_spacing)
        dense_distances = np.r_[
            0.0,
            np.cumsum(np.linalg.norm(np.diff(dense_axis, axis=0), axis=1)),
        ]
        nearest = cKDTree(dense_axis).query(unassigned_yx[:, ::-1], workers=-1)[1]
        labels = np.searchsorted(
            target_distances, dense_distances[nearest], side="right"
        ) - 1
        labels = np.clip(labels, 0, len(cells) - 1)
        for cell_index in np.unique(labels):
            selected = unassigned_yx[labels == cell_index]
            cells[cell_index, selected[:, 0], selected[:, 1]] = True

    if np.any(np.count_nonzero(cells, axis=0) > 1):
        raise RuntimeError("internal error: constructed cell masks overlap")
    if not np.array_equal(np.any(cells, axis=0), envelope):
        raise RuntimeError("internal error: cells do not cover the belt envelope")
    if np.any(np.count_nonzero(cells, axis=(1, 2)) == 0):
        raise RuntimeError(
            "spacing creates empty raster cells; use spacing of at least one pixel"
        )
    return cells


def channel_belt_from_masks(
    channel_masks: Sequence[ArrayLike] | NDArray[np.generic],
    exit_sides: str,
    nominal_width: float,
    spacing: float,
    *,
    dilation_factor: float = 10.0,
    smoothing_factor: float = 50.0,
    dilation_increment_factor: float = 1.0 / 3.0,
    padding_factor: float = 20.0,
    max_iterations: int = 12,
    max_padded_pixels: int = 100_000_000,
) -> ChannelBelt:
    """Construct an automatic channel belt from observed channel masks.

    The default dilation (``10 * nominal_width``), smoothing window
    (``50 * nominal_width``), and padding (``20 * nominal_width``) expose
    RivMAP's empirical legacy factors.  Every dilation attempt is recomputed
    from the original temporal union and the loop is bounded by
    ``max_iterations``.  The function raises rather than returning crossed
    or incomplete cells.

    ``spacing`` is expressed in pixels and must be at least one because the
    output cells are raster masks.  The final shortfall is spread evenly over
    all cells, avoiding a tiny terminal cell.  On a strongly curved belt the
    target applies along the constructed axis, so Euclidean chords between
    returned stations can be shorter than ``spacing``; use
    ``midpoint_distances`` as the along-belt coordinate.

    A series of ``GeoRaster`` inputs preserves the common grid and CRS.
    Unknown pixels are excluded from the envelope and cells. Missing data
    must not touch any observed channel, because an incomplete channel union
    cannot define a reliable belt; crop to a fully observed reach first.
    """

    masks_input = list(channel_masks)
    referenced = [isinstance(mask, GeoRaster) for mask in masks_input]
    if any(referenced):
        if not all(referenced):
            raise ValueError("all channel masks must be GeoRaster objects; mixed inputs are ambiguous")
        grid = masks_input[0].grid
        valid = grid.valid.copy()
        for mask in masks_input:
            grid.assert_aligned(mask.grid)
            valid &= mask.valid_mask
        common_grid = grid.with_valid_mask(valid)
        for mask in masks_input:
            _validate_spatial_geometry(mask.binary_array() & valid, common_grid)
        result = channel_belt_from_masks(
            [mask.binary_array() & valid for mask in masks_input],
            exit_sides, nominal_width, spacing,
            dilation_factor=dilation_factor, smoothing_factor=smoothing_factor,
            dilation_increment_factor=dilation_increment_factor,
            padding_factor=padding_factor, max_iterations=max_iterations,
            max_padded_pixels=max_padded_pixels,
        )
        return replace(
            result, grid=common_grid,
            cell_masks=result.cell_masks & valid, envelope=result.envelope & valid,
        )

    sides = validate_exit_sides(exit_sides)
    width = _positive_scalar(nominal_width, name="nominal_width")
    cell_spacing = _positive_scalar(spacing, name="spacing")
    if cell_spacing < 1.0:
        raise ValueError("spacing must be at least one pixel for raster cells")
    dilation = _positive_scalar(dilation_factor, name="dilation_factor")
    smoothing = _nonnegative_scalar(smoothing_factor, name="smoothing_factor")
    increment = _positive_scalar(
        dilation_increment_factor, name="dilation_increment_factor"
    )
    padding = _positive_scalar(padding_factor, name="padding_factor")
    iteration_limit = _positive_integer(max_iterations, name="max_iterations")
    pixel_limit = _positive_integer(max_padded_pixels, name="max_padded_pixels")
    masks, _ = _validate_masks(masks_input, sides)
    union = np.logical_or.reduce(masks)

    polygon, envelope = _build_envelope(
        union,
        sides,
        width,
        dilation_factor=dilation,
        smoothing_factor=smoothing,
        dilation_increment_factor=increment,
        padding_factor=padding,
        max_iterations=iteration_limit,
        max_padded_pixels=pixel_limit,
    )
    try:
        (
            left,
            right,
            centerline,
            midpoint_distances,
            station_distances,
        ) = _station_edges(
            polygon, envelope, sides, width, cell_spacing, smoothing
        )
        _validated_quadrilaterals(left, right)
    except (RuntimeError, ValueError):
        (
            left,
            right,
            centerline,
            midpoint_distances,
            station_distances,
        ) = _proportional_station_edges(envelope, sides, cell_spacing)
    cells = _partition_envelope(
        polygon,
        envelope,
        left,
        right,
        centerline,
        station_distances,
    )

    if np.any(union & ~envelope):
        raise RuntimeError("internal error: returned envelope omits channel pixels")
    return ChannelBelt(
        left=np.asarray(left, dtype=float),
        right=np.asarray(right, dtype=float),
        centerline=np.asarray(centerline, dtype=float),
        midpoint_distances=np.asarray(midpoint_distances, dtype=float),
        cell_masks=np.asarray(cells, dtype=bool),
        envelope=np.asarray(envelope, dtype=bool),
    )


__all__ = ["ChannelBelt", "channel_belt_from_masks"]
