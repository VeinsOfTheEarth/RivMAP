"""Temporal channel-change analysis.

This module ports the analytical intent of RivMAP's ``migration_mask.m``,
``migration_cl.m``, and the zonal-summary portion of ``spatial_changes.m``.
Images are Boolean arrays indexed as ``image[y, x]``; vector coordinates are
zero-based ``(x, y)`` pixel-centre coordinates.

The original MATLAB names *erosion* and *accretion* describe changes to the
land bordering the channel.  Consequently, channel pixels gained at time 2
are ``erosion`` and channel pixels lost at time 2 are ``accretion``.  That
published convention is retained here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.geometry import MultiPoint, Point, Polygon
from skimage.draw import line as draw_line
from skimage.draw import polygon as draw_polygon

from ._validation import as_coordinates, positive_number, validate_exit_sides
from .mask import CenterlineResult
from .spatial import GeoRaster, RasterGrid, SpatialResultMixin


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class MaskMigrationResult(SpatialResultMixin):
    """Pixel classifications from two channel masks.

    Attributes
    ----------
    erosion
        Pixels outside the time-1 channel and inside the time-2 channel.
    accretion
        Pixels inside the time-1 channel and outside the time-2 channel,
        excluding pixels classified as cutoffs.
    unchanged
        Pixels with equal values in both masks.  As in RivMAP, this includes
        both persistent channel and persistent background pixels.
    cutoffs
        Complete time-1-only connected components classified as cutoffs.
    """

    erosion: BoolArray
    accretion: BoolArray
    unchanged: BoolArray
    cutoffs: BoolArray
    grid: RasterGrid | None = None


@dataclass(frozen=True)
class CenterlineMigrationResult(SpatialResultMixin):
    """Swept centerline area and cutoff measurements.

    ``cutoff_indices`` contains zero-based vertex indices into the time-1
    centerline.  Its rows correspond to ``cutoff_area``, ``cutoff_length``,
    and ``chute_length``.
    """

    migrated: BoolArray
    cutoffs: BoolArray
    cutoff_indices: IntArray
    cutoff_area: FloatArray
    cutoff_length: FloatArray
    chute_length: FloatArray
    grid: RasterGrid | None = None


@dataclass(frozen=True)
class SpatialChangeResult(SpatialResultMixin):
    """Per-cell temporal summaries for a preconstructed spatial belt.

    Arrays have shape ``(n_cells, n_times)``.  ``analyzed_area`` maps each
    caller-provided analysis name to its pixel-count array.
    """

    analyzed_area: Mapping[str, NDArray[np.int64]]
    channel_area: NDArray[np.int64]
    centerline_length: FloatArray
    cell_area: NDArray[np.int64]
    grid: RasterGrid | None = None


def migration_mask(
    mask_t1: ArrayLike | GeoRaster,
    mask_t2: ArrayLike | GeoRaster,
    nominal_width: float,
    cutoff_mask: ArrayLike | GeoRaster | None = None,
    *,
    cutoff_area_factor: float = 2.0,
    connectivity: int = 8,
) -> MaskMigrationResult:
    """Classify erosion, accretion, unchanged area, and cutoffs.

    This is the array-first equivalent of ``migration_mask.m``.  If
    ``cutoff_mask`` is omitted, a time-1-only connected component is a cutoff
    when its area is strictly greater than
    ``cutoff_area_factor * nominal_width**2``.  The default factor of 2 is
    the value used by the released MATLAB implementation (the publication
    describes a factor of 3 for one application).

    If ``cutoff_mask`` is supplied, every time-1-only component touching it
    is a cutoff.  Unlike an omission in the MATLAB ``else`` branch, selected
    components are always removed from ``accretion``.

    Both masks may be ``GeoRaster`` objects on the same grid and CRS. Their
    common observation validity is retained, and unknown pixels are excluded
    from every classification (including ``unchanged``). Automatic cutoff
    classification requires an abandoned component not to touch nodata;
    explicit cutoff seeds can identify a partially observed component.
    """

    if isinstance(mask_t1, GeoRaster) or isinstance(mask_t2, GeoRaster):
        if not isinstance(mask_t1, GeoRaster) or not isinstance(mask_t2, GeoRaster):
            raise ValueError("both masks must be GeoRaster objects; mixed inputs lose georeferencing")
        mask_t1.grid.assert_aligned(mask_t2.grid)
        valid = mask_t1.valid_mask & mask_t2.valid_mask
        grid = mask_t1.grid.with_valid_mask(valid)
        seed = cutoff_mask
        if isinstance(seed, GeoRaster):
            grid.assert_aligned(seed.grid)
            seed = seed.binary_array() & valid
        elif seed is not None:
            seed = _as_binary_image(seed, name="cutoff_mask")
            if seed.shape != grid.shape:
                raise ValueError("cutoff_mask must match the channel-mask shape")
            seed &= valid
        result = migration_mask(
            mask_t1.binary_array() & valid, mask_t2.binary_array() & valid,
            nominal_width, seed,
            cutoff_area_factor=cutoff_area_factor, connectivity=connectivity,
        )
        if cutoff_mask is None and not np.all(valid) and np.any(result.cutoffs):
            # The whole abandoned component must be observed before its area
            # can establish a cutoff.  The observed lost-channel pixels remain
            # accretion when a component reaches an unknown part of the image.
            labels, _ = ndi.label(result.cutoffs, structure=_connectivity_structure(connectivity))
            unknown_neighbors = ndi.binary_dilation(~valid, structure=np.ones((3, 3), dtype=bool))
            incomplete = np.unique(labels[unknown_neighbors & result.cutoffs])
            uncertain = np.isin(labels, incomplete[incomplete > 0])
            result = replace(
                result, cutoffs=result.cutoffs & ~uncertain,
                accretion=result.accretion | uncertain,
            )
        return replace(result, unchanged=result.unchanged & valid, grid=grid)
    if isinstance(cutoff_mask, GeoRaster):
        raise ValueError("a georeferenced cutoff mask requires georeferenced channel masks")

    first = _as_binary_image(mask_t1, name="mask_t1")
    second = _as_binary_image(mask_t2, name="mask_t2")
    _require_same_shape(first, second, names=("mask_t1", "mask_t2"))
    width = positive_number(nominal_width, name="nominal_width")
    factor = positive_number(cutoff_area_factor, name="cutoff_area_factor")
    structure = _connectivity_structure(connectivity)

    erosion = ~first & second
    accretion = first & ~second
    unchanged = first == second

    labels, count = ndi.label(accretion, structure=structure)
    selected = np.zeros(count + 1, dtype=bool)
    if count:
        if cutoff_mask is None:
            areas = np.bincount(labels.ravel(), minlength=count + 1)
            selected[1:] = areas[1:] > factor * width**2
        else:
            supplied = _as_binary_image(cutoff_mask, name="cutoff_mask")
            _require_same_shape(first, supplied, names=("mask_t1", "cutoff_mask"))
            touched = np.unique(labels[supplied & (labels > 0)])
            selected[touched] = True
    elif cutoff_mask is not None:
        supplied = _as_binary_image(cutoff_mask, name="cutoff_mask")
        _require_same_shape(first, supplied, names=("mask_t1", "cutoff_mask"))

    cutoffs = selected[labels]
    accretion &= ~cutoffs
    return MaskMigrationResult(
        erosion=erosion,
        accretion=accretion,
        unchanged=unchanged,
        cutoffs=cutoffs,
    )


def migration_centerlines(
    centerline_t1: ArrayLike | CenterlineResult,
    centerline_t2: ArrayLike | CenterlineResult,
    shape: Sequence[int],
    nominal_width: float,
    *,
    exit_sides_t1: str | None = None,
    exit_sides_t2: str | None = None,
    cutoff_length_factor: float = 2.0,
    neighboring_area_factor: float = 5.0,
    tail_erosion_iterations: int = 5,
    tail_dilation_iterations: int = 10,
    connectivity: int = 8,
    grid: RasterGrid | None = None,
) -> CenterlineMigrationResult:
    """Rasterize swept area between two centerlines and identify cutoffs.

    Centerlines must be ordered consistently from upstream to downstream.
    Coordinates are rounded to pixel centres before analysis, matching the
    MATLAB routine.  A cutoff is an interval between consecutive line
    intersections for which the time-1 length exceeds the time-2 length by
    more than ``cutoff_length_factor * nominal_width``.

    The defaults retain RivMAP's 2-width shortening threshold, its 5-width²
    neighboring-polygon check, and its 5-erode/10-dilate tail cleanup.  Empty
    erosion results safely fall back to the uncleaned cutoff rather than
    failing as the MATLAB implementation did for small cutoffs.

    Referenced ``CenterlineResult`` inputs preserve their aligned source grid.
    Raster products and cutoff areas exclude pixels that are unknown in
    either observation. ``shape`` must still match that grid; ``grid`` can
    explicitly attach known georeferencing to pixel-coordinate arrays.
    """

    first_grid = centerline_t1.grid if isinstance(centerline_t1, CenterlineResult) else None
    second_grid = centerline_t2.grid if isinstance(centerline_t2, CenterlineResult) else None
    if (first_grid is None) != (second_grid is None):
        raise ValueError("both centerlines must carry georeferencing; mixed inputs are ambiguous")
    if first_grid is not None:
        first_grid.assert_aligned(second_grid)
        if grid is not None:
            grid.assert_aligned(first_grid)
            grid = grid.with_valid_mask(grid.valid & first_grid.valid & second_grid.valid)
        else:
            grid = first_grid.with_valid_mask(first_grid.valid & second_grid.valid)
    if isinstance(centerline_t1, CenterlineResult):
        centerline_t1 = centerline_t1.coordinates
    if isinstance(centerline_t2, CenterlineResult):
        centerline_t2 = centerline_t2.coordinates
    raster_shape = _validate_shape(shape)
    if grid is not None and grid.shape != raster_shape:
        raise ValueError("shape must match the georeferenced centerline grid")
    width = positive_number(nominal_width, name="nominal_width")
    length_factor = positive_number(
        cutoff_length_factor, name="cutoff_length_factor"
    )
    neighbor_factor = positive_number(
        neighboring_area_factor, name="neighboring_area_factor"
    )
    tail_erode = _nonnegative_integer(
        tail_erosion_iterations, name="tail_erosion_iterations"
    )
    tail_dilate = _nonnegative_integer(
        tail_dilation_iterations, name="tail_dilation_iterations"
    )
    structure = _connectivity_structure(connectivity)

    first = _rounded_centerline(centerline_t1, raster_shape, name="centerline_t1")
    second = _rounded_centerline(centerline_t2, raster_shape, name="centerline_t2")
    cumulative_first = _cumulative_distance(first, name="centerline_t1")
    cumulative_second = _cumulative_distance(second, name="centerline_t2")

    sides_first = (
        _infer_exit_sides(first, raster_shape)
        if exit_sides_t1 is None
        else validate_exit_sides(exit_sides_t1)
    )
    sides_second = (
        _infer_exit_sides(second, raster_shape)
        if exit_sides_t2 is None
        else validate_exit_sides(exit_sides_t2)
    )
    range_first = np.ptp(first, axis=0)
    range_second = np.ptp(second, axis=0)
    # MATLAB chooses the second line on equal bounding-box products.
    sides = (
        sides_first
        if float(np.prod(range_first)) < float(np.prod(range_second))
        else sides_second
    )

    line_first = _rasterize_polyline(first, raster_shape)
    line_second = _rasterize_polyline(second, raster_shape)
    row_slice, column_slice = _shared_exit_domain(first, second, sides, raster_shape)
    first_crop = line_first[row_slice, column_slice]
    second_crop = line_second[row_slice, column_slice]
    half_first = _filled_line_half(first_crop, sides)
    half_second = _filled_line_half(second_crop, sides)
    migrated_crop = half_first ^ half_second
    # Preserve the original asymmetric boundary convention: t2 is included,
    # t1 is excluded, and common line pixels are therefore excluded.
    migrated_crop[second_crop] = True
    migrated_crop[first_crop] = False
    migrated = np.zeros(raster_shape, dtype=bool)
    migrated[row_slice, column_slice] = migrated_crop
    if grid is not None:
        migrated &= grid.valid

    cutoffs = np.zeros(raster_shape, dtype=bool)
    points, positions_first, positions_second = _polyline_intersections(first, second)
    if len(points) < 2:
        return replace(_centerline_result(migrated, cutoffs, [], [], [], []), grid=grid)

    old_spans = np.diff(positions_first)
    new_spans = np.abs(np.diff(positions_second))
    candidates = np.flatnonzero(
        old_spans - new_spans > length_factor * width
    )
    if not len(candidates):
        return replace(_centerline_result(migrated, cutoffs, [], [], [], []), grid=grid)

    cutoff_indices: list[tuple[int, int]] = []
    cutoff_areas: list[float] = []
    cutoff_lengths: list[float] = []
    chute_lengths: list[float] = []

    for candidate in candidates:
        upstream_intersection = int(candidate)
        downstream_intersection = int(candidate + 1)

        # RivMAP expands a candidate to a neighboring intersection if that
        # adjacent swept polygon is large.  The MATLAB code indexed outside
        # the array for end candidates; bounds checks here fix that defect.
        if upstream_intersection > 0:
            area = _interval_polygon_area(
                first,
                second,
                positions_first[upstream_intersection - 1],
                positions_first[upstream_intersection],
                positions_second[upstream_intersection - 1],
                positions_second[upstream_intersection],
                cumulative_first,
                cumulative_second,
            )
            if area > neighbor_factor * width**2:
                upstream_intersection -= 1
        if downstream_intersection + 1 < len(points):
            area = _interval_polygon_area(
                first,
                second,
                positions_first[downstream_intersection],
                positions_first[downstream_intersection + 1],
                positions_second[downstream_intersection],
                positions_second[downstream_intersection + 1],
                cumulative_first,
                cumulative_second,
            )
            if area > neighbor_factor * width**2:
                downstream_intersection += 1

        old_start = positions_first[upstream_intersection]
        old_stop = positions_first[downstream_intersection]
        new_start = positions_second[upstream_intersection]
        new_stop = positions_second[downstream_intersection]
        old_segment = _subline(first, cumulative_first, old_start, old_stop)
        new_segment = _subline(second, cumulative_second, new_start, new_stop)
        polygon_vertices = np.vstack((old_segment, new_segment[::-1]))
        cutoff_envelope = _rasterize_polygon(polygon_vertices, raster_shape)
        cutoff_envelope = ndi.binary_dilation(cutoff_envelope, iterations=1)
        raw_cutoff = migrated & cutoff_envelope
        if not np.any(raw_cutoff):
            # The bounded polygon is the scientific object of interest.  This
            # fallback covers interior endpoints for which no image half can
            # be closed at a raster edge.
            raw_cutoff = cutoff_envelope & ~line_first
        cleaned = _clean_cutoff(
            raw_cutoff,
            erosion_iterations=tail_erode,
            dilation_iterations=tail_dilate,
            structure=structure,
        )
        if grid is not None:
            cleaned &= grid.valid
        if not np.any(cleaned):
            continue

        # Do not double-count overlapping expanded candidates.
        cleaned &= ~cutoffs
        if not np.any(cleaned):
            continue
        cutoffs |= cleaned
        migrated[cleaned] = False

        start_index = _nearest_vertex_index(cumulative_first, old_start)
        stop_index = _nearest_vertex_index(cumulative_first, old_stop)
        cutoff_indices.append(tuple(sorted((start_index, stop_index))))
        cutoff_areas.append(float(np.count_nonzero(cleaned)))
        cutoff_lengths.append(float(abs(old_stop - old_start)))
        chute_lengths.append(float(abs(new_stop - new_start)))

    return replace(_centerline_result(
        migrated,
        cutoffs,
        cutoff_indices,
        cutoff_areas,
        cutoff_lengths,
        chute_lengths,
    ), grid=grid)


def migration_cl(
    centerline_t1: ArrayLike,
    centerline_t2: ArrayLike,
    exit_sides_t1: str,
    exit_sides_t2: str,
    nominal_width: float,
    shape: Sequence[int],
    **kwargs: object,
) -> CenterlineMigrationResult:
    """Compatibility spelling and argument order for ``migration_cl.m``.

    The return value is a :class:`CenterlineMigrationResult`, not the six
    positional MATLAB outputs.
    """

    return migration_centerlines(
        centerline_t1,
        centerline_t2,
        shape,
        nominal_width,
        exit_sides_t1=exit_sides_t1,
        exit_sides_t2=exit_sides_t2,
        **kwargs,
    )


def spatial_cells_from_edges(
    left_edge: ArrayLike,
    right_edge: ArrayLike,
    shape: Sequence[int],
) -> BoolArray:
    """Rasterize non-overlapping quadrilateral cells between paired edges.

    The edge arrays must contain corresponding, upstream-to-downstream
    vertices.  Shared raster-boundary pixels are assigned to the upstream
    cell, making the returned cell masks disjoint.

    This lower-level function creates cells from reviewed edges.  For
    automatic belt construction from a time series of channel masks, use
    :func:`rivmapy.belt.channel_belt_from_masks`.
    """

    left = as_coordinates(left_edge, name="left_edge")
    right = as_coordinates(right_edge, name="right_edge")
    if left.shape != right.shape:
        raise ValueError("left_edge and right_edge must have matching shapes")
    raster_shape = _validate_shape(shape)
    cells = np.zeros((len(left) - 1, *raster_shape), dtype=bool)
    assigned = np.zeros(raster_shape, dtype=bool)
    for index in range(len(cells)):
        vertices = np.vstack(
            (left[index], right[index], right[index + 1], left[index + 1])
        )
        cell = _rasterize_polygon(vertices, raster_shape)
        cell &= ~assigned
        cells[index] = cell
        assigned |= cell
    return cells


def spatial_changes(
    cell_masks: ArrayLike,
    channel_masks: Sequence[ArrayLike],
    centerlines: Sequence[ArrayLike],
    analyzed_masks: Mapping[str, Sequence[ArrayLike]] | None = None,
) -> SpatialChangeResult:
    """Summarize temporal masks and centerline lengths within belt cells.

    This is the reusable analytical core of ``spatial_changes.m``.  Belt
    construction is separate: pass ``ChannelBelt.cell_masks`` from
    :func:`rivmapy.belt.channel_belt_from_masks`, the result of
    :func:`spatial_cells_from_edges`, or another reviewed set of disjoint cell
    masks.

    Centerline length is split exactly at pixel-footprint boundaries and
    assigned once to the cell containing each subsegment midpoint.  Thus a
    line on a cell boundary cannot be double-counted.

    A ``ChannelBelt`` can be passed directly in place of ``cell_masks``.
    Georeferenced workflows use ``GeoRaster`` channel/analysis masks and
    ``CenterlineResult`` lines. Their grids must align, and all summaries use
    the intersection of the supplied observation-validity masks. Results
    retain that common grid; areas and lengths remain in pixel units.
    """

    # Import locally: belt construction uses this module's raster helpers.
    from .belt import ChannelBelt

    belt = cell_masks if isinstance(cell_masks, ChannelBelt) else None
    belt_grid = belt.grid if belt is not None else None
    if belt is not None:
        cell_masks = belt.cell_masks
    referenced = [isinstance(mask, GeoRaster) for mask in channel_masks]
    if any(referenced) or belt_grid is not None:
        if not referenced or not all(referenced):
            raise ValueError("all channel masks must be GeoRaster objects in a spatial workflow")
        if len(centerlines) != len(channel_masks):
            raise ValueError("channel_masks and centerlines must have equal lengths")
        grid = channel_masks[0].grid
        valid = grid.valid.copy()
        if belt_grid is not None:
            grid.assert_aligned(belt_grid)
            valid &= belt_grid.valid
        for mask in channel_masks:
            grid.assert_aligned(mask.grid)
            valid &= mask.valid_mask
        lines = []
        for line in centerlines:
            if not isinstance(line, CenterlineResult) or line.grid is None:
                raise ValueError("centerlines must carry georeferencing in a spatial workflow")
            grid.assert_aligned(line.grid)
            valid &= line.grid.valid
            lines.append(line.coordinates)
        analyses = {}
        for name, masks in ({} if analyzed_masks is None else analyzed_masks).items():
            values = []
            for mask in masks:
                if not isinstance(mask, GeoRaster):
                    raise ValueError("analyzed masks must be GeoRaster objects in a spatial workflow")
                grid.assert_aligned(mask.grid)
                valid &= mask.valid_mask
                values.append(mask.binary_array())
            analyses[name] = values
        result = spatial_changes(
            np.asarray(cell_masks, dtype=bool) & valid,
            [mask.binary_array() & valid for mask in channel_masks], lines,
            {name: [value & valid for value in values] for name, values in analyses.items()},
        )
        return replace(result, grid=grid.with_valid_mask(valid))
    if any(isinstance(line, CenterlineResult) and line.grid is not None for line in centerlines):
        raise ValueError("georeferenced centerlines require GeoRaster channel masks")
    centerlines = [line.coordinates if isinstance(line, CenterlineResult) else line for line in centerlines]

    cells_array = np.asarray(cell_masks)
    if cells_array.ndim != 3 or cells_array.shape[0] == 0:
        raise ValueError("cell_masks must have shape (n_cells, rows, columns)")
    cells = np.empty(cells_array.shape, dtype=bool)
    for index in range(len(cells_array)):
        cells[index] = _as_binary_image(
            cells_array[index], name=f"cell_masks[{index}]"
        )
    if np.any(np.count_nonzero(cells, axis=0) > 1):
        raise ValueError("cell_masks must not overlap")
    if len(channel_masks) == 0:
        raise ValueError("channel_masks must contain at least one time step")
    if len(channel_masks) != len(centerlines):
        raise ValueError("channel_masks and centerlines must have equal lengths")

    image_shape = cells.shape[1:]
    channels: list[BoolArray] = []
    lines: list[FloatArray] = []
    for time, (mask, line) in enumerate(zip(channel_masks, centerlines, strict=True)):
        channel = _as_binary_image(mask, name=f"channel_masks[{time}]")
        if channel.shape != image_shape:
            raise ValueError("all channel masks must match the cell-mask shape")
        channels.append(channel)
        lines.append(as_coordinates(line, name=f"centerlines[{time}]"))

    analyses = {} if analyzed_masks is None else dict(analyzed_masks)
    validated_analyses: dict[str, list[BoolArray]] = {}
    for name, masks in analyses.items():
        if not isinstance(name, str) or not name:
            raise ValueError("analyzed_masks keys must be non-empty strings")
        if len(masks) != len(channels):
            raise ValueError(
                f"analyzed_masks[{name!r}] must have one mask per time step"
            )
        values: list[BoolArray] = []
        for time, mask in enumerate(masks):
            value = _as_binary_image(mask, name=f"analyzed_masks[{name!r}][{time}]")
            if value.shape != image_shape:
                raise ValueError("all analyzed masks must match the cell-mask shape")
            values.append(value)
        validated_analyses[name] = values

    n_cells = len(cells)
    n_times = len(channels)
    channel_area = np.zeros((n_cells, n_times), dtype=np.int64)
    centerline_length = np.zeros((n_cells, n_times), dtype=float)
    analyzed_area = {
        name: np.zeros((n_cells, n_times), dtype=np.int64)
        for name in validated_analyses
    }
    for time, (channel, line) in enumerate(zip(channels, lines, strict=True)):
        channel_area[:, time] = np.count_nonzero(cells & channel, axis=(1, 2))
        centerline_length[:, time] = _lengths_in_cells(line, cells)
        for name, masks in validated_analyses.items():
            analyzed_area[name][:, time] = np.count_nonzero(
                cells & masks[time], axis=(1, 2)
            )

    return SpatialChangeResult(
        analyzed_area=MappingProxyType(analyzed_area),
        channel_area=channel_area,
        centerline_length=centerline_length,
        cell_area=np.count_nonzero(cells, axis=(1, 2)).astype(np.int64),
    )


def _as_binary_image(image: ArrayLike, *, name: str) -> BoolArray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(array.dtype, np.bool_):
        return np.array(array, dtype=bool, copy=True)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain Boolean or binary numeric values")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only zero and one")
    return np.array(array, dtype=bool, copy=True)


def _require_same_shape(
    first: NDArray[np.generic],
    second: NDArray[np.generic],
    *,
    names: tuple[str, str],
) -> None:
    if first.shape != second.shape:
        raise ValueError(f"{names[0]} and {names[1]} must have matching shapes")


def _connectivity_structure(connectivity: int) -> BoolArray:
    if connectivity == 4:
        return ndi.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndi.generate_binary_structure(2, 2)
    raise ValueError("connectivity must be 4 or 8")


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape must contain (rows, columns)")
    rows, columns = shape
    if (
        isinstance(rows, (bool, np.bool_))
        or isinstance(columns, (bool, np.bool_))
        or int(rows) != rows
        or int(columns) != columns
        or rows <= 0
        or columns <= 0
    ):
        raise ValueError("shape values must be positive integers")
    return int(rows), int(columns)


def _rounded_centerline(
    coordinates: ArrayLike,
    shape: tuple[int, int],
    *,
    name: str,
) -> FloatArray:
    line = as_coordinates(coordinates, name=name)
    # MATLAB rounds positive half-integers away from zero; raster coordinates
    # are required to be non-negative, so floor(x + 0.5) is equivalent.
    if np.any(line < 0):
        raise ValueError(f"{name} coordinates must lie inside shape")
    rounded = np.floor(line + 0.5)
    if np.any(rounded[:, 0] >= shape[1]) or np.any(rounded[:, 1] >= shape[0]):
        raise ValueError(f"{name} coordinates must lie inside shape")
    keep = np.r_[True, np.any(np.diff(rounded, axis=0) != 0, axis=1)]
    rounded = rounded[keep]
    if len(rounded) < 2:
        raise ValueError(f"{name} must span at least two distinct pixels")
    return rounded.astype(float, copy=False)


def _cumulative_distance(line: FloatArray, *, name: str) -> FloatArray:
    increments = np.linalg.norm(np.diff(line, axis=0), axis=1)
    if not np.all(increments > 0):
        raise ValueError(f"{name} must not contain duplicate adjacent points")
    return np.r_[0.0, np.cumsum(increments)]


def _rasterize_polyline(line: FloatArray, shape: tuple[int, int]) -> BoolArray:
    result = np.zeros(shape, dtype=bool)
    pixels = line.astype(int)
    for start, stop in zip(pixels[:-1], pixels[1:], strict=True):
        rows, columns = draw_line(start[1], start[0], stop[1], stop[0])
        result[rows, columns] = True
    return result


def _infer_exit_sides(line: FloatArray, shape: tuple[int, int]) -> str:
    def nearest_side(point: FloatArray, forbidden: str | None = None) -> str:
        distances = {
            "N": point[1],
            "S": shape[0] - 1 - point[1],
            "W": point[0],
            "E": shape[1] - 1 - point[0],
        }
        order = "NSWE"
        return min(
            (side for side in order if side != forbidden),
            key=lambda side: (distances[side], order.index(side)),
        )

    upstream = nearest_side(line[0])
    downstream = nearest_side(line[-1], forbidden=upstream)
    return upstream + downstream


def _shared_exit_domain(
    first: FloatArray,
    second: FloatArray,
    exit_sides: str,
    shape: tuple[int, int],
) -> tuple[slice, slice]:
    row_start, row_stop = 0, shape[0]
    column_start, column_stop = 0, shape[1]
    if "N" in exit_sides:
        row_start = int(max(np.min(first[:, 1]), np.min(second[:, 1])))
    if "S" in exit_sides:
        row_stop = int(min(np.max(first[:, 1]), np.max(second[:, 1]))) + 1
    if "W" in exit_sides:
        column_start = int(max(np.min(first[:, 0]), np.min(second[:, 0])))
    if "E" in exit_sides:
        column_stop = int(min(np.max(first[:, 0]), np.max(second[:, 0]))) + 1
    if row_start >= row_stop or column_start >= column_stop:
        raise ValueError("centerlines have no shared domain between their exits")
    return slice(row_start, row_stop), slice(column_start, column_stop)


def _filled_line_half(line_mask: BoolArray, exit_sides: str) -> BoolArray:
    padded = np.pad(line_mask, 1, mode="constant", constant_values=True)
    if exit_sides in {"SN", "NS", "NW", "WN", "SW", "WS"}:
        opened = padded[:, :-1]
        return ndi.binary_fill_holes(opened)[1:-1, 1:]
    if exit_sides in {"EW", "WE"}:
        opened = padded[:-1, :]
        return ndi.binary_fill_holes(opened)[1:, 1:-1]
    if exit_sides in {"NE", "EN", "SE", "ES"}:
        opened = padded[:, 1:]
        return ndi.binary_fill_holes(opened)[1:-1, :-1]
    raise ValueError(f"unsupported exit-side pair {exit_sides!r}")


def _geometry_points(geometry: object) -> list[Point]:
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    if isinstance(geometry, LineString):
        coordinates = list(geometry.coords)
        if not coordinates:
            return []
        return [Point(coordinates[0]), Point(coordinates[-1])]
    if isinstance(geometry, (MultiLineString, GeometryCollection)):
        points: list[Point] = []
        for part in geometry.geoms:
            points.extend(_geometry_points(part))
        return points
    return []


def _polyline_intersections(
    first: FloatArray,
    second: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    first_geometry = LineString(first)
    second_geometry = LineString(second)
    raw_points = _geometry_points(first_geometry.intersection(second_geometry))
    records: list[tuple[float, float, float, float]] = []
    tolerance = 1e-8
    for point in raw_points:
        position_first = float(first_geometry.project(point))
        position_second = float(second_geometry.project(point))
        record = (position_first, position_second, float(point.x), float(point.y))
        if not any(
            abs(position_first - old[0]) <= tolerance
            and abs(position_second - old[1]) <= tolerance
            for old in records
        ):
            records.append(record)
    records.sort(key=lambda value: (value[0], value[1]))
    if not records:
        empty_points = np.empty((0, 2), dtype=float)
        empty_positions = np.empty(0, dtype=float)
        return empty_points, empty_positions, empty_positions.copy()
    array = np.asarray(records, dtype=float)
    return array[:, 2:4], array[:, 0], array[:, 1]


def _subline(
    line: FloatArray,
    cumulative: FloatArray,
    start: float,
    stop: float,
) -> FloatArray:
    if stop < start:
        return _subline(line, cumulative, stop, start)[::-1]
    start = float(np.clip(start, 0.0, cumulative[-1]))
    stop = float(np.clip(stop, 0.0, cumulative[-1]))

    def point_at(distance: float) -> FloatArray:
        if distance <= 0:
            return line[0].copy()
        if distance >= cumulative[-1]:
            return line[-1].copy()
        index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        fraction = (distance - cumulative[index]) / (
            cumulative[index + 1] - cumulative[index]
        )
        return line[index] + fraction * (line[index + 1] - line[index])

    interior = line[(cumulative > start) & (cumulative < stop)]
    points = np.vstack((point_at(start), interior, point_at(stop)))
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
    return points[keep]


def _interval_polygon_area(
    first: FloatArray,
    second: FloatArray,
    first_start: float,
    first_stop: float,
    second_start: float,
    second_stop: float,
    cumulative_first: FloatArray,
    cumulative_second: FloatArray,
) -> float:
    old_segment = _subline(first, cumulative_first, first_start, first_stop)
    new_segment = _subline(second, cumulative_second, second_start, second_stop)
    vertices = np.vstack((old_segment, new_segment[::-1]))
    if len(np.unique(vertices, axis=0)) < 3:
        return 0.0
    polygon = Polygon(vertices)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return float(polygon.area)


def _rasterize_polygon(vertices: FloatArray, shape: tuple[int, int]) -> BoolArray:
    result = np.zeros(shape, dtype=bool)
    if len(np.unique(vertices, axis=0)) < 3:
        return result
    rows, columns = draw_polygon(vertices[:, 1], vertices[:, 0], shape=shape)
    result[rows, columns] = True
    return result


def _clean_cutoff(
    cutoff: BoolArray,
    *,
    erosion_iterations: int,
    dilation_iterations: int,
    structure: BoolArray,
) -> BoolArray:
    cleaned = cutoff.copy()
    if erosion_iterations:
        eroded = ndi.binary_erosion(
            cleaned, structure=structure, iterations=erosion_iterations
        )
        if np.any(eroded):
            cleaned = eroded
            if dilation_iterations:
                cleaned = ndi.binary_dilation(
                    cleaned, structure=structure, iterations=dilation_iterations
                )
            cleaned &= cutoff
    elif dilation_iterations:
        cleaned = ndi.binary_dilation(
            cleaned, structure=structure, iterations=dilation_iterations
        )
        cleaned &= cutoff
    return _largest_component(cleaned, structure=structure)


def _largest_component(mask: BoolArray, *, structure: BoolArray) -> BoolArray:
    labels, count = ndi.label(mask, structure=structure)
    if count == 0:
        return np.zeros_like(mask)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    areas[0] = 0
    return labels == int(np.argmax(areas))


def _nearest_vertex_index(cumulative: FloatArray, distance: float) -> int:
    return int(np.argmin(np.abs(cumulative - distance)))


def _centerline_result(
    migrated: BoolArray,
    cutoffs: BoolArray,
    cutoff_indices: Sequence[tuple[int, int]],
    cutoff_area: Sequence[float],
    cutoff_length: Sequence[float],
    chute_length: Sequence[float],
) -> CenterlineMigrationResult:
    indices = np.asarray(cutoff_indices, dtype=int)
    if not len(indices):
        indices = np.empty((0, 2), dtype=int)
    else:
        indices = indices.reshape(-1, 2)
    return CenterlineMigrationResult(
        migrated=np.asarray(migrated, dtype=bool),
        cutoffs=np.asarray(cutoffs, dtype=bool),
        cutoff_indices=indices,
        cutoff_area=np.asarray(cutoff_area, dtype=float),
        cutoff_length=np.asarray(cutoff_length, dtype=float),
        chute_length=np.asarray(chute_length, dtype=float),
    )


def _lengths_in_cells(line: FloatArray, cells: BoolArray) -> FloatArray:
    labels = np.full(cells.shape[1:], -1, dtype=int)
    for index, cell in enumerate(cells):
        labels[cell] = index
    lengths = np.zeros(len(cells), dtype=float)
    rows, columns = labels.shape

    for start, stop in zip(line[:-1], line[1:], strict=True):
        delta = stop - start
        segment_length = float(np.linalg.norm(delta))
        if segment_length == 0:
            continue
        breakpoints = [0.0, 1.0]
        for coordinate, change in zip(start, delta, strict=True):
            if change == 0:
                continue
            low = min(coordinate, coordinate + change)
            high = max(coordinate, coordinate + change)
            candidate_boundaries = np.arange(
                np.floor(low) - 1, np.ceil(high) + 1, dtype=float
            ) + 0.5
            for boundary in candidate_boundaries:
                if low < boundary < high:
                    breakpoints.append(float((boundary - coordinate) / change))
        ordered = np.unique(np.clip(breakpoints, 0.0, 1.0))
        for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
            midpoint = start + ((lower + upper) / 2.0) * delta
            column = int(np.floor(midpoint[0] + 0.5))
            row = int(np.floor(midpoint[1] + 0.5))
            if 0 <= row < rows and 0 <= column < columns:
                cell_index = labels[row, column]
                if cell_index >= 0:
                    lengths[cell_index] += segment_length * (upper - lower)
    return lengths


__all__ = [
    "CenterlineMigrationResult",
    "MaskMigrationResult",
    "SpatialChangeResult",
    "migration_centerlines",
    "migration_cl",
    "migration_mask",
    "spatial_cells_from_edges",
    "spatial_changes",
]
