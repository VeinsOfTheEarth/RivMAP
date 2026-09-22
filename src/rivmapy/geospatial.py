"""Small, explicit helpers for combining georeferenced RivMAPy results.

The original MATLAB routine accepted opaque Mapping Toolbox reference objects
and inferred whether its inputs were rasters or centerlines.  RivMAPy keeps the
two operations explicit: :func:`mosaic_rasters` combines aligned north-up
rasters, while :func:`stitch_polylines` joins line pieces that are already in a
shared coordinate system.

Raster transforms use the ``affine.Affine``/``rasterio.Affine`` convention and
describe pixel *corners*.  Pixel ``(0, 0)`` is therefore centred at
``transform * (0.5, 0.5)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import as_coordinates


OverlapPolicy = Literal["first", "last"]


@dataclass(frozen=True)
class MosaicResult:
    """A mosaicked raster and the transform for its upper-left pixel corner."""

    array: NDArray[Any]
    transform: Any
    coverage: NDArray[np.bool_]
    crs: Any | None = None
    nodata: float | int | None = np.nan


def _affine_class():
    try:
        from affine import Affine
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "RivMAPy's geospatial helpers require the 'geospatial' extra: "
            "install RivMAPy[geospatial]"
        ) from exc
    return Affine


def _as_affine(transform: Any):
    Affine = _affine_class()
    if isinstance(transform, Affine):
        return transform
    try:
        values = tuple(transform)
    except TypeError as exc:
        raise TypeError("each transform must be affine-like") from exc
    if len(values) == 6:
        return Affine(*values)
    if len(values) == 9:
        return Affine(*values[:6])
    raise ValueError("an affine transform must contain six coefficients")


def _validate_north_up(transform: Any, *, atol: float) -> tuple[float, float]:
    if not np.isclose(transform.b, 0.0, atol=atol) or not np.isclose(
        transform.d, 0.0, atol=atol
    ):
        raise ValueError("rotated or sheared rasters are not supported")
    if transform.a <= 0 or transform.e >= 0:
        raise ValueError("transforms must be north-up with positive x and negative y scale")
    return float(transform.a), float(-transform.e)


def _valid_pixels(array: NDArray[Any], nodata: float | int | None) -> NDArray[np.bool_]:
    if nodata is None:
        return np.ones(array.shape, dtype=bool)
    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        if np.issubdtype(array.dtype, np.inexact):
            return ~np.isnan(array)
        return np.ones(array.shape, dtype=bool)
    return array != nodata


def mosaic_rasters(
    rasters: Sequence[ArrayLike],
    transforms: Sequence[Any],
    *,
    nodata: float | int | None = np.nan,
    source_nodata: float | int | None = None,
    overlap: OverlapPolicy = "last",
    crs: Any | None = None,
    tolerance: float = 1e-7,
) -> MosaicResult:
    """Mosaic aligned, north-up two-dimensional rasters.

    Parameters
    ----------
    rasters, transforms
        Equal-length sequences.  Every raster must be two-dimensional and use
        the same pixel size and grid alignment.
    nodata
        Fill value in cells not covered by a valid source pixel.  ``NaN``
        promotes integer sources to floating point.  ``None`` uses zero as a
        storage fill while leaving ``MosaicResult.nodata`` unset; consult the
        returned ``coverage`` mask to distinguish uncovered cells.
    source_nodata
        Value to ignore in every source.  ``None`` treats every source pixel as
        valid; use ``np.nan`` to ignore NaNs.
    overlap
        ``"last"`` reproduces RivMAP's overwrite behavior. ``"first"`` keeps
        the first valid value encountered.

    Notes
    -----
    This intentionally supports the projected, axis-aligned case used by
    RivMAP.  It refuses sub-pixel offsets instead of silently resampling data.
    Use rasterio's reprojection tools before calling it when grids differ.
    """

    if not rasters:
        raise ValueError("at least one raster is required")
    if len(rasters) != len(transforms):
        raise ValueError("rasters and transforms must have the same length")
    if overlap not in {"first", "last"}:
        raise ValueError("overlap must be 'first' or 'last'")

    arrays = [np.asarray(raster) for raster in rasters]
    if any(array.ndim != 2 or array.size == 0 for array in arrays):
        raise ValueError("every raster must be a non-empty two-dimensional array")
    affine_transforms = [_as_affine(transform) for transform in transforms]
    resolutions = [
        _validate_north_up(transform, atol=tolerance)
        for transform in affine_transforms
    ]
    xres, yres = resolutions[0]
    for current_xres, current_yres in resolutions[1:]:
        if not np.isclose(current_xres, xres, rtol=tolerance, atol=tolerance) or not np.isclose(
            current_yres, yres, rtol=tolerance, atol=tolerance
        ):
            raise ValueError("all rasters must have the same pixel size")
    reference_transform = affine_transforms[0]
    for transform in affine_transforms[1:]:
        column_offset = (transform.c - reference_transform.c) / xres
        row_offset = (reference_transform.f - transform.f) / yres
        if not np.isclose(column_offset, round(column_offset), rtol=tolerance, atol=tolerance) or not np.isclose(
            row_offset, round(row_offset), rtol=tolerance, atol=tolerance
        ):
            raise ValueError("source rasters have a sub-pixel grid offset")

    lefts: list[float] = []
    rights: list[float] = []
    tops: list[float] = []
    bottoms: list[float] = []
    for array, transform in zip(arrays, affine_transforms, strict=True):
        height, width = array.shape
        lefts.append(float(transform.c))
        tops.append(float(transform.f))
        rights.append(float(transform.c + width * xres))
        bottoms.append(float(transform.f - height * yres))

    left, right = min(lefts), max(rights)
    top, bottom = max(tops), min(bottoms)
    width_float = (right - left) / xres
    height_float = (top - bottom) / yres
    width = int(round(width_float))
    height = int(round(height_float))
    if not np.isclose(width_float, width, rtol=tolerance, atol=tolerance) or not np.isclose(
        height_float, height, rtol=tolerance, atol=tolerance
    ):
        raise ValueError("raster bounds do not align to a common integer grid")

    dtype = np.result_type(*[array.dtype for array in arrays])
    fill_value: float | int = 0 if nodata is None else nodata
    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        dtype = np.result_type(dtype, np.float64)
    elif nodata is not None:
        dtype = np.result_type(dtype, np.min_scalar_type(nodata))
    output = np.full((height, width), fill_value, dtype=dtype)
    coverage = np.zeros((height, width), dtype=bool)

    for array, transform in zip(arrays, affine_transforms, strict=True):
        col_float = (transform.c - left) / xres
        row_float = (top - transform.f) / yres
        col = int(round(col_float))
        row = int(round(row_float))
        if not np.isclose(col_float, col, rtol=tolerance, atol=tolerance) or not np.isclose(
            row_float, row, rtol=tolerance, atol=tolerance
        ):
            raise ValueError("source rasters have a sub-pixel grid offset")
        rows = slice(row, row + array.shape[0])
        columns = slice(col, col + array.shape[1])
        valid = _valid_pixels(array, source_nodata)
        destination = output[rows, columns]
        destination_coverage = coverage[rows, columns]
        write = valid if overlap == "last" else valid & ~destination_coverage
        destination[write] = array[write]
        destination_coverage[write] = True

    Affine = _affine_class()
    output_transform = Affine(xres, 0.0, left, 0.0, -yres, top)
    return MosaicResult(output, output_transform, coverage, crs, nodata)


def pixel_to_world(
    coordinates: ArrayLike,
    transform: Any,
    *,
    pixel_center: bool = True,
) -> NDArray[np.float64]:
    """Transform zero-based image ``(x, y)`` coordinates to world coordinates."""

    xy = as_coordinates(coordinates, minimum_length=1)
    affine = _as_affine(transform)
    offset = 0.5 if pixel_center else 0.0
    x = xy[:, 0] + offset
    y = xy[:, 1] + offset
    return np.column_stack(
        (
            affine.a * x + affine.b * y + affine.c,
            affine.d * x + affine.e * y + affine.f,
        )
    )


def stitch_polylines(
    polylines: Sequence[ArrayLike],
    *,
    max_gap: float | None = None,
    deduplicate_tolerance: float = 0.0,
) -> NDArray[np.float64]:
    """Join polyline pieces by repeatedly attaching the closest endpoints.

    The first piece determines the final orientation, matching the original
    RivMAP routine.  ``max_gap`` can be used to reject accidentally unrelated
    pieces.  Adjacent duplicate endpoints are removed when their distance is
    no greater than ``deduplicate_tolerance``.
    """

    if not polylines:
        raise ValueError("at least one polyline is required")
    if max_gap is not None and (not np.isfinite(max_gap) or max_gap < 0):
        raise ValueError("max_gap must be finite and non-negative")
    if deduplicate_tolerance < 0 or not np.isfinite(deduplicate_tolerance):
        raise ValueError("deduplicate_tolerance must be finite and non-negative")

    remaining = [as_coordinates(line, name="polyline") for line in polylines]
    combined = remaining.pop(0)
    while remaining:
        candidates: list[tuple[float, int, Literal["prepend", "append"], bool]] = []
        for index, line in enumerate(remaining):
            candidates.extend(
                [
                    (float(np.linalg.norm(combined[0] - line[-1])), index, "prepend", False),
                    (float(np.linalg.norm(combined[0] - line[0])), index, "prepend", True),
                    (float(np.linalg.norm(combined[-1] - line[0])), index, "append", False),
                    (float(np.linalg.norm(combined[-1] - line[-1])), index, "append", True),
                ]
            )
        distance, index, location, reverse = min(candidates, key=lambda item: item[0])
        if max_gap is not None and distance > max_gap:
            raise ValueError(
                f"closest remaining polyline endpoint is {distance:g}, exceeding max_gap"
            )
        line = remaining.pop(index)
        if reverse:
            line = line[::-1]
        if location == "prepend":
            if np.linalg.norm(line[-1] - combined[0]) <= deduplicate_tolerance:
                line = line[:-1]
            combined = np.vstack((line, combined))
        else:
            if np.linalg.norm(combined[-1] - line[0]) <= deduplicate_tolerance:
                line = line[1:]
            combined = np.vstack((combined, line))
    return combined


def combine_georeferenced_images(
    items: Sequence[ArrayLike],
    transforms: Sequence[Any],
    *,
    kind: Literal["raster", "polyline"] | None = None,
    pixel_center: bool = True,
    **kwargs: Any,
) -> MosaicResult | NDArray[np.float64]:
    """Compatibility wrapper for RivMAP's ``combine_georeffed_images``.

    Prefer calling :func:`mosaic_rasters` or :func:`stitch_polylines`
    explicitly in new code.  Polyline inputs are transformed to world
    coordinates before stitching.
    """

    if not items:
        raise ValueError("at least one item is required")
    if len(items) != len(transforms):
        raise ValueError("items and transforms must have the same length")
    first = np.asarray(items[0])
    if kind is None:
        kind = "polyline" if first.ndim == 2 and first.shape[1:] == (2,) else "raster"
    if kind == "raster":
        return mosaic_rasters(items, transforms, **kwargs)
    if kind == "polyline":
        lines = [
            pixel_to_world(line, transform, pixel_center=pixel_center)
            for line, transform in zip(items, transforms, strict=True)
        ]
        return stitch_polylines(lines, **kwargs)
    raise ValueError("kind must be 'raster' or 'polyline'")


# Original MATLAB spelling retained as a discoverable compatibility alias.
combine_georeffed_images = combine_georeferenced_images

__all__ = [
    "MosaicResult",
    "combine_georeferenced_images",
    "combine_georeffed_images",
    "mosaic_rasters",
    "pixel_to_world",
    "stitch_polylines",
]
