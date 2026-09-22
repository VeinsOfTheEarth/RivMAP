"""Small spatial containers shared by the georeferenced analysis workflow.

Pixel coordinates and scientific calculations keep RivMAPy's existing units.
The source grid travels with each result and supplies map coordinates at export.
Rasterio is imported only when a spatial object is constructed, so the ordinary
array API remains usable without the optional geospatial dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import LineString, Point, shape as geometry_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .geospatial import _as_affine, pixel_to_world


def _compose(left, right):
    # Affine 2.4 used multiplication; 3.x prefers matrix multiplication.
    return left @ right if hasattr(left, "__matmul__") else left * right


def _crs(value: Any):
    try:
        from rasterio.crs import CRS
    except ImportError as exc:
        raise ImportError(
            "Georeferenced workflows require RivMAPy[geospatial]. "
            'From this checkout: pip install -e ".[geospatial]"'
        ) from exc
    if value is None:
        raise ValueError("a CRS is required; unreferenced arrays use the array API")
    result = CRS.from_user_input(value)
    if not result:
        raise ValueError("a non-empty CRS is required")
    return result


@dataclass(frozen=True, eq=False)
class RasterGrid:
    """An affine pixel-corner transform, CRS, shape, and observation validity.

    ``valid_mask=None`` means every pixel is observed. Unknown pixels are kept
    separate from observed pixels whose channel value is zero.
    """

    transform: Any
    crs: Any
    shape: tuple[int, int]
    valid_mask: NDArray[np.bool_] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        affine = _as_affine(self.transform)
        if not np.all(np.isfinite(tuple(affine)[:6])) or affine.determinant == 0:
            raise ValueError("transform must be finite and invertible")
        dimensions = tuple(self.shape)
        if len(dimensions) != 2 or any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
            or x <= 0 for x in dimensions
        ):
            raise ValueError("shape must contain two positive integers")
        object.__setattr__(self, "transform", affine)
        object.__setattr__(self, "crs", _crs(self.crs))
        object.__setattr__(self, "shape", tuple(map(int, dimensions)))
        if self.valid_mask is not None:
            valid = np.asarray(self.valid_mask)
            if valid.shape != dimensions or valid.dtype != np.bool_:
                raise ValueError("valid_mask must be Boolean and match the raster shape")
            valid = np.array(valid, copy=True)
            valid.flags.writeable = False
            object.__setattr__(self, "valid_mask", valid)

    @property
    def valid(self) -> NDArray[np.bool_]:
        if self.valid_mask is not None:
            return self.valid_mask
        result = np.ones(self.shape, dtype=bool)
        result.flags.writeable = False
        return result

    def with_valid_mask(self, mask: ArrayLike) -> RasterGrid:
        return RasterGrid(self.transform, self.crs, self.shape, np.asarray(mask))

    def assert_aligned(self, other: RasterGrid) -> None:
        """Reject differing CRSs or grids, even if their arrays have equal shapes."""
        if not isinstance(other, RasterGrid):
            raise TypeError("expected a RasterGrid")
        if self.crs != other.crs:
            raise ValueError("CRS mismatch; reproject observations to one CRS first")
        if self.shape != other.shape:
            raise ValueError("raster grids must have the same shape")
        relative = _compose(~self.transform, other.transform)
        if not np.allclose(tuple(relative)[:6], (1, 0, 0, 0, 1, 0), rtol=0, atol=1e-7):
            raise ValueError("raster grid alignment differs; align observations before analysis")

    def crop(self, top: int, bottom: int, left: int, right: int) -> RasterGrid:
        values = (top, bottom, left, right)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in values):
            raise ValueError("crop bounds must be integer pixel indices")
        if not (0 <= top < bottom <= self.shape[0] and 0 <= left < right <= self.shape[1]):
            raise ValueError("crop bounds must define a non-empty window within the raster")
        from affine import Affine
        valid = None if self.valid_mask is None else self.valid_mask[top:bottom, left:right]
        return RasterGrid(
            _compose(self.transform, Affine.translation(left, top)), self.crs,
            (bottom - top, right - left), valid,
        )

    def world_coordinates(self, coordinates: ArrayLike) -> NDArray[np.float64]:
        """Map zero-based pixel centers to the source CRS (x, y axis order)."""
        return pixel_to_world(coordinates, self.transform)

    @property
    def pixel_area(self) -> float:
        """Pixel area in squared CRS units; geographic coordinates are rejected."""
        if not self.crs.is_projected:
            raise ValueError("physical area requires a projected CRS; degrees are not distances")
        return float(abs(self.transform.determinant))

    @property
    def pixel_size(self) -> float:
        """Length scale for square, orthogonal pixels in a projected CRS."""
        if not self.crs.is_projected:
            raise ValueError("physical lengths require a projected CRS")
        x = np.array([self.transform.a, self.transform.d])
        y = np.array([self.transform.b, self.transform.e])
        sx, sy = np.linalg.norm(x), np.linalg.norm(y)
        if not np.isclose(sx, sy, rtol=1e-7, atol=0) or abs(np.dot(x, y)) > 1e-7 * sx * sy:
            raise ValueError("one length scale requires square, orthogonal pixels")
        return float(sx)


@dataclass(frozen=True, eq=False)
class GeoRaster:
    """A two-dimensional raster with its source grid and optional nodata value."""

    array: NDArray[Any]
    grid: RasterGrid
    nodata: float | int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.grid, RasterGrid):
            raise TypeError("grid must be a RasterGrid")
        array = np.asarray(self.array)
        if array.shape != self.grid.shape or array.dtype.kind not in "biuf":
            raise ValueError("array must be a real numeric or Boolean raster matching its grid")
        array = np.array(array, copy=True)
        array.flags.writeable = False
        object.__setattr__(self, "array", array)
        if self.nodata is not None:
            if not np.isscalar(self.nodata) or not isinstance(self.nodata, (int, float, np.integer, np.floating)):
                raise ValueError("nodata must be a numeric scalar or None")
            missing = np.isnan(array) if np.isnan(self.nodata) else array == self.nodata
            object.__setattr__(self, "grid", self.grid.with_valid_mask(self.grid.valid & ~missing))

    @property
    def transform(self):
        return self.grid.transform

    @property
    def crs(self):
        return self.grid.crs

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        return self.grid.valid

    def binary_array(self) -> NDArray[np.bool_]:
        values = self.array[self.valid_mask]
        if not np.all(np.isfinite(values)) or not np.all((values == 0) | (values == 1)):
            raise ValueError("analysis masks must contain only zero and one at valid pixels")
        return np.asarray(self.array, dtype=bool) & self.valid_mask

    def with_array(self, array: ArrayLike, *, nodata: float | int | None = None) -> GeoRaster:
        """Attach a same-grid derived raster without inheriting a conflicting nodata value."""
        return GeoRaster(np.asarray(array), self.grid, nodata)

    def crop(self, top: int, bottom: int, left: int, right: int) -> GeoRaster:
        grid = self.grid.crop(top, bottom, left, right)
        return GeoRaster(self.array[top:bottom, left:right], grid, self.nodata)

    def to_geotiff(self, path: str | Path, *, overwrite: bool = False) -> Path:
        from .spatial_io import write_geotiff
        return write_geotiff(path, self, overwrite=overwrite)


@dataclass(frozen=True, eq=False)
class GeoVector:
    """Shapely geometries in map coordinates with CRS and scalar attributes."""

    geometries: Sequence[BaseGeometry]
    crs: Any
    properties: Sequence[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        geometries = tuple(self.geometries)
        if any(not isinstance(g, BaseGeometry) or g.is_empty for g in geometries):
            raise ValueError("geometries must be non-empty Shapely geometries")
        properties = tuple({} for _ in geometries) if self.properties is None else tuple(self.properties)
        if len(properties) != len(geometries) or any(not isinstance(p, Mapping) for p in properties):
            raise ValueError("properties must contain one mapping per geometry")
        object.__setattr__(self, "geometries", geometries)
        object.__setattr__(self, "crs", _crs(self.crs))
        object.__setattr__(self, "properties", tuple(MappingProxyType(dict(p)) for p in properties))

    def to_geopackage(
        self, path: str | Path, *, layer: str = "features", overwrite: bool = False
    ) -> Path:
        from .spatial_io import write_geopackage
        return write_geopackage(path, self, layer=layer, overwrite=overwrite)


class SpatialResultMixin:
    """Export helpers used by analysis results with an optional ``grid`` field."""

    grid: RasterGrid | None

    def _require_grid(self) -> RasterGrid:
        if self.grid is None:
            raise ValueError("this result has no georeferencing; start with read_mask or GeoRaster")
        return self.grid

    def raster(self, field: str = "mask", *, index: int | None = None) -> GeoRaster:
        grid = self._require_grid()
        array = np.asarray(getattr(self, field))
        if index is not None:
            if array.ndim != 3:
                raise ValueError("index is only used for a raster stack such as cell_masks")
            array = array[index]
        if array.shape != grid.shape:
            raise ValueError(f"{field!r} is not a two-dimensional raster on the source grid")
        return GeoRaster(array, grid)

    def world_coordinates(self, field: str = "coordinates") -> NDArray[np.float64]:
        return self._require_grid().world_coordinates(getattr(self, field))

    def line(self, field: str = "coordinates", *, properties: Mapping[str, Any] | None = None) -> GeoVector:
        coordinates = self.world_coordinates(field)
        if len(coordinates) < 2:
            raise ValueError("a line requires at least two coordinates")
        return GeoVector((LineString(coordinates),), self._require_grid().crs, (properties or {},))

    def points(
        self, field: str = "coordinates", *, properties: Sequence[Mapping[str, Any]] | None = None
    ) -> GeoVector:
        coordinates = self.world_coordinates(field)
        return GeoVector(tuple(Point(xy) for xy in coordinates), self._require_grid().crs, properties)

    def polygons(self, field: str = "cutoffs") -> GeoVector:
        """Polygonize true pixel footprints; cell stacks retain their cell index.

        The affine acts on pixel corners here, so polygonized raster bounds do
        not receive the half-pixel offset used for centerlines and banklines.
        """
        from rasterio.features import shapes
        grid = self._require_grid()
        data = np.asarray(getattr(self, field))
        if data.ndim == 2:
            data = data[None]
        if data.ndim != 3 or data.shape[1:] != grid.shape or data.dtype != np.bool_:
            raise ValueError("polygons requires a Boolean raster or a stack of Boolean rasters")
        geometries = []
        properties = []
        for index, mask in enumerate(data):
            selected = mask & grid.valid
            # Four-connectivity yields valid polygons even at diagonal contacts.
            parts = [geometry_shape(g) for g, value in shapes(
                selected.astype(np.uint8), mask=selected, transform=grid.transform, connectivity=4
            ) if value == 1]
            if not parts:
                continue
            if len(data) > 1 or field == "cell_masks":
                geometries.append(unary_union(parts))
                properties.append({"cell_index": index})
            else:
                geometries.extend(parts)
                properties.extend({"component": i} for i in range(len(parts)))
        return GeoVector(tuple(geometries), grid.crs, properties)

    def to_geotiff(
        self, path: str | Path, field: str = "mask", *, index: int | None = None, overwrite: bool = False
    ) -> Path:
        return self.raster(field, index=index).to_geotiff(path, overwrite=overwrite)


def mosaic_georasters(rasters: Sequence[GeoRaster], *, overlap: str = "last") -> GeoRaster:
    """Mosaic aligned north-up rasters while preserving CRS and valid data.

    Differently sized tiles are allowed. CRS, resolution, and grid alignment
    must match. Only valid source pixels participate in the overlap policy;
    holes retain a false validity mask independently of their stored value.
    """
    from .geospatial import mosaic_rasters
    items = list(rasters)
    if not items or any(not isinstance(item, GeoRaster) for item in items):
        raise ValueError("mosaic_georasters requires at least one GeoRaster")
    reference = items[0]
    for item in items[1:]:
        if item.crs != reference.crs:
            raise ValueError("CRS mismatch; reproject tiles to one CRS first")
        relative = _compose(~reference.transform, item.transform)
        if not np.allclose((relative.a, relative.b, relative.d, relative.e), (1, 0, 0, 1), rtol=0, atol=1e-7):
            raise ValueError("tile resolution or orientation differs")
        if not np.allclose((relative.c, relative.f), np.rint((relative.c, relative.f)), rtol=0, atol=1e-7):
            raise ValueError("tile grids have a sub-pixel alignment offset")
    layout = mosaic_rasters(
        [item.array for item in items], [item.transform for item in items],
        nodata=None, overlap=overlap, crs=reference.crs,
    )
    data = np.zeros_like(layout.array)
    valid = np.zeros(data.shape, dtype=bool)
    for item in items:
        relative = _compose(~layout.transform, item.transform)
        col, row = int(round(relative.c)), int(round(relative.f))
        rows, cols = slice(row, row + item.grid.shape[0]), slice(col, col + item.grid.shape[1])
        target, observed = data[rows, cols], valid[rows, cols]
        selected = item.valid_mask if overlap == "last" else item.valid_mask & ~observed
        target[selected] = item.array[selected]
        observed[selected] = True
    return GeoRaster(data, RasterGrid(layout.transform, reference.crs, data.shape, valid))


__all__ = ["GeoRaster", "GeoVector", "RasterGrid", "SpatialResultMixin", "mosaic_georasters"]
