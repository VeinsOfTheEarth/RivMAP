"""Static planform measurements derived from binary channel masks.

The functions in this module are modern, deterministic implementations of
the mask-analysis portion of the original MATLAB RivMAP toolbox.  Pixel
coordinates are always returned as ``(x, y)`` pairs in a zero-based image
coordinate system: ``x`` increases to the right and ``y`` increases down.

The implementations intentionally retain the scientific intent and the
important empirical defaults of RivMAP, but they do not promise pixel-for-
pixel agreement with MATLAB's Image Processing Toolbox.  In particular,
scikit-image and MATLAB can make different choices when a skeleton has two
equally valid one-pixel paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from math import hypot, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from skimage import draw, morphology

from .geometry import smooth_polyline
from .spatial import GeoRaster, RasterGrid, SpatialResultMixin


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]

_SIDES = frozenset("NSEW")
_INWARD_VECTOR = {
    "N": np.array([0.0, 1.0]),
    "S": np.array([0.0, -1.0]),
    "E": np.array([-1.0, 0.0]),
    "W": np.array([1.0, 0.0]),
}


@dataclass(frozen=True)
class CropResult(SpatialResultMixin):
    """A cropped mask and its location in the original image.

    ``bottom`` and ``right`` are exclusive, so the original extent is
    ``mask[top:bottom, left:right]`` before boundary-hole filling.
    """

    mask: BoolArray
    top: int
    bottom: int
    left: int
    right: int
    grid: RasterGrid | None = None

    @property
    def y_top(self) -> int:
        """Alias matching the MATLAB output name."""

        return self.top

    @property
    def y_bottom(self) -> int:
        """Exclusive bottom row."""

        return self.bottom

    @property
    def x_left(self) -> int:
        """Alias matching the MATLAB output name."""

        return self.left

    @property
    def x_right(self) -> int:
        """Exclusive right column."""

        return self.right


@dataclass(frozen=True)
class CenterlineResult(SpatialResultMixin):
    """Vector and raster representations of a channel centerline."""

    coordinates: FloatArray
    mask: BoolArray
    grid: RasterGrid | None = None

    @property
    def cl(self) -> FloatArray:
        """Short alias for the centerline coordinates."""

        return self.coordinates

    @property
    def image(self) -> BoolArray:
        """Alias for the centerline raster."""

        return self.mask


@dataclass(frozen=True)
class BanklineResult(SpatialResultMixin):
    """Left and right banks, each oriented upstream to downstream."""

    left: FloatArray
    right: FloatArray
    grid: RasterGrid | None = None

    @property
    def lb(self) -> FloatArray:
        """Short alias for the left bank."""

        return self.left

    @property
    def rb(self) -> FloatArray:
        """Short alias for the right bank."""

        return self.right


@dataclass(frozen=True)
class WidthProfile(SpatialResultMixin):
    """Segment-average widths and their along-centerline positions."""

    widths: FloatArray
    distances: FloatArray
    breakpoints: FloatArray
    grid: RasterGrid | None = None
    sample_coordinates: FloatArray | None = None

    @property
    def width(self) -> FloatArray:
        """Singular alias for compatibility with tabular naming."""

        return self.widths

    @property
    def distance(self) -> FloatArray:
        """Singular alias for compatibility with tabular naming."""

        return self.distances


@dataclass(frozen=True)
class WidthSamples(SpatialResultMixin):
    """Normal-to-centerline widths in pixels at georeferenced sample points."""

    widths: FloatArray
    coordinates: FloatArray
    distances: FloatArray
    grid: RasterGrid | None = None


def _validate_spatial_geometry(mask: BoolArray, grid: RasterGrid) -> None:
    """Do not interpret missing bank observations as dry land."""

    if grid.transform.determinant > 0:
        raise ValueError(
            "geometry analysis requires a north-up-compatible transform handedness "
            "(negative determinant); reorient reflected or south-up grids first"
        )
    if grid.valid_mask is not None and np.any(
        ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)) & ~grid.valid
    ):
        raise ValueError(
            "nodata touches the channel; geometry requires fully observed banks "
            "and channel pixels, so crop to an observed reach first"
        )


def _validate_mask(mask: ArrayLike, *, name: str = "mask") -> BoolArray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype")
    if 0 in array.shape:
        raise ValueError(f"{name} must have non-zero height and width")
    return array.copy()


def _validate_exit_sides(exit_sides: str) -> str:
    if not isinstance(exit_sides, str):
        raise TypeError("exit_sides must be a two-character string")
    sides = exit_sides.upper()
    if len(sides) != 2 or any(side not in _SIDES for side in sides):
        raise ValueError("exit_sides must contain two N/S/E/W characters")
    if sides[0] == sides[1]:
        raise ValueError("exit_sides must name two distinct sides")
    return sides


def _validate_positive(value: float, *, name: str, allow_zero: bool = False) -> float:
    result = float(value)
    lower_ok = result >= 0 if allow_zero else result > 0
    if not np.isfinite(result) or not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return result


def _validate_line(
    line: ArrayLike, *, name: str, minimum_points: int = 2
) -> FloatArray:
    coordinates = np.asarray(line, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if len(coordinates) < minimum_points:
        raise ValueError(f"{name} must contain at least {minimum_points} points")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return coordinates.copy()


def _fill_boundary_holes(mask: BoolArray) -> BoolArray:
    """Fill holes, including holes that leak through one image boundary.

    RivMAP temporarily sealed each of the four boundaries in turn before
    calling ``imfill``.  Repeating the operation here preserves that useful
    behavior for multithread channels that separate at an image edge.
    """

    filled = ndi.binary_fill_holes(mask)
    pad_specs = (
        ((0, 1), (0, 0)),
        ((1, 0), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    )
    unpad = (
        (slice(None, -1), slice(None)),
        (slice(1, None), slice(None)),
        (slice(None), slice(None, -1)),
        (slice(None), slice(1, None)),
    )
    for pads, keep in zip(pad_specs, unpad, strict=True):
        sealed = np.pad(filled, pads, mode="constant", constant_values=True)
        filled = ndi.binary_fill_holes(sealed)[keep]
    return np.asarray(filled, dtype=bool)


def _largest_component(mask: BoolArray) -> BoolArray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        raise ValueError("mask contains no foreground pixels")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def crop_to_mask(mask: ArrayLike | GeoRaster, exit_sides: str) -> CropResult:
    """Crop a mask to its declared channel exits and fill channel holes.

    Only dimensions corresponding to ``exit_sides`` are cropped, matching
    RivMAP's behavior.  For example, ``"WE"`` crops the left and right
    edges to the foreground extent while retaining the original row extent.

    Parameters
    ----------
    mask
        Two-dimensional boolean channel mask.
    exit_sides
        Two distinct N/S/E/W characters, with the upstream side first.

    A ``GeoRaster`` input retains its CRS and validity mask, with the affine
    transform shifted to the returned crop's upper-left pixel corner.
    """

    if isinstance(mask, GeoRaster):
        result = crop_to_mask(mask.binary_array(), exit_sides)
        grid = mask.grid.crop(result.top, result.bottom, result.left, result.right)
        return replace(result, mask=result.mask & grid.valid, grid=grid)

    image = _validate_mask(mask)
    sides = _validate_exit_sides(exit_sides)
    rows, columns = np.nonzero(image)
    if len(rows) == 0:
        raise ValueError("mask contains no foreground pixels")

    height, width = image.shape
    top = int(rows.min()) if "N" in sides else 0
    bottom = int(rows.max()) + 1 if "S" in sides else height
    left = int(columns.min()) if "W" in sides else 0
    right = int(columns.max()) + 1 if "E" in sides else width
    cropped = _fill_boundary_holes(image[top:bottom, left:right])
    return CropResult(cropped, top, bottom, left, right)


# Neighbor order starts with the four cardinal directions, mirroring the
# deterministic trace order used by the MATLAB implementation.
_NEIGHBORS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 1),
    (1, 1),
    (1, -1),
    (-1, -1),
)


def _pixel_graph(
    pixels: BoolArray,
    *,
    suppress_corner_diagonals: bool = True,
) -> tuple[NDArray[np.int64], list[list[tuple[int, float]]]]:
    nodes = np.argwhere(pixels).astype(np.int64, copy=False)
    if not len(nodes):
        raise ValueError("pixel path contains no foreground pixels")
    lookup = {(int(y), int(x)): index for index, (y, x) in enumerate(nodes)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(nodes))]

    for index, (y_raw, x_raw) in enumerate(nodes):
        y, x = int(y_raw), int(x_raw)
        for dy, dx in _NEIGHBORS:
            neighbor = lookup.get((y + dy, x + dx))
            if neighbor is None:
                continue
            if suppress_corner_diagonals and dy and dx:
                # Avoid the artificial diagonal shortcut in a three-pixel
                # right-angle corner, while retaining a true diagonal path.
                if (y, x + dx) in lookup or (y + dy, x) in lookup:
                    continue
            adjacency[index].append((neighbor, sqrt(2.0) if dy and dx else 1.0))
    return nodes, adjacency


def _graph_is_connected(adjacency: list[list[tuple[int, float]]]) -> bool:
    seen = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor, _ in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(adjacency)


def _connected_pixel_graph(
    pixels: BoolArray,
) -> tuple[NDArray[np.int64], list[list[tuple[int, float]]]]:
    nodes, adjacency = _pixel_graph(pixels, suppress_corner_diagonals=True)
    if _graph_is_connected(adjacency):
        return nodes, adjacency
    # Some legitimate skeletons contain a diagonal beside an orthogonal
    # pixel.  Falling back to the full 8-neighbor graph keeps them usable.
    nodes, adjacency = _pixel_graph(pixels, suppress_corner_diagonals=False)
    if not _graph_is_connected(adjacency):
        raise ValueError("pixel path must be a single 8-connected component")
    return nodes, adjacency


def _shortest_path(
    adjacency: list[list[tuple[int, float]]], source: int, target: int
) -> list[int]:
    distances = np.full(len(adjacency), np.inf)
    previous = np.full(len(adjacency), -1, dtype=np.int64)
    distances[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        distance, current = heappop(queue)
        if distance > distances[current]:
            continue
        if current == target:
            break
        for neighbor, weight in adjacency[current]:
            candidate = distance + weight
            if candidate + 1e-12 < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = current
                heappush(queue, (candidate, neighbor))
    if not np.isfinite(distances[target]):
        raise ValueError("no connected path exists between the channel exits")

    path = [target]
    while path[-1] != source:
        path.append(int(previous[path[-1]]))
    path.reverse()
    return path


def _distance_to_side(x: float, y: float, side: str, shape: tuple[int, int]) -> float:
    height, width = shape
    return {
        "N": y,
        "S": height - 1 - y,
        "E": width - 1 - x,
        "W": x,
    }[side]


def _orient_upstream(
    coordinates: FloatArray, exit_sides: str, shape: tuple[int, int]
) -> FloatArray:
    upstream, downstream = exit_sides

    def key(point: FloatArray) -> tuple[float, float, float, float]:
        x, y = float(point[0]), float(point[1])
        return (
            _distance_to_side(x, y, upstream, shape),
            -_distance_to_side(x, y, downstream, shape),
            y,
            x,
        )

    if key(coordinates[-1]) < key(coordinates[0]):
        return coordinates[::-1].copy()
    return coordinates.copy()


def skeleton_coords(
    skeleton: ArrayLike,
    exit_sides: str,
    *,
    start: tuple[int, int] | None = None,
) -> FloatArray:
    """Trace a branchless, 8-connected skeleton into ordered coordinates.

    Parameters
    ----------
    skeleton
        Two-dimensional boolean image containing a one-pixel path.
    exit_sides
        Upstream and downstream image sides.
    start
        Optional zero-based ``(x, y)`` endpoint.  Orientation is still
        normalized so the returned first point is upstream.

    Raises
    ------
    ValueError
        If the skeleton is disconnected, branched, cyclic, or too short.
    """

    image = _validate_mask(skeleton, name="skeleton")
    sides = _validate_exit_sides(exit_sides)
    nodes, adjacency = _connected_pixel_graph(image)
    if len(nodes) < 2:
        raise ValueError("skeleton must contain at least two pixels")

    degrees = np.fromiter((len(edges) for edges in adjacency), dtype=int)
    if np.any(degrees > 2):
        raise ValueError("skeleton must be branchless")
    endpoints = np.flatnonzero(degrees == 1)
    if len(endpoints) != 2:
        raise ValueError("skeleton must have exactly two endpoints")

    if start is not None:
        if len(start) != 2:
            raise ValueError("start must be an (x, y) pair")
        x, y = int(start[0]), int(start[1])
        matches = np.flatnonzero((nodes[:, 0] == y) & (nodes[:, 1] == x))
        if not len(matches) or int(matches[0]) not in endpoints:
            raise ValueError("start must identify a skeleton endpoint")
        source = int(matches[0])
    else:
        first, second = (int(index) for index in endpoints)
        first_xy = nodes[first, ::-1]
        second_xy = nodes[second, ::-1]
        source = first
        if _distance_to_side(*second_xy, sides[0], image.shape) < _distance_to_side(
            *first_xy, sides[0], image.shape
        ):
            source = second
    target = int(endpoints[0] if source == int(endpoints[1]) else endpoints[1])
    path = _shortest_path(adjacency, source, target)
    coordinates = nodes[path][:, ::-1].astype(float)
    return _orient_upstream(coordinates, sides, image.shape)


def _side_contact_center(mask: BoolArray, side: str) -> FloatArray:
    height, width = mask.shape
    if side == "N":
        contacts = np.flatnonzero(mask[0])
        if not len(contacts):
            raise ValueError("channel does not intersect its declared north exit")
        return np.array([float(np.median(contacts)), 0.0])
    if side == "S":
        contacts = np.flatnonzero(mask[-1])
        if not len(contacts):
            raise ValueError("channel does not intersect its declared south exit")
        return np.array([float(np.median(contacts)), float(height - 1)])
    if side == "W":
        contacts = np.flatnonzero(mask[:, 0])
        if not len(contacts):
            raise ValueError("channel does not intersect its declared west exit")
        return np.array([0.0, float(np.median(contacts))])
    contacts = np.flatnonzero(mask[:, -1])
    if not len(contacts):
        raise ValueError("channel does not intersect its declared east exit")
    return np.array([float(width - 1), float(np.median(contacts))])


def _anchor_for_side(
    nodes_yx: NDArray[np.int64],
    side: str,
    shape: tuple[int, int],
    contact_center: FloatArray,
    *,
    exclude: frozenset[int] = frozenset(),
) -> int:
    def key(index: int) -> tuple[float, float, int, int]:
        y, x = (int(value) for value in nodes_yx[index])
        tangent_distance = (
            abs(x - contact_center[0])
            if side in "NS"
            else abs(y - contact_center[1])
        )
        return (_distance_to_side(x, y, side, shape), tangent_distance, y, x)

    candidates = (index for index in range(len(nodes_yx)) if index not in exclude)
    try:
        return min(candidates, key=key)
    except ValueError as error:
        raise ValueError("could not identify distinct channel exits") from error


def centerline_from_mask(
    mask: ArrayLike | GeoRaster,
    exit_sides: str,
    nominal_width: float,
    *,
    min_mask_pixels: int = 100,
    min_centerline_length_factor: float = 5.0,
) -> CenterlineResult:
    """Extract the principal upstream-to-downstream channel centerline.

    The mask is hole-filled and reduced to its largest 8-connected
    component.  A skeleton is then computed and the minimum-cost path
    between the two declared exits is retained, which removes side spurs
    without an iterative pruning heuristic.

    ``min_mask_pixels=100`` and ``min_centerline_length_factor=5`` preserve
    the two principal size thresholds used by RivMAP.

    ``GeoRaster`` inputs preserve the source grid. Missing observations may
    not touch the channel; an incomplete reach must be cropped first.
    Coordinates and width parameters remain in pixels.
    """

    if isinstance(mask, GeoRaster):
        _validate_spatial_geometry(mask.binary_array(), mask.grid)
        result = centerline_from_mask(
            mask.binary_array(), exit_sides, nominal_width,
            min_mask_pixels=min_mask_pixels,
            min_centerline_length_factor=min_centerline_length_factor,
        )
        if np.any(result.mask & ~mask.valid_mask):
            raise ValueError("centerline crosses nodata; use a fully observed channel reach")
        return replace(result, grid=mask.grid)

    image = _validate_mask(mask)
    sides = _validate_exit_sides(exit_sides)
    width = _validate_positive(nominal_width, name="nominal_width")
    if not isinstance(min_mask_pixels, (int, np.integer)) or min_mask_pixels < 1:
        raise ValueError("min_mask_pixels must be a positive integer")
    length_factor = _validate_positive(
        min_centerline_length_factor,
        name="min_centerline_length_factor",
        allow_zero=True,
    )
    if int(image.sum()) < int(min_mask_pixels):
        raise ValueError(
            f"mask must contain at least {int(min_mask_pixels)} foreground pixels"
        )

    cleaned = _largest_component(ndi.binary_fill_holes(image))
    cropped = crop_to_mask(cleaned, sides)
    channel = _largest_component(cropped.mask)
    skeleton = morphology.skeletonize(channel).astype(bool, copy=False)
    nodes, adjacency = _connected_pixel_graph(skeleton)

    upstream_center = _side_contact_center(channel, sides[0])
    downstream_center = _side_contact_center(channel, sides[1])
    source = _anchor_for_side(nodes, sides[0], channel.shape, upstream_center)
    target = _anchor_for_side(
        nodes,
        sides[1],
        channel.shape,
        downstream_center,
        exclude=frozenset((source,)),
    )
    path = _shortest_path(adjacency, source, target)
    local_coordinates = nodes[path][:, ::-1].astype(float)
    local_coordinates = _orient_upstream(local_coordinates, sides, channel.shape)
    if len(local_coordinates) < 2:
        raise ValueError("centerline path is too short")
    path_length = float(
        np.linalg.norm(np.diff(local_coordinates, axis=0), axis=1).sum()
    )
    if path_length < width * length_factor:
        raise ValueError(
            "centerline is shorter than "
            f"nominal_width * min_centerline_length_factor ({width * length_factor:g})"
        )

    coordinates = local_coordinates + np.array([cropped.left, cropped.top])
    centerline_mask = np.zeros_like(image)
    integer_coordinates = np.rint(coordinates).astype(int)
    centerline_mask[integer_coordinates[:, 1], integer_coordinates[:, 0]] = True
    return CenterlineResult(coordinates, centerline_mask)


def _remove_exit_caps(boundary: BoolArray, exit_sides: str) -> BoolArray:
    result = boundary.copy()
    for side in exit_sides:
        if side == "N":
            result[0, :] = False
        elif side == "S":
            result[-1, :] = False
        elif side == "W":
            result[:, 0] = False
        else:
            result[:, -1] = False
    return result


def _component_bank_path(
    component: BoolArray,
    channel: BoolArray,
    exit_sides: str,
) -> FloatArray:
    nodes, adjacency = _connected_pixel_graph(component)
    source = _anchor_for_side(
        nodes,
        exit_sides[0],
        component.shape,
        _side_contact_center(channel, exit_sides[0]),
    )
    target = _anchor_for_side(
        nodes,
        exit_sides[1],
        component.shape,
        _side_contact_center(channel, exit_sides[1]),
        exclude=frozenset((source,)),
    )
    coordinates = nodes[_shortest_path(adjacency, source, target)][
        :, ::-1
    ].astype(float)
    return _orient_upstream(coordinates, exit_sides, component.shape)


def banklines_from_mask(
    mask: ArrayLike | GeoRaster,
    exit_sides: str,
    *,
    min_mask_pixels: int = 100,
    min_bank_component_pixels: int = 20,
) -> BanklineResult:
    """Extract left and right banks from a binary channel mask.

    Left and right are defined while looking downstream.  The filled outer
    perimeter is opened at the two channel exits, leaving two bank paths;
    the two best-connected paths are retained and ordered deterministically.
    The 20-pixel component threshold is the legacy RivMAP default.

    ``GeoRaster`` inputs preserve the source grid and require fully observed
    banks. Image left/right corresponds to map left/right only for transforms
    with negative determinant; reflected grids are rejected.
    """

    if isinstance(mask, GeoRaster):
        _validate_spatial_geometry(mask.binary_array(), mask.grid)
        result = banklines_from_mask(
            mask.binary_array(), exit_sides,
            min_mask_pixels=min_mask_pixels,
            min_bank_component_pixels=min_bank_component_pixels,
        )
        return replace(result, grid=mask.grid)

    image = _validate_mask(mask)
    sides = _validate_exit_sides(exit_sides)
    for value, name in (
        (min_mask_pixels, "min_mask_pixels"),
        (min_bank_component_pixels, "min_bank_component_pixels"),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if int(image.sum()) < int(min_mask_pixels):
        raise ValueError(
            f"mask must contain at least {int(min_mask_pixels)} foreground pixels"
        )

    cleaned = _largest_component(ndi.binary_fill_holes(image))
    cropped = crop_to_mask(cleaned, sides)
    channel = _largest_component(cropped.mask)
    eroded = ndi.binary_erosion(
        channel,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = _remove_exit_caps(channel & ~eroded, sides)
    labels, count = ndi.label(boundary, structure=np.ones((3, 3), dtype=bool))

    candidates: list[tuple[tuple[float, int, int], BoolArray]] = []
    for label_index in range(1, count + 1):
        component = labels == label_index
        size = int(component.sum())
        if size < int(min_bank_component_pixels):
            continue
        rows, columns = np.nonzero(component)
        upstream_distance = min(
            _distance_to_side(float(x), float(y), sides[0], channel.shape)
            for y, x in zip(rows, columns, strict=True)
        )
        downstream_distance = min(
            _distance_to_side(float(x), float(y), sides[1], channel.shape)
            for y, x in zip(rows, columns, strict=True)
        )
        candidates.append(
            ((upstream_distance + downstream_distance, -size, label_index), component)
        )
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < 2:
        raise ValueError("could not separate the channel perimeter into two banks")

    banks = [
        _component_bank_path(component, channel, sides)
        for _, component in candidates[:2]
    ]
    contact_center = _side_contact_center(channel, sides[0])
    inward = _INWARD_VECTOR[sides[0]]

    def left_score(bank: FloatArray) -> tuple[float, float, float]:
        sample = bank[: min(5, len(bank))].mean(axis=0) - contact_center
        # Image y increases downward, so left-of-heading has a negative
        # image-coordinate cross product.
        cross = inward[0] * sample[1] - inward[1] * sample[0]
        return (float(cross), float(sample[1]), float(sample[0]))

    banks.sort(key=left_score)
    offset = np.array([cropped.left, cropped.top], dtype=float)
    return BanklineResult(banks[0] + offset, banks[1] + offset)


def _smooth_coordinates(coordinates: FloatArray, window: float) -> FloatArray:
    if window <= 1 or len(coordinates) <= 2:
        return coordinates.copy()
    return smooth_polyline(coordinates, window)


def width_from_banklines(
    centerline: ArrayLike | CenterlineResult,
    left_bank: ArrayLike | BanklineResult,
    right_bank: ArrayLike | None = None,
    nominal_width: float | None = None,
    *,
    smoothing_window: float | None = None,
    search_radius_factor: float = 3.0,
    include_endpoints: bool = False,
    grid: RasterGrid | None = None,
) -> FloatArray | WidthSamples:
    """Measure width normal to a centerline by intersecting both banks.

    The default normal search radius is three nominal widths, matching
    RivMAP.  If several intersections occur on one bank, the nearest one is
    used.  Values are ``NaN`` where either bank is not intersected.

    For referenced results, use ``width_from_banklines(centerline, banks,
    nominal_width=...)``. This returns :class:`WidthSamples` with pixel widths,
    pixel sample coordinates, and the aligned source grid. The four-argument
    array form continues to return a NumPy width vector. ``grid`` may attach
    known georeferencing to explicit pixel-coordinate arrays.
    """

    source_grid = grid
    if isinstance(centerline, CenterlineResult):
        if source_grid is not None and centerline.grid is not None:
            source_grid.assert_aligned(centerline.grid)
            source_grid = source_grid.with_valid_mask(
                source_grid.valid & centerline.grid.valid
            )
        source_grid = centerline.grid if source_grid is None else source_grid
        centerline = centerline.coordinates
    if isinstance(left_bank, BanklineResult):
        if right_bank is not None:
            raise ValueError("omit right_bank when passing a BanklineResult")
        if (source_grid is None) != (left_bank.grid is None):
            raise ValueError("centerline and banks must both carry georeferencing")
        if source_grid is not None:
            source_grid.assert_aligned(left_bank.grid)
            source_grid = source_grid.with_valid_mask(
                source_grid.valid & left_bank.grid.valid
            )
        right_bank = left_bank.right
        left_bank = left_bank.left
    if nominal_width is None:
        raise ValueError("nominal_width is required")
    if source_grid is not None:
        center = _validate_line(centerline, name="centerline", minimum_points=3)
        if source_grid.transform.determinant > 0:
            raise ValueError("geometry analysis requires north-up-compatible transform handedness")
        # Input banks can come from another observed epoch.  A normal width
        # requires the entire intervening channel corridor to be observed.
        left = _validate_line(left_bank, name="left_bank")
        right = _validate_line(right_bank, name="right_bank")
        corridor = _rasterize_polygon(
            Polygon(np.vstack((left, right[::-1]))).buffer(0),
            source_grid.shape,
        )
        _validate_spatial_geometry(corridor, source_grid)
        widths = width_from_banklines(
            center, left_bank, right_bank, nominal_width,
            smoothing_window=smoothing_window,
            search_radius_factor=search_radius_factor,
            include_endpoints=include_endpoints,
        )
        pixels = np.rint(center).astype(int)
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < source_grid.shape[1])
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < source_grid.shape[0]))
        observed = np.zeros(len(center), dtype=bool)
        observed[inside] = source_grid.valid[pixels[inside, 1], pixels[inside, 0]]
        widths[~observed] = np.nan
        return WidthSamples(widths, center, _cumulative_distance(center), source_grid)

    center = _validate_line(centerline, name="centerline", minimum_points=3)
    left = _validate_line(left_bank, name="left_bank")
    right = _validate_line(right_bank, name="right_bank")
    width = _validate_positive(nominal_width, name="nominal_width")
    radius_factor = _validate_positive(
        search_radius_factor, name="search_radius_factor"
    )
    window = width if smoothing_window is None else _validate_positive(
        smoothing_window, name="smoothing_window", allow_zero=True
    )
    smoothed = _smooth_coordinates(center, window)
    left_geometry = LineString(left)
    right_geometry = LineString(right)
    results = np.full(len(center), np.nan, dtype=float)
    start = 0 if include_endpoints else 1
    stop = len(center) if include_endpoints else len(center) - 1
    radius = width * radius_factor

    for index in range(start, stop):
        if index == 0:
            tangent = smoothed[1] - smoothed[0]
        elif index == len(smoothed) - 1:
            tangent = smoothed[-1] - smoothed[-2]
        else:
            tangent = smoothed[index + 1] - smoothed[index - 1]
        tangent_norm = hypot(float(tangent[0]), float(tangent[1]))
        if tangent_norm <= np.finfo(float).eps:
            continue
        tangent /= tangent_norm
        normal = np.array([-tangent[1], tangent[0]])
        point = Point(center[index])
        cross_section = LineString(
            (center[index] - normal * radius, center[index] + normal * radius)
        )
        left_intersection = cross_section.intersection(left_geometry)
        right_intersection = cross_section.intersection(right_geometry)
        if left_intersection.is_empty or right_intersection.is_empty:
            continue
        results[index] = point.distance(left_intersection) + point.distance(
            right_intersection
        )
    return results


def _cumulative_distance(coordinates: FloatArray) -> FloatArray:
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1)))
    )


def _point_at_distance(
    coordinates: FloatArray, cumulative: FloatArray, distance: float
) -> FloatArray:
    if distance <= 0:
        return coordinates[0].copy()
    if distance >= cumulative[-1]:
        return coordinates[-1].copy()
    upper = int(np.searchsorted(cumulative, distance, side="right"))
    lower = upper - 1
    segment_length = cumulative[upper] - cumulative[lower]
    if segment_length <= np.finfo(float).eps:
        return coordinates[lower].copy()
    fraction = (distance - cumulative[lower]) / segment_length
    return coordinates[lower] + fraction * (coordinates[upper] - coordinates[lower])


def _subline(
    coordinates: FloatArray,
    cumulative: FloatArray,
    start: float,
    stop: float,
) -> FloatArray:
    interior = coordinates[(cumulative > start) & (cumulative < stop)]
    points = np.vstack(
        (
            _point_at_distance(coordinates, cumulative, start),
            interior,
            _point_at_distance(coordinates, cumulative, stop),
        )
    )
    if len(points) > 1:
        keep = np.concatenate(
            ([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-12)
        )
        points = points[keep]
    if len(points) < 2:
        raise ValueError("centerline segment collapsed to a single point")
    return points


def _polygon_parts(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type == "Polygon":
        return (geometry,)
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        parts: list[BaseGeometry] = []
        for child in geometry.geoms:
            parts.extend(_polygon_parts(child))
        return tuple(parts)
    return ()


def _rasterize_polygon(geometry: BaseGeometry, shape: tuple[int, int]) -> BoolArray:
    raster = np.zeros(shape, dtype=bool)
    for polygon in _polygon_parts(geometry):
        exterior = np.asarray(polygon.exterior.coords)
        rows, columns = draw.polygon(exterior[:, 1], exterior[:, 0], shape=shape)
        raster[rows, columns] = True
        for interior_ring in polygon.interiors:
            interior = np.asarray(interior_ring.coords)
            rows, columns = draw.polygon(interior[:, 1], interior[:, 0], shape=shape)
            raster[rows, columns] = False
    return raster


def _breakpoint_distances(spacing: float | ArrayLike, total: float) -> FloatArray:
    if np.isscalar(spacing):
        interval = _validate_positive(float(spacing), name="spacing")
        breakpoints = np.arange(0.0, total, interval, dtype=float)
        if not len(breakpoints) or not np.isclose(breakpoints[-1], total):
            breakpoints = np.append(breakpoints, total)
        else:
            breakpoints[-1] = total
        return breakpoints

    breakpoints = np.asarray(spacing, dtype=float)
    if breakpoints.ndim != 1 or len(breakpoints) < 2:
        raise ValueError("spacing breakpoints must be a one-dimensional vector")
    if not np.all(np.isfinite(breakpoints)):
        raise ValueError("spacing breakpoints must be finite")
    if np.any(np.diff(breakpoints) <= 0):
        raise ValueError("spacing breakpoints must be strictly increasing")
    tolerance = max(1.0, total) * 1e-12
    if breakpoints[0] < -tolerance or breakpoints[-1] > total + tolerance:
        raise ValueError("spacing breakpoints must lie on the centerline")
    result = breakpoints.copy()
    result[np.isclose(result, 0.0)] = 0.0
    result[np.isclose(result, total)] = total
    return result


def width_from_mask(
    mask: ArrayLike | GeoRaster,
    centerline: ArrayLike | CenterlineResult,
    spacing: float | ArrayLike,
    *,
    buffer_half_width_factor: float = 2.0,
    smoothing_width_factor: float = 2.0,
    centroid_filter_factor: float = 1.5,
) -> WidthProfile:
    """Compute segment-average width as channel area / centerline length.

    Scalar ``spacing`` and vector breakpoints are interpreted as physical
    along-centerline distances, correcting the MATLAB implementation's
    accidental use of node indices.  Segment polygons extend two reach-
    average widths on each side by default, matching the legacy buffer.
    Disconnected mask pieces are filtered using the legacy midpoint rule.

    With ``GeoRaster`` and ``CenterlineResult`` inputs the source grids must
    align; output ``grid`` and ``sample_coordinates`` locate the width samples
    without changing the pixel units used for the calculations. Missing data
    adjoining the channel are rejected because they would bias width low.
    """

    if isinstance(mask, GeoRaster):
        source_grid = mask.grid
        if isinstance(centerline, CenterlineResult):
            if centerline.grid is None:
                raise ValueError("centerline must carry georeferencing with a GeoRaster mask")
            source_grid.assert_aligned(centerline.grid)
            source_grid = source_grid.with_valid_mask(source_grid.valid & centerline.grid.valid)
            centerline = centerline.coordinates
        _validate_spatial_geometry(mask.binary_array() & source_grid.valid, source_grid)
        coordinates = _validate_line(centerline, name="centerline")
        result = width_from_mask(
            mask.binary_array() & source_grid.valid, coordinates, spacing,
            buffer_half_width_factor=buffer_half_width_factor,
            smoothing_width_factor=smoothing_width_factor,
            centroid_filter_factor=centroid_filter_factor,
        )
        cumulative = _cumulative_distance(coordinates)
        samples = np.array([
            _point_at_distance(coordinates, cumulative, distance)
            for distance in result.distances
        ])
        return replace(result, grid=source_grid, sample_coordinates=samples)
    if isinstance(centerline, CenterlineResult):
        if centerline.grid is not None:
            raise ValueError("a georeferenced centerline requires a GeoRaster mask")
        centerline = centerline.coordinates

    image = _validate_mask(mask)
    center = _validate_line(centerline, name="centerline")
    half_width_factor = _validate_positive(
        buffer_half_width_factor, name="buffer_half_width_factor"
    )
    smooth_factor = _validate_positive(
        smoothing_width_factor,
        name="smoothing_width_factor",
        allow_zero=True,
    )
    component_factor = _validate_positive(
        centroid_filter_factor, name="centroid_filter_factor"
    )
    cumulative = _cumulative_distance(center)
    total_length = float(cumulative[-1])
    if total_length <= np.finfo(float).eps:
        raise ValueError("centerline must have non-zero length")
    breakpoints = _breakpoint_distances(spacing, total_length)
    segment_count = len(breakpoints) - 1
    reach_average_width = float(image.sum()) / total_length
    smoothed = _smooth_coordinates(center, reach_average_width * smooth_factor)
    smoothed_cumulative = _cumulative_distance(smoothed)
    # Smoothing changes total length slightly.  Parameterize the smoothed
    # line by the original normalized arclength so public breakpoints retain
    # their stated distances.
    normalized_breakpoints = breakpoints / total_length
    smooth_breakpoints = normalized_breakpoints * smoothed_cumulative[-1]

    widths = np.full(segment_count, np.nan, dtype=float)
    distances = (breakpoints[:-1] + breakpoints[1:]) / 2.0
    centroid_threshold = total_length / len(breakpoints) / component_factor
    buffer_half_width = reach_average_width * half_width_factor

    for index, (start, stop) in enumerate(
        zip(smooth_breakpoints[:-1], smooth_breakpoints[1:], strict=True)
    ):
        physical_length = breakpoints[index + 1] - breakpoints[index]
        reference_segment = _subline(
            center,
            cumulative,
            float(breakpoints[index]),
            float(breakpoints[index + 1]),
        )
        segment = _subline(smoothed, smoothed_cumulative, float(start), float(stop))
        polygon = LineString(segment).buffer(
            buffer_half_width,
            cap_style=2,
            join_style=2,
        )
        candidate = image & _rasterize_polygon(polygon, image.shape)
        if float(candidate.sum()) < physical_length:
            # A moving average can cut across the inside of an exceptionally
            # tight bend.  If that leaves less than one pixel of channel area
            # per pixel of centreline, retry the same physical interval on
            # the unsmoothed (and mask-contained) centreline.  This local
            # fallback keeps smoothing for ordinary segments while avoiding
            # near-zero widths caused solely by geometric overshoot.
            raw_polygon = LineString(reference_segment).buffer(
                buffer_half_width,
                cap_style=2,
                join_style=2,
            )
            raw_candidate = image & _rasterize_polygon(raw_polygon, image.shape)
            if raw_candidate.sum() > candidate.sum():
                candidate = raw_candidate
        labels, count = ndi.label(candidate, structure=np.ones((3, 3), dtype=bool))
        if count > 1:
            reference_pixels = np.rint(reference_segment).astype(int)
            reference_pixels[:, 0] = np.clip(reference_pixels[:, 0], 0, image.shape[1] - 1)
            reference_pixels[:, 1] = np.clip(reference_pixels[:, 1], 0, image.shape[0] - 1)
            touched_labels = np.unique(
                labels[reference_pixels[:, 1], reference_pixels[:, 0]]
            )
            touched_labels = touched_labels[touched_labels > 0]
            if len(touched_labels):
                # The component crossed by this exact centreline interval is
                # the intended local channel piece, even when an adjacent
                # hairpin occupies more of the broad buffer polygon.
                retained = np.isin(labels, touched_labels)
            else:
                retained = np.zeros_like(candidate)
                midpoint = distances[index]
                component_offsets: list[tuple[float, int]] = []
                for label_index in range(1, count + 1):
                    rows, columns = np.nonzero(labels == label_index)
                    if not len(rows):
                        continue
                    centroid = np.array([columns.mean(), rows.mean()])
                    nearest = int(np.argmin(np.linalg.norm(center - centroid, axis=1)))
                    offset = float(abs(cumulative[nearest] - midpoint))
                    component_offsets.append((offset, label_index))
                    if offset < centroid_threshold:
                        retained[labels == label_index] = True
                if not np.any(retained) and component_offsets:
                    _, nearest_label = min(component_offsets)
                    retained[labels == nearest_label] = True
            candidate = retained
        if float(candidate.sum()) < physical_length:
            # The smoothed buffer may contain plenty of distant hairpin area
            # while intersecting the intended local component in only a pixel
            # or two.  Re-evaluate component selection on the exact segment,
            # not just the total buffered area.
            raw_polygon = LineString(reference_segment).buffer(
                buffer_half_width,
                cap_style=2,
                join_style=2,
            )
            raw_candidate = image & _rasterize_polygon(raw_polygon, image.shape)
            raw_labels, raw_count = ndi.label(
                raw_candidate, structure=np.ones((3, 3), dtype=bool)
            )
            if raw_count > 1:
                reference_pixels = np.rint(reference_segment).astype(int)
                reference_pixels[:, 0] = np.clip(
                    reference_pixels[:, 0], 0, image.shape[1] - 1
                )
                reference_pixels[:, 1] = np.clip(
                    reference_pixels[:, 1], 0, image.shape[0] - 1
                )
                local_labels = np.unique(
                    raw_labels[reference_pixels[:, 1], reference_pixels[:, 0]]
                )
                local_labels = local_labels[local_labels > 0]
                if len(local_labels):
                    raw_candidate = np.isin(raw_labels, local_labels)
            if raw_candidate.sum() > candidate.sum():
                candidate = raw_candidate
        widths[index] = float(candidate.sum()) / physical_length

    return WidthProfile(widths, distances, breakpoints)


__all__ = [
    "BanklineResult",
    "CenterlineResult",
    "CropResult",
    "WidthProfile",
    "WidthSamples",
    "banklines_from_mask",
    "centerline_from_mask",
    "crop_to_mask",
    "skeleton_coords",
    "width_from_banklines",
    "width_from_mask",
]
