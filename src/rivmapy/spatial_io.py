"""GeoTIFF and GeoPackage I/O that retains native spatial metadata.

Rasterio and Pyogrio are imported when needed; install ``RivMAPy[geospatial]``
to enable these functions. Vector I/O uses NumPy arrays and Shapely directly
and does not need GeoPandas, pandas, Fiona, or a separate GDAL installation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import shapely

from .spatial import GeoRaster, GeoVector, RasterGrid


def _rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - tested without the extra
        raise ImportError(
            "GeoTIFF I/O requires RivMAPy[geospatial]; "
            "install it with pip install 'RivMAPy[geospatial]'"
        ) from exc
    return rasterio


def _pyogrio():
    try:
        import pyogrio
        import pyogrio.raw
    except ImportError as exc:  # pragma: no cover - tested without the extra
        raise ImportError(
            "GeoPackage I/O requires RivMAPy[geospatial]; "
            "install it with pip install 'RivMAPy[geospatial]'"
        ) from exc
    return pyogrio


def read_geotiff(path: str | Path, *, band: int = 1) -> GeoRaster:
    """Read one band, its CRS and affine transform, and its valid-pixel mask.

    Both band nodata and the GDAL validity mask are honored. A missing CRS
    raises an error. Images located only through GCPs or RPCs must first be
    warped to an affine grid with a tool such as Rasterio.
    """
    rasterio = _rasterio()
    if isinstance(band, bool) or not isinstance(band, (int, np.integer)) or band < 1:
        raise ValueError("band must be a positive integer (bands are one-based)")
    with rasterio.open(path) as source:
        if band > source.count:
            raise ValueError(f"band {band} is unavailable; the raster has {source.count} bands")
        gcps, _ = source.gcps
        if source.transform.is_identity and (gcps or source.rpcs):
            raise ValueError("GCP/RPC georeferencing requires warping to an affine grid first")
        if source.crs is None:
            raise ValueError("the raster has no CRS; supply georeferencing before reading it")
        array = source.read(band)
        valid = source.read_masks(band) != 0
        nodata = source.nodatavals[band - 1]
        if nodata is not None:
            if np.isnan(nodata):
                valid &= ~np.isnan(array)
            else:
                valid &= array != nodata
        grid = RasterGrid(source.transform, source.crs, array.shape, valid_mask=valid)
    return GeoRaster(array, grid, nodata=nodata)


def read_mask(
    path: str | Path, *, band: int = 1, channel_value: int | float | None = None
) -> GeoRaster:
    """Read a binary channel mask while preserving its grid and validity.

    Without ``channel_value``, accept the standard 0/1 and 0/255 encodings.
    For a classified image, explicitly select the value representing channel.
    Invalid pixels become False but remain distinguishable from valid land
    through ``valid_mask``. The binary result has no numeric nodata sentinel.
    """
    raster = read_geotiff(path, band=band)
    valid = raster.valid_mask
    values = np.unique(raster.array[valid])
    if channel_value is None:
        if not (np.isin(values, [0, 1]).all() or np.isin(values, [0, 255]).all()):
            raise ValueError("mask values must be 0/1 or 0/255; specify channel_value for classified images")
        binary = raster.array != 0
    else:
        if not isinstance(channel_value, (int, float, np.integer, np.floating)) or not np.isfinite(channel_value):
            raise ValueError("channel_value must be a finite numeric scalar")
        binary = raster.array == channel_value
    return GeoRaster(binary & valid, raster.grid, nodata=None)


def _checked_nodata(nodata: Any, dtype: np.dtype, array: np.ndarray, valid: np.ndarray):
    if nodata is None:
        return None
    if not isinstance(nodata, (int, float, np.integer, np.floating)):
        raise ValueError("nodata must be a real numeric scalar or None")
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        if not np.isfinite(nodata) or nodata != int(nodata) or not limits.min <= nodata <= limits.max:
            raise ValueError(f"nodata {nodata!r} is not representable in {dtype}")
        encoded = int(nodata)
        if int(float(encoded)) != encoded:
            raise ValueError("integer nodata must also be exactly representable in GDAL's floating metadata")
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            encoded = np.asarray(nodata, dtype=dtype).item()
        if np.isfinite(nodata) and (not np.isfinite(encoded) or encoded != nodata):
            raise ValueError(f"nodata {nodata!r} is not exactly representable in {dtype}")
    collision = np.isnan(array) if np.isnan(encoded) else array == encoded
    if np.any(collision & valid):
        raise ValueError("nodata collides with valid pixel values; use nodata=None and the validity mask")
    return encoded


def write_geotiff(
    path: str | Path, raster: GeoRaster, *, overwrite: bool = False
) -> Path:
    """Write one GeoTIFF band with native CRS, affine transform, and mask.

    Boolean data is stored as uint8 (0/1). An internal TIFF validity mask
    preserves invalid pixels independently of zero-valued land or output.
    A numeric nodata sentinel must fit the storage dtype and cannot coincide
    with valid data. Existing files are protected unless ``overwrite=True``.
    """
    rasterio = _rasterio()
    if not isinstance(raster, GeoRaster):
        raise TypeError("raster must be a GeoRaster")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; set overwrite=True to replace it")
    array = raster.array.astype(np.uint8) if raster.array.dtype == bool else raster.array
    if array.dtype.kind not in "iuf" or not rasterio.dtypes.check_dtype(array.dtype):
        raise ValueError(f"GeoTIFF storage does not support array dtype {array.dtype}")
    valid = raster.valid_mask
    nodata = _checked_nodata(raster.nodata, array.dtype, array, valid)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            destination, "w", driver="GTiff", height=array.shape[0],
            width=array.shape[1], count=1, dtype=array.dtype,
            crs=raster.crs, transform=raster.transform, nodata=nodata,
            compress="deflate",
        ) as target:
            target.write(array, 1)
            target.write_mask(valid.astype(np.uint8) * 255)
    return destination


def _property_columns(properties):
    fields = list(dict.fromkeys(key for record in properties for key in record))
    if any(not isinstance(key, str) or not key or "\x00" in key for key in fields):
        raise ValueError("property names must be nonempty strings without NUL characters")
    if len({field.casefold() for field in fields}) != len(fields):
        raise ValueError("GeoPackage property names must be unique ignoring case")
    columns = []
    masks = []
    for field in fields:
        values = [record.get(field) for record in properties]
        present = [value for value in values if value is not None]
        null = np.asarray([value is None for value in values], dtype=bool)
        if all(isinstance(value, (bool, np.bool_)) for value in present) and present:
            column = np.asarray([False if value is None else value for value in values], dtype=bool)
        elif all(isinstance(value, (int, np.integer)) for value in present) and present:
            if any(value < -(2**63) or value >= 2**63 for value in present):
                raise ValueError(f"property {field!r} exceeds the GeoPackage signed 64-bit integer range")
            if null.any() and any(abs(value) >= 2**53 for value in present):
                raise ValueError(
                    f"nullable integer property {field!r} exceeds Pyogrio's exact read range; "
                    "store these identifiers as strings"
                )
            column = np.asarray([0 if value is None else value for value in values], dtype=np.int64)
        elif all(isinstance(value, (int, float, np.integer, np.floating)) for value in present) and present:
            if any(isinstance(value, (int, np.integer)) and abs(value) > 2**53 for value in present):
                raise ValueError(f"mixed numeric property {field!r} would lose integer precision")
            column = np.asarray([0.0 if value is None else value for value in values], dtype=np.float64)
            if not np.isfinite(column).all():
                raise ValueError(f"property {field!r} contains nonfinite values; use None for missing data")
        elif all(isinstance(value, str) for value in present):
            column = np.asarray(values, dtype=object)
        else:
            raise ValueError(f"property {field!r} must contain compatible scalar int, float, bool, str, or None values")
        columns.append(column)
        masks.append(null)
    return fields, columns, masks


def write_geopackage(
    path: str | Path, vectors: GeoVector, *, layer: str = "features", overwrite: bool = False
) -> Path:
    """Write map-coordinate geometries and scalar attributes in their native CRS.

    ``overwrite`` applies only to the named layer; other layers are retained.
    New layers may be added to an existing GeoPackage without overwriting.
    Nullable integer identifiers at or beyond +/-2**53 must be encoded as strings
    because Pyogrio's NumPy reader represents nullable integers as floats.
    """
    pyogrio = _pyogrio()
    if not isinstance(vectors, GeoVector):
        raise TypeError("vectors must be a GeoVector")
    if not isinstance(layer, str) or not layer.strip() or "\x00" in layer:
        raise ValueError("layer must be a nonempty string without NUL characters")
    destination = Path(path)
    if destination.exists():
        existing_layers = pyogrio.list_layers(destination)
        info = pyogrio.read_info(destination, layer=str(existing_layers[0, 0]) if len(existing_layers) else None)
        if info["driver"] != "GPKG":
            raise ValueError("the existing destination is not a GeoPackage")
        layers = {str(name).casefold() for name in existing_layers[:, 0]}
        if layer.casefold() in layers and not overwrite:
            raise FileExistsError(f"layer {layer!r} already exists; set overwrite=True to replace it")
    fields, columns, masks = _property_columns(vectors.properties)
    geometry_types = {geometry.geom_type for geometry in vectors.geometries}
    geometry_type = next(iter(geometry_types)) if len(geometry_types) == 1 else "Unknown"
    used = {field.casefold() for field in fields}
    fid_name, geometry_name = "_rivmapy_fid", "_rivmapy_geometry"
    while fid_name.casefold() in used:
        fid_name += "_"
    while geometry_name.casefold() in used:
        geometry_name += "_"
    pyogrio.raw.write(
        destination, shapely.to_wkb(vectors.geometries), columns, fields,
        field_mask=masks, layer=layer, driver="GPKG", geometry_type=geometry_type,
        crs=vectors.crs.to_wkt(), promote_to_multi=False,
        layer_options={"FID": fid_name, "GEOMETRY_NAME": geometry_name},
    )
    return destination


def read_geopackage(path: str | Path, *, layer: str | None = None) -> GeoVector:
    """Read native-coordinate geometries, CRS, and scalar GeoPackage attributes.

    A multi-layer file requires an explicit layer so that selection is never
    dependent on storage order. Missing CRS and null geometries are rejected.
    """
    pyogrio = _pyogrio()
    if layer is None:
        layers = pyogrio.list_layers(path)
        if len(layers) != 1:
            raise ValueError("specify layer when a GeoPackage contains multiple layers")
        layer = str(layers[0, 0])
    info = pyogrio.read_info(path, layer=layer)
    if info["driver"] != "GPKG":
        raise ValueError("the source is not a GeoPackage")
    if not info["crs"]:
        raise ValueError("the vector layer has no CRS")
    metadata, _, wkb, columns = pyogrio.raw.read(path, layer=layer)
    if wkb is None or any(geometry is None for geometry in wkb):
        raise ValueError("the vector layer must contain a geometry for every feature")
    properties = [dict() for _ in range(len(wkb))]
    for index, (name, column) in enumerate(zip(metadata["fields"], columns, strict=True)):
        dtype = np.dtype(metadata["dtypes"][index])
        is_integer = dtype.kind in "iu"
        is_boolean = dtype.kind == "b"
        if is_integer and column.dtype.kind == "f" and np.any(np.abs(column[np.isfinite(column)]) >= 2**53):
            raise ValueError(f"nullable integer property {name!r} exceeds Pyogrio's exact read range; use string identifiers")
        for record, value in zip(properties, column, strict=True):
            if value is None or isinstance(value, (float, np.floating)) and np.isnan(value):
                scalar = None
            elif is_integer:
                scalar = int(value)
            elif is_boolean:
                scalar = bool(value)
            else:
                scalar = value.item() if isinstance(value, np.generic) else value
            record[str(name)] = scalar
    return GeoVector(tuple(shapely.from_wkb(wkb)), info["crs"], tuple(properties))


__all__ = [
    "read_geotiff", "read_mask", "write_geotiff", "read_geopackage", "write_geopackage",
]
