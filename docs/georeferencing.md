# Georeferenced workflows

For a step-by-step introduction with figures, see the
[end-to-end walkthrough](walkthrough.md), including real Ucayali analysis and
a separate georeferenced example.

Start with `read_mask()` and pass the returned `GeoRaster` through the analysis
functions. Its `RasterGrid` carries the source CRS, pixel-corner affine
transform, raster shape, and observation-validity mask. Results retain that
grid and can export directly to GeoTIFF or GeoPackage.

## Installation

From the repository checkout, in your activated Python environment:

```console
python -m pip install -e ".[geospatial]"
python examples/georeferenced_workflow.py
```

Rasterio handles rasters; Pyogrio handles vector files using Shapely geometries
and NumPy arrays. These are also used in current RivGraph. Their standard
binary wheels bundle GDAL, so supported Python/platform combinations do not
need a separate GDAL installation. RivMAPy does not require GeoPandas, Fiona,
or the `osgeo` Python bindings. To require wheels and fail explicitly if your
platform lacks one, add `--only-binary=:all:` to the installation command.

The geospatial extra is optional for array-only use. Install `.[dev]` to run
the complete test suite, including raster/vector round trips.

## Read, analyze, and export

```python
from rivmapy import (
    read_mask, centerline_from_mask, banklines_from_mask,
    width_from_banklines, width_from_mask, migration_mask,
)

first = read_mask("channel_2000.tif")
second = read_mask("channel_2001.tif")
centerline = centerline_from_mask(first, "SW", nominal_width=30)
banks = banklines_from_mask(first, "SW")
widths = width_from_banklines(centerline, banks, nominal_width=30)
segments = width_from_mask(first, centerline, spacing=15)
change = migration_mask(first, second, nominal_width=30)

centerline.to_geotiff("centerline.tif")
change.to_geotiff("erosion.tif", "erosion")
centerline.line().to_geopackage("results.gpkg", layer="centerline")
banks.line("left").to_geopackage("results.gpkg", layer="left_bank")
banks.line("right").to_geopackage("results.gpkg", layer="right_bank")
change.polygons("cutoffs").to_geopackage("results.gpkg", layer="cutoffs")
```

Choose exit sides and nominal width for your reach; width and spacing arguments
are in pixels. `read_mask` accepts 0/1 and 0/255 masks. Use
`read_mask(path, channel_value=...)` for a classified raster. The source must
have a CRS and affine georeferencing. GCP/RPC-only imagery must be warped to an
affine grid before analysis.

The result's existing arrays remain accessible (`centerline.coordinates`,
`change.erosion`, etc.). For continued spatial processing, retain the result
object or use `result.raster("field")`. Extracting bare arrays removes their
connection to metadata; it does not convert coordinates to map units.

`centerline.world_coordinates()` returns map-coordinate vertices.
`widths.points(properties=...)` exports bank-width sample locations, and
`segments.points("sample_coordinates", properties=...)` exports segment
sample locations. Supply one attribute dictionary per point, with explicit
unit names such as `width_px`. Use `None` for missing attributes rather than
`NaN`. Files and existing GeoPackage layers are protected unless
`overwrite=True`; replacing one named layer preserves other layers.

## What is preserved

- Centerlines, banks, both width methods, mask/centerline migration, channel
  belts, and spatial summaries retain their source grid.
- `crop_to_mask(georaster, exits)` and `georaster.crop(...)` update the affine
  origin to match the crop. A cropped result's `raster()` can feed another
  analysis function without manual coordinate adjustments.
- Temporal inputs must share CRS, shape, resolution, and alignment. The
  package checks these and does not silently reproject or resample.
- `mosaic_georasters([...])` combines aligned north-up tiles, retaining the
  CRS, output transform, and validity. Only observed pixels participate in
  the selected `"first"` or `"last"` overlap rule. The older array-level
  `mosaic_rasters` remains available with caller-supplied transforms.
- GeoTIFF exports retain the native CRS and affine transform. Boolean outputs
  are stored as 0/1 with an internal validity mask: observed land or an observed
  zero change is distinct from nodata.
- GeoPackage vectors remain in the source CRS. Line/point coordinates use
  pixel centers; polygonized raster regions use pixel edges. This avoids a
  half-pixel shift between vector and raster outputs.

If observations differ in validity, change analyses use their common observed
footprint. Static geometry and width analyses reject unknown pixels touching
the channel, since an unobserved bank cannot support a reliable width.
Automatic mask-cutoff classification does not promote a component touching
nodata to a cutoff based on an incomplete area. Clipped belt cells retain
their original indices; empty cells have zero counts and no polygon feature.

## Coordinates versus measurement units

Georeferencing does not change the existing pixel-based scientific algorithms.
Widths, along-line distances, and nominal-width parameters are in pixels;
mask areas are pixel counts. For a projected, square-pixel grid:

```python
width_in_crs_units = widths.widths * first.grid.pixel_size
erosion_in_squared_crs_units = change.erosion.sum() * first.grid.pixel_area
```

These are metres and square metres only if the CRS uses metres. Geographic
degree grids are preserved for I/O but rejected by the physical-scale helpers.
`pixel_size` also rejects rectangular or sheared pixels: one scalar cannot
convert their pixel-based lengths correctly. Use a suitable projected,
square-pixel grid for quantitative geometry analysis. Rotated affine metadata
is preserved; geometry workflows reject reflected orientation that would
reverse map-space left/right bank labels. Tile mosaicking currently requires
north-up grids.

## Demonstration and validation

`examples/georeferenced_workflow.py` creates two synthetic observations on a
known 30 m UTM test grid with a nodata margin. It exercises static geometry,
widths, temporal change, cropping, channel belts, and spatial summaries. It
writes GeoTIFFs and a multilayer GeoPackage under
`examples/output/georeferenced/`, then rereads products to check CRS, affine
alignment, validity, map coordinates, and crop offsets. The default run
produces 10 belt cells and a 10-by-2 spatial channel-area summary.

The test coordinates are **not** georeferencing for the Ucayali MATLAB fixture.
`load_rivmap_mat` still does not deserialize the Mapping Toolbox MCOS objects
in `data/riv_georeffed.mat`; no real-world origin is invented for those arrays.

To use your own two aligned GeoTIFF masks:

```console
python examples/georeferenced_workflow.py --mask-a channel_2000.tif --mask-b channel_2001.tif --exit-sides SW --nominal-width 30 --output examples/output/my_reach
```

The demonstration uses explicitly chosen small-belt tuning parameters; adapt
them if necessary for a real reach. See [validation](validation.md) for the
regression suite and [legacy behavior](legacy_behavior.md) for scientific
conventions retained from RivMAP.

Dependency references: [RivGraph source](https://github.com/VeinsOfTheEarth/RivGraph/blob/master/rivgraph/classes.py),
[Rasterio installation](https://rasterio.readthedocs.io/en/stable/installation.html),
and [Pyogrio installation](https://pyogrio.readthedocs.io/en/latest/install.html).
