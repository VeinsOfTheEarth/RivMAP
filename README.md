# RivMAPy

RivMAPy (**River Morphodynamics from Analysis of Planforms**) is a Python
port of the RivMAP MATLAB toolbox for measuring river geometry and change
from binary channel masks. It provides deterministic, array-first workflows
for centerlines, banklines, widths, migration, erosion and accretion,
cutoffs, channel belts, and spatial summaries.

This is a stable, archival-scope port: the documented workflows are tested
and demonstrated on the bundled Ucayali River data, but the project is not
intended to become a broad, actively expanding river-analysis platform.
There is no active feature roadmap or promise of ongoing maintenance.

The original MATLAB functions remain in `src/*.m`, the original walkthrough
remains in `examples/DEMO.m`, and its companion document remains in `docs/`.
Python code does not modify those files or the bundled input data.

## What is included

- Binary-mask cleanup, cropping, skeleton tracing, centerlines, and left/right
  banklines
- Pointwise normal widths and segment-average area/length widths
- Centerline direction, curvature, smoothing, resampling, and intersections
- Consecutive-mask erosion, accretion, no-change, and cutoff classification
- Swept centerline migration areas and neck-cutoff measurements
- Automatic channel-belt envelopes, centerlines, cells, and spatial summaries
- Aligned raster mosaicking, pixel-to-world conversion, and polyline stitching
- Georeferenced analysis workflows with GeoTIFF and GeoPackage exports
- A reader for the historical `data/riv.mat` annual-mask structure
- Synthetic tests plus static and 1984–2015 Ucayali demonstrations

The main public functions are exported from `rivmapy`. Result objects are
frozen dataclasses when an operation naturally returns several related
arrays. See the module docstrings in `src/rivmapy/` for parameters and result
fields.

## Requirements and installation

RivMAPy requires Python 3.10 or newer. From a repository checkout:

```console
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[examples,test]"
```

Activate the environment first if `python` does not already resolve to it.
The base installation (`python -m pip install -e .`) includes NumPy, SciPy,
scikit-image, and Shapely. Matplotlib is in the `examples` extra and pytest is
in the `test` extra. The optional `geospatial` extra adds Rasterio, Pyogrio,
and affine-transform support using standard binary wheels.
`python -m pip install -e ".[dev]"` installs everything, including dependencies
for the georeferencing tests.

## Run the demonstrations

Start with the [end-to-end walkthrough](docs/walkthrough.md) for runnable code,
figures, and interpretation: real Ucayali geometry and migration, followed by
georeferenced channel-belt analysis and GIS exports.

The examples read `data/riv.mat` without changing it and write review figures
under `examples/output/`.

```console
python examples/static_geometry.py
python examples/multiyear_change.py --maximum-records 5
python examples/multiyear_change.py
```

The first command analyzes the 1984 mask. The record limit provides a quick
temporal smoke test; omit it for the demonstrated full 32-year result. The
full-resolution temporal workflow can take a few minutes. Both scripts accept
`--help`, `--output`, and other documented command-line options.

A minimal library workflow is:

```python
import numpy as np

from rivmapy import (
    banklines_from_mask,
    centerline_from_mask,
    load_rivmap_mat,
    width_from_banklines,
)

record = load_rivmap_mat("data/riv.mat")[0]
mask = record.mask("single_thread")
centerline = centerline_from_mask(
    mask, record.exit_sides, record.nominal_width
)
banks = banklines_from_mask(mask, record.exit_sides)
width = width_from_banklines(
    centerline.coordinates,
    banks.left,
    banks.right,
    record.nominal_width,
)
print(f"mean width: {np.nanmean(width):.2f} pixels")
```

## Preserve georeferencing

Install `.[geospatial]` and start with `read_mask("channel.tif")`. Pass the
returned `GeoRaster` directly to the analysis functions; results retain the
source CRS, affine transform, and validity mask. Cropping updates the origin,
and temporal analysis checks grid alignment. Export raster fields with
`result.to_geotiff(...)` and map-coordinate vectors with
`result.line().to_geopackage(...)` or `result.polygons(...).to_geopackage(...)`.
Numerical measurements remain in pixel units until explicitly scaled.

```console
python -m pip install -e ".[geospatial]"
python examples/georeferenced_workflow.py
```

This third example uses a synthetic georeferenced test grid and verifies its
GeoTIFF/GeoPackage outputs by rereading them. It also accepts your own pair of
aligned GeoTIFF masks. See the [georeferencing guide](docs/georeferencing.md)
for a complete workflow, units, nodata handling, and supported grids.

## Coordinate and measurement conventions

- Masks are two-dimensional Boolean arrays indexed as `mask[y, x]`, or
  equivalently `[row, column]`.
- Vector coordinates have shape `(N, 2)` and are zero-based image `(x, y)`
  coordinates. `y` increases downward. Integer coordinates denote pixel
  centers in the analysis API.
- `exit_sides` contains two distinct `N`, `S`, `E`, or `W` characters. The
  upstream side is first; for example, `"SW"` flows from south to west.
- Left and right banks are named while looking downstream.
- Geometry is measured in pixel units unless the caller applies a spatial
  scale. The Ucayali temporal example uses 30 m pixels.
- In mask differencing, `erosion` means newly occupied channel pixels (land
  eroded between observations), while `accretion` means abandoned channel
  pixels (land gained).
- Affine geospatial transforms use the usual pixel-corner convention;
  `pixel_to_world(..., pixel_center=True)` applies the half-pixel offset.

These conventions differ from MATLAB's one-based indexing. Do not pass stored
MATLAB vector coordinates directly without converting them.

## Validation and legacy compatibility

Run the complete test suite from the repository root (`.[dev]` includes the
optional geospatial test dependencies):

```console
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
```

The current suite passes in full. The 1984 regression reproduces the legacy
width scale, and the mask time series reproduces approximately 102 km² more
erosion than accretion. The full Python demonstration identifies six distinct,
definite centerline-cutoff events. The legacy walkthrough's seventh event is a
small event that is spatially overprinted by a later cutoff and is not a
separate deterministic raster event in the current workflow.

See [validation targets](docs/validation.md) for numerical results and
acceptance criteria, and [legacy behavior](docs/legacy_behavior.md) for
intentional differences and corrected MATLAB defects. Scientific intent and
empirical defaults are preserved where practical; pixel-for-pixel equivalence
is not expected across different morphology and geometry implementations.

## Repository layout

```text
RivMAP/
├── src/rivmapy/       # Python package
├── src/*.m           # preserved MATLAB functions
├── examples/         # Python demonstrations and legacy DEMO.m
├── tests/            # synthetic and Ucayali regression tests
├── data/             # bundled historical MATLAB fixtures
└── docs/             # validation, compatibility, and walkthrough
```

`load_rivmap_mat` supports the MATLAB v5 structure in `data/riv.mat`. It does
not deserialize the Mapping Toolbox MCOS objects in `data/riv_georeffed.mat`.
New geospatial workflows read metadata from GeoTIFF masks or use a
`GeoRaster` with an explicitly supplied `RasterGrid`.

## Citation

If RivMAPy contributes to research, cite the software metadata in
`CITATION.cff` and the associated paper:

Schwenk, J., Khandelwal, A., Fratkin, M., Kumar, V., &
Foufoula-Georgiou, E. (2017). High spatiotemporal resolution of river
planform dynamics from Landsat: The RivMAP toolbox and results from the
Ucayali River. *Earth and Space Science*, 4(2), 46–75.
[https://doi.org/10.1002/2016EA000196](https://doi.org/10.1002/2016EA000196)

The original release is also archived on
[MATLAB Central File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/58264-rivmap-river-morphodynamics-from-analysis-of-planforms).

## License

RivMAPy is distributed under the BSD 2-Clause license in `LICENSE`. Preserve
the existing notices and in-file headers when redistributing the retained
third-party MATLAB utilities.
