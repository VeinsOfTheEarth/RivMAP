# Legacy behavior and compatibility

RivMAPy preserves RivMAP's scientific intent and the empirical defaults used
in the released MATLAB code. It is not a line-by-line translation and does
not promise identical pixels, vertex counts, or floating-point values.
scikit-image, SciPy, and Shapely use different morphology, rasterization, and
intersection implementations from MATLAB R2015a and its toolboxes.

The historical MATLAB functions remain in `src/*.m`; `examples/DEMO.m` and
`docs/RivMAP Demo Walkthrough.docx` remain as provenance and interpretation
records. They are not imported or modified by RivMAPy.

## Public correspondence

| MATLAB operation | RivMAPy operation | Compatibility note |
| --- | --- | --- |
| `crop_to_mask` | `crop_to_mask` | Returns a `CropResult` with the crop offsets. |
| `skeleton_coords` | `skeleton_coords` | Rejects branches, cycles, and disconnected paths explicitly. |
| `centerline_from_mask` | `centerline_from_mask` | Returns a `CenterlineResult`; uses a deterministic principal skeleton path. |
| `banklines_from_mask` | `banklines_from_mask` | Returns a `BanklineResult`; left/right are defined looking downstream. |
| `width_from_banklines` | `width_from_banklines` | Measures intersections normal to the centerline and returns `NaN` where unresolved. |
| `width_from_mask` | `width_from_mask` | Returns a `WidthProfile` parameterized by true along-line distance. |
| `angles`, `curvatures` | `angles`, `curvatures` | Operate on `(x, y)` polylines and return radians and inverse-pixel curvature. |
| `migration_mask` | `migration_mask` | Returns a `MaskMigrationResult`. |
| `migration_cl` | `migration_centerlines` | `migration_cl` remains as a compatibility spelling, but returns one result dataclass. |
| `spatial_changes` | `channel_belt_from_masks`, `spatial_changes` | Belt construction and change aggregation are separate, testable operations. |
| `combine_georeffed_images` | `mosaic_rasters`, `pixel_to_world`, `stitch_polylines` | Explicit operations replace input-shape inference. Both spellings of the compatibility wrapper are exported. |

`smooth_polyline`, `resample_polyline`, and `polyline_intersections` replace
the roles of the bundled `savfilt`, `interparc`, and `intersections` helpers.
The interactive MATLAB-only `hand_clean` workflow is not ported; mask review
or editing must occur before calling the deterministic Python API.

## Preserved conventions and defaults

- The two exit sides identify upstream first and downstream second.
- Centerline extraction retains the 100-pixel minimum mask and minimum
  centerline length of five nominal widths.
- Bank extraction retains the 20-pixel minimum component size.
- Bank-intersection width searches three nominal widths to either side by
  default.
- Area/length width buffers each segment by two reach-average widths on each
  side and retains the legacy disconnected-component filter.
- Mask cutoffs are time-1-only connected components with area strictly larger
  than `2 * nominal_width**2`. The paper describes a factor of three for one
  application; the code's released factor of two is the library default and
  is configurable.
- Centerline cutoffs retain the shortening threshold of two nominal widths,
  the neighboring-polygon threshold of five width-squared units, and the
  5-erosion/10-dilation tail cleanup.
- Automatic belt construction exposes RivMAP's empirical dilation,
  smoothing, and padding scales as named parameters: 10, 50, and 20 nominal
  widths, respectively.

All Python masks are two-dimensional Boolean arrays. Polylines are zero-based
image `(x, y)` coordinates with `y` increasing downward. MATLAB vectors are
normally one-based, so archived MATLAB coordinates require an explicit
one-pixel conversion before use. Python result masks retain the input grid.

## Intentional differences and corrected defects

### Deterministic topology

Centerline extraction skeletonizes the largest filled component and finds the
minimum-cost path between declared exits. This removes side spurs without the
legacy iterative pruning sequence. Bank extraction similarly traces two
exit-to-exit perimeter paths. The outputs preserve reach orientation and
scale, but may choose a different pixel at ambiguous junctions.

Channel-belt construction is bounded by `max_iterations` and
`max_padded_pixels`, recomputes each dilation from the original temporal
union, verifies that the envelope contains all observations, and requires
disjoint cells that cover the envelope. Invalid or crossed geometry raises an
error instead of looping indefinitely or returning partial cells.

### Along-channel width

The MATLAB `width_from_mask` documentation defines scalar `spacing` as
along-centerline distance, but the implementation applies it to centerline
node indices. RivMAPy uses cumulative Euclidean distance for scalar spacing
and for explicit breakpoint vectors. This matters whenever consecutive
vertices are not exactly one unit apart.

### Cutoff classification

When a cutoff seed mask is supplied to MATLAB `migration_mask`, the selected
components are copied to the cutoff image but are not removed from accretion.
RivMAPy removes them in both automatic and seed-assisted branches, so cutoff
and accretion masks are disjoint.

The MATLAB `migration_cl` neighboring-intersection logic can index outside the
intersection array at an end candidate, and its tail cleanup can eliminate a
small cutoff before a largest-component lookup. RivMAPy bounds neighboring
lookups and safely falls back to the uncleaned cutoff when erosion removes
everything.

### Data and geospatial behavior

`load_rivmap_mat` validates the complete annual structure, returns read-only
masks, and supports the MATLAB v5 `riv` structure in `data/riv.mat`. It does
not deserialize Mapping Toolbox MCOS reference objects from
`data/riv_georeffed.mat`.

For new data, `read_mask` reads GeoTIFF metadata into a `GeoRaster`. Analysis
results retain its CRS, affine transform, and validity through geometry,
change, and belt workflows; cropping updates the origin. GeoTIFF and
GeoPackage exports reuse the source CRS. Measurements still use pixel units.
See [georeferenced workflows](georeferencing.md) for nodata handling and units.

New mosaics require explicit affine transforms, aligned north-up rasters, a
common pixel size, and integer grid offsets. Rotated, sheared, or sub-pixel
misaligned grids are rejected rather than silently resampled. Overlap policy
is explicit (`"first"` or `"last"`), and output bounds use the actual pixel
size. This removes the legacy routine's hard-coded 30-unit x-extent and stale
shape-inference state. Polyline pieces are transformed into world coordinates
before stitching.

### Historical demo caveats

The retained MATLAB demo is useful as a narrative record but is not a clean
automated script:

- It defines `data_dir`, then `clearvars -except riv` removes that variable;
  later load/save cells therefore depend on interactive workspace state.
- It mentions a preprocessed `riv_processed` fixture that is not distributed.
- Its processing cells save derived fields back into `data/riv.mat`.

The Python examples avoid workspace state, always recompute their products,
and write figures separately without mutating the fixtures.

## The seventh legacy cutoff

The walkthrough's cutoff section first shows five features in a binary
aggregate, then six when colored by interval, and finally describes seven
events in the cutoff-area time series. The seventh is a small event whose
location is overprinted by a later cutoff; it is not a seventh distinct
spatial feature available for direct comparison.

With the documented defaults, RivMAPy's full Ucayali example resolves six
distinct centerline-cutoff intervals: 1989–1990, 1992–1993, 1996–1997,
2004–2005, 2006–2007, and 2014–2015. These are the definite spatial
regression target. The legacy seventh remains an interpretive, temporally
separate signal, not a failure that should be hidden by tuning thresholds
until the count reaches seven.
