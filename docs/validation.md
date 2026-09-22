# Validation

RivMAPy's validation strategy is deliberately small and reproducible. Fast
synthetic tests establish algorithm and API contracts; two bundled Ucayali
workflows check that the port remains at the scale and interpretation of the
legacy RivMAP results. Pixel-for-pixel MATLAB equivalence is not an acceptance
criterion.

## Reproduce the checks

Use Python 3.10 or newer from the repository root:

```console
python -m pip install -e ".[dev]"
python -m pytest -q -p no:cacheprovider
python examples/static_geometry.py
python examples/multiyear_change.py
python examples/georeferenced_workflow.py
```

At port completion on 2026-08-31, the test command completes with all tests
passing. The exact count may grow as focused regression cases are added. The
full temporal example processes 32 full-resolution masks and can take a few
minutes; `--maximum-records 5` is useful only as a smoke test and does not
exercise the documented long-term result.

The Ucayali examples are deterministic, read `data/riv.mat` without
modification, and generate review figures under `examples/output/`. Generated
images are not golden test fixtures. The georeferenced example creates separate
synthetic GeoTIFF inputs and checks exported GIS products by rereading them.

## Automated coverage

The test suite covers:

- Boolean-mask, coordinate, exit-side, shape, and numeric validation
- Horizontal, vertical, diagonal, curved, and reversed polyline geometry
- Skeleton connectivity, centerline orientation, bank assignment, and both
  width methods
- Synthetic migration, cutoff thresholds, seeded cutoff handling, and
  spatial aggregation
- Channel-belt containment, cell coverage, disjointness, and bounded failure
- Raster alignment, overlap policy, affine conversion, and polyline stitching
- Automatic metadata propagation through static, temporal, and belt workflows
- Native-CRS GeoTIFF/GeoPackage round trips, crop offsets, pixel-center versus
  pixel-edge coordinates, nodata/valid-zero separation, and layer-safe overwrite
- Unknown-bank rejection and common-validity handling across observation dates
- Loading and validating the distributed MATLAB v5 fixture
- A 1984 static Ucayali regression and the complete mask-derived
  erosion/accretion balance

The full 32-year centerline/cutoff run is documented as a manual acceptance
workflow because it is substantially slower than the unit suite.

The georeferencing extension was verified on 2026-09-06 with Python 3.12 on
Windows: 133 tests passed, the synthetic GIS demonstration completed, and the
package wheel built successfully. Array-only import and mask differencing
also worked with the optional geospatial dependencies blocked.

Geospatial tests are skipped when their optional dependencies are missing;
`.[dev]` is required for full coverage. The georeferenced workflow is also
covered by an end-to-end automated test. Its known UTM grid is synthetic and
does not establish the real-world origin of the bundled MATLAB masks. See
[georeferencing](georeferencing.md) for the workflow and its limitations.

## Bundled fixture contract

`data/riv.mat` contains 32 annual records from 1984 through 2015. The masks
are `1823 × 1345`, the first record declares south-to-west (`"SW"`) flow,
and the nominal width is 30 pixels. The Python loader returns read-only
Boolean arrays and rejects incomplete records or inconsistent grids.

The following targets are encoded in `tests/test_io.py` and
`tests/test_ucayali_regression.py`:

| Measurement | Reference result | Acceptance target |
| --- | ---: | ---: |
| Record count and years | 32; 1984–2015 | Exact |
| Mask shape | 1823 × 1345 | Exact |
| 1984 centerline length | 4228.79 px | 4228.8 ± 10 px |
| 1984 area/length reach width | 25.096 px | 25.1 ± 0.3 px |
| 1984 mean bank-intersection width | 25.365 px | 25.4 ± 1.0 px |
| 1984–2015 erosion minus accretion at 30 m | 102.003 km² | 102 ± 3 km² |

The width values reproduce the legacy demonstration's approximately 25-pixel
channel scale. The net mask balance reproduces its conclusion of roughly
100 km² more erosion than accretion.

## Full temporal reference result

With all records and a 30 m pixel size, `examples/multiyear_change.py`
currently reports:

```text
years: 1984-2015
centerline cutoffs: 6
width change: 39.9 m
post-1995 low-to-peak widening: 305.8 m (1997-2012)
cumulative erosion - accretion: 102.00 km^2
mean centerline migration: 72.97 m/yr
```

The six definite cutoff intervals are:

| Time 1 | Time 2 | Detected events |
| ---: | ---: | ---: |
| 1989 | 1990 | 1 |
| 1992 | 1993 | 1 |
| 1996 | 1997 | 1 |
| 2004 | 2005 | 1 |
| 2006 | 2007 | 1 |
| 2014 | 2015 | 1 |

The legacy demo labels intervals by their starting year, which is why it
refers to the 2004 cutoff while the Python plot places its bar at the ending
year 2005.

The 39.9 m value is the difference between the 1984 and 2015 endpoint widths;
it is not the same statistic as the walkthrough's discussion of widening
after the mid-1990s. The 305.8 m low-to-peak value is the comparable Python
diagnostic and is consistent in scale with the walkthrough's approximately
250 m interpretation. It is not expected to match exactly because the
centerline paths and smoothing implementations differ.

## Full channel-belt smoke result

A separate manual smoke run used all 32 single-thread masks with `"SW"`
exits, a 30-pixel nominal width, and 63-pixel cell spacing. It returned 35
nonempty cells and a 1,770,270-pixel envelope. The union of the cell masks
equaled the envelope exactly, no pixel belonged to more than one cell, and
the envelope contained the complete temporal channel union.

For this strongly curved adjacent-exit belt, automatic normal pairing is
ambiguous and the validated monotone bank-progress fallback is used. The
63-pixel target is applied along its dense midline; straight chords between
the returned sparse stations range from about 51 to 61 pixels. Consumers
should use `midpoint_distances` for spatial plotting rather than assuming
constant Euclidean chord lengths.

On the completion environment, that run took approximately 46 seconds and
reached approximately 544 MiB peak memory. These values are useful sizing
observations, not portable performance guarantees. Belt construction pads
the raster according to nominal width and materializes a dense
`(n_cells, rows, columns)` Boolean result, so runtime and memory are sensitive
to the input extent. Crop a large source raster to a common, scientifically
defensible reach before construction, while retaining both declared exits and
every channel position in the analysis period. The crop is part of the
scientific definition of the reach: changing it can alter the belt geometry
and cell count, not only performance.

## Six events versus the legacy seventh

The legacy walkthrough successively describes five features in a binary
cutoff map, six in a year-colored map, and seven in the cutoff-area time
series. The seventh is a small event later overprinted at the same location,
so it does not remain a seventh separable spatial object.

RivMAPy's regression target is the six distinct, definite events listed
above. The seventh remains useful qualitative evidence of a small temporal
event, but it is not used to force a seventh deterministic raster component.
A future change should not be accepted merely because its total count is
seven; event locations, intervals, areas, and false positives require review.

## Scientific invariants

In addition to the Ucayali summary values, changes should preserve these
properties:

- Returned centerlines, banks, and belt axes are ordered upstream to
  downstream in zero-based image `(x, y)` coordinates.
- Left and right banks remain correct while looking downstream, including
  when the channel direction is reversed in the image.
- Mask-derived erosion, accretion, unchanged area, and cutoffs are mutually
  exclusive classifications; seeded cutoffs are not double-counted as
  accretion.
- Centerline migration and cutoff masks are disjoint.
- On fully observed grids, belt cells are nonempty and disjoint, cover the
  returned envelope, and the envelope contains every input channel pixel.
  On partially observed grids, the envelope and cells are clipped to common
  validity; empty clipped cells retain their indices and have zero counts.
- Scalar segment spacing means cumulative along-line distance, not vertex
  index.
- Geospatial mosaics never silently resample, rotate, or accept a sub-pixel
  offset.
- Invalid, disconnected, crossed, or undersized inputs fail explicitly
  rather than returning plausible partial output.

## Interpreting numerical changes

Small pixel-boundary differences can result from dependency upgrades, and
the regression tolerances allow modest implementation-level variation. A
change outside those tolerances needs an explanation and visual review; it
should not be normalized by simply widening the tolerance. For an immutable
archival computation, record the Python and dependency versions alongside
the derived products.

Review `docs/legacy_behavior.md` before treating a MATLAB/Python difference as
a defect. Several differences correct known legacy behavior or make an
implicit convention explicit.
