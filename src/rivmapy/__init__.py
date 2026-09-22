"""RivMAPy: river morphodynamics from binary channel planforms.

RivMAPy is a Python port of the original MATLAB RivMAP toolbox.  Array
coordinates follow image convention: ``(x, y) == (column, row)``, use
zero-based indices, and increase downward along the y axis.
"""

from __future__ import annotations

from .belt import ChannelBelt, channel_belt_from_masks
from .change import (
    CenterlineMigrationResult,
    MaskMigrationResult,
    SpatialChangeResult,
    migration_centerlines,
    migration_cl,
    migration_mask,
    spatial_cells_from_edges,
    spatial_changes,
)
from .geometry import (
    angles,
    curvatures,
    polyline_intersections,
    resample_polyline,
    smooth_polyline,
)
from .geospatial import (
    MosaicResult,
    combine_georeferenced_images,
    combine_georeffed_images,
    mosaic_rasters,
    pixel_to_world,
    stitch_polylines,
)
from .io import RiverRecord, load_rivmap_mat
from .mask import (
    BanklineResult,
    CenterlineResult,
    CropResult,
    WidthProfile,
    WidthSamples,
    banklines_from_mask,
    centerline_from_mask,
    crop_to_mask,
    skeleton_coords,
    width_from_banklines,
    width_from_mask,
)
from .spatial import GeoRaster, GeoVector, RasterGrid, mosaic_georasters
from .spatial_io import (
    read_geotiff,
    read_mask,
    read_geopackage,
    write_geotiff,
    write_geopackage,
)

__version__ = "0.1.0"

__all__ = [
    "BanklineResult",
    "CenterlineMigrationResult",
    "CenterlineResult",
    "ChannelBelt",
    "CropResult",
    "MaskMigrationResult",
    "MosaicResult",
    "RiverRecord",
    "SpatialChangeResult",
    "WidthProfile",
    "WidthSamples",
    "GeoRaster",
    "GeoVector",
    "RasterGrid",
    "angles",
    "banklines_from_mask",
    "centerline_from_mask",
    "channel_belt_from_masks",
    "combine_georeferenced_images",
    "combine_georeffed_images",
    "crop_to_mask",
    "curvatures",
    "load_rivmap_mat",
    "migration_centerlines",
    "migration_cl",
    "migration_mask",
    "mosaic_rasters",
    "mosaic_georasters",
    "pixel_to_world",
    "polyline_intersections",
    "resample_polyline",
    "read_geotiff",
    "read_mask",
    "read_geopackage",
    "skeleton_coords",
    "smooth_polyline",
    "spatial_cells_from_edges",
    "spatial_changes",
    "stitch_polylines",
    "width_from_banklines",
    "width_from_mask",
    "write_geotiff",
    "write_geopackage",
    "__version__",
]
