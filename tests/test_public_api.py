from __future__ import annotations

import rivmapy


def test_public_api_is_importable() -> None:
    expected = {
        "ChannelBelt",
        "angles",
        "banklines_from_mask",
        "centerline_from_mask",
        "channel_belt_from_masks",
        "curvatures",
        "load_rivmap_mat",
        "migration_centerlines",
        "migration_mask",
        "mosaic_rasters",
        "spatial_changes",
        "width_from_mask",
    }

    assert expected <= set(rivmapy.__all__)
    assert all(hasattr(rivmapy, name) for name in expected)
    assert rivmapy.__version__ == "0.1.0"
