from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rivmapy.change import migration_mask
from rivmapy.io import load_rivmap_mat
from rivmapy.mask import (
    banklines_from_mask,
    centerline_from_mask,
    width_from_banklines,
)


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "riv.mat"


def test_1984_static_workflow_matches_legacy_scale() -> None:
    records = load_rivmap_mat(DATA_FILE)
    record = records[0]
    centerline = centerline_from_mask(
        record.single_thread,
        record.exit_sides,
        record.nominal_width,
    )
    banks = banklines_from_mask(record.single_thread, record.exit_sides)
    bank_width = width_from_banklines(
        centerline.coordinates,
        banks.left,
        banks.right,
        record.nominal_width,
    )
    length = np.linalg.norm(np.diff(centerline.coordinates, axis=0), axis=1).sum()
    area_width = record.single_thread.sum() / length

    assert len(records) == 32
    assert (records[0].year, records[-1].year) == (1984, 2015)
    assert length == pytest.approx(4228.8, abs=10.0)
    assert area_width == pytest.approx(25.1, abs=0.3)
    assert np.nanmean(bank_width) == pytest.approx(25.4, abs=1.0)


def test_multiyear_mask_balance_matches_walkthrough_scale() -> None:
    records = load_rivmap_mat(DATA_FILE)
    net_erosion_pixels = 0
    for first, second in zip(records[:-1], records[1:], strict=True):
        result = migration_mask(
            first.single_thread,
            second.single_thread,
            first.nominal_width,
        )
        net_erosion_pixels += int(result.erosion.sum() - result.accretion.sum())

    net_erosion_square_kilometres = net_erosion_pixels * 30.0**2 / 1_000_000
    assert net_erosion_square_kilometres == pytest.approx(102.0, abs=3.0)
