from pathlib import Path

import numpy as np
import pytest

from rivmapy.io import load_rivmap_mat


DATA_FILE = Path(__file__).parents[1] / "data" / "riv.mat"


def test_load_distributed_rivmap_fixture():
    records = load_rivmap_mat(DATA_FILE)
    assert len(records) == 32
    assert [records[0].year, records[-1].year] == [1984, 2015]
    assert records[0].exit_sides == "SW"
    assert records[0].nominal_width == 30
    assert records[0].hydraulically_connected.shape == (1823, 1345)
    assert records[0].single_thread.dtype == bool
    assert not records[0].single_thread.flags.writeable


def test_record_mask_aliases_and_validation():
    record = load_rivmap_mat(DATA_FILE)[0]
    assert record.mask("st") is record.single_thread
    assert record.mask("connected") is record.hydraulically_connected
    with pytest.raises(ValueError, match="kind"):
        record.mask("water")


def test_load_missing_file_is_explicit():
    with pytest.raises(FileNotFoundError):
        load_rivmap_mat(DATA_FILE.with_name("missing.mat"))
