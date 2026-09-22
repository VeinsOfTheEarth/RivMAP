import numpy as np
import pytest

from rivmapy._validation import as_coordinates, as_mask, validate_exit_sides


def test_as_mask_accepts_binary_numeric_and_copies():
    source = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    result = as_mask(source)
    assert result.dtype == bool
    result[0, 0] = True
    assert source[0, 0] == 0


@pytest.mark.parametrize(
    "bad_mask",
    [np.zeros((2, 2)), np.array([0, 1]), np.array([[0, 2]])],
)
def test_as_mask_rejects_invalid_inputs(bad_mask):
    with pytest.raises(ValueError):
        as_mask(bad_mask)


def test_as_coordinates_enforces_shape_and_finiteness():
    result = as_coordinates([[0, 1], [2, 3]])
    assert result.dtype == float
    with pytest.raises(ValueError):
        as_coordinates([[0, np.nan], [2, 3]])


def test_validate_exit_sides_normalizes_and_rejects_duplicates():
    assert validate_exit_sides("s w") == "SW"
    with pytest.raises(ValueError):
        validate_exit_sides("SS")
