"""Shared validation helpers for RivMAPy's array-first public API."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


VALID_SIDES = frozenset("NSEW")


def as_mask(mask: ArrayLike, *, name: str = "mask") -> NDArray[np.bool_]:
    """Return *mask* as a validated, two-dimensional Boolean array.

    Numeric arrays are accepted only when all values are exactly zero or one.
    A copy is returned so public functions cannot mutate caller-owned arrays.
    """

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.bool_):
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must contain Boolean or binary numeric values")
        finite = np.isfinite(array)
        if not np.all(finite) or not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} must contain only zero and one")
    result = np.array(array, dtype=bool, copy=True)
    if not np.any(result):
        raise ValueError(f"{name} must contain at least one channel pixel")
    return result


def as_coordinates(
    coordinates: ArrayLike,
    *,
    name: str = "coordinates",
    minimum_length: int = 2,
) -> NDArray[np.float64]:
    """Return finite ``(x, y)`` coordinates as a floating-point array."""

    array = np.asarray(coordinates, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if len(array) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} points")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def validate_exit_sides(exit_sides: str | Iterable[str]) -> str:
    """Normalize and validate the upstream/downstream image exit sides."""

    if isinstance(exit_sides, str):
        normalized = exit_sides.upper().replace(" ", "")
    else:
        normalized = "".join(exit_sides).upper()
    if len(normalized) != 2 or any(side not in VALID_SIDES for side in normalized):
        raise ValueError("exit_sides must contain two N/S/E/W characters")
    if normalized[0] == normalized[1]:
        raise ValueError("upstream and downstream exit sides must be distinct")
    return normalized


def positive_number(value: float, *, name: str) -> float:
    """Validate and return a finite, positive scalar."""

    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result

