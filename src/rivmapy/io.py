"""Readers for the compact MATLAB fixtures distributed with RivMAP.

The analytical API operates on ordinary NumPy arrays; these helpers are only
an interchange bridge for historical ``.mat`` files.  They deliberately do
not attempt to deserialize Mapping Toolbox MCOS objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from ._validation import as_mask, positive_number, validate_exit_sides


@dataclass(frozen=True)
class RiverRecord:
    """One RivMAP observation loaded from the historical MATLAB structure."""

    year: int
    hydraulically_connected: NDArray[np.bool_]
    single_thread: NDArray[np.bool_] | None
    exit_sides: str
    nominal_width: float

    def mask(self, kind: str = "single_thread") -> NDArray[np.bool_]:
        """Return a read-only mask by descriptive or MATLAB field name."""

        normalized = kind.lower().replace("-", "_").replace(" ", "_")
        if normalized in {"single_thread", "st"}:
            if self.single_thread is None:
                raise ValueError("this record does not contain a single-thread mask")
            return self.single_thread
        if normalized in {"hydraulically_connected", "connected", "hc"}:
            return self.hydraulically_connected
        raise ValueError(
            "kind must be 'single_thread'/'st' or "
            "'hydraulically_connected'/'hc'"
        )


def _readonly_mask(value: Any, *, name: str) -> NDArray[np.bool_]:
    result = as_mask(value, name=name)
    result.flags.writeable = False
    return result


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        records = list(value)
    elif isinstance(value, np.ndarray):
        records = list(value.ravel())
    else:
        raise ValueError("the MATLAB 'riv' variable has an unsupported structure")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every element of MATLAB 'riv' must be a structure")
    return records


def load_rivmap_mat(
    path: str | PathLike[str],
    *,
    variable: str = "riv",
) -> tuple[RiverRecord, ...]:
    """Load RivMAP's annual-mask MATLAB structure.

    The loader supports the MATLAB v5 structure used by ``data/riv.mat`` and
    exposes its masks as read-only Boolean arrays.  All metadata and array
    shapes are validated before any records are returned.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not variable:
        raise ValueError("variable must be a non-empty string")
    contents = loadmat(source, simplify_cells=True)
    if variable not in contents:
        raise KeyError(f"{source} does not contain a {variable!r} variable")

    result: list[RiverRecord] = []
    expected_shape: tuple[int, int] | None = None
    for index, raw in enumerate(_records(contents[variable])):
        try:
            metadata = raw["meta"]
            images = raw["im"]
            year = int(metadata["year"])
            exit_sides = validate_exit_sides(str(metadata["exit_sides"]))
            nominal_width = positive_number(metadata["Wn"], name="Wn")
            connected = _readonly_mask(images["hc"], name=f"riv[{index}].im.hc")
            single = (
                _readonly_mask(images["st"], name=f"riv[{index}].im.st")
                if "st" in images and images["st"] is not None
                else None
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"MATLAB record {index} is missing required RivMAP fields"
            ) from exc
        if single is not None and single.shape != connected.shape:
            raise ValueError(f"MATLAB record {index} contains mismatched mask shapes")
        if expected_shape is None:
            expected_shape = connected.shape
        elif connected.shape != expected_shape:
            raise ValueError("all annual RivMAP masks must use the same grid")
        result.append(
            RiverRecord(
                year=year,
                hydraulically_connected=connected,
                single_thread=single,
                exit_sides=exit_sides,
                nominal_width=nominal_width,
            )
        )
    if not result:
        raise ValueError("the MATLAB 'riv' structure contains no observations")
    return tuple(result)


__all__ = ["RiverRecord", "load_rivmap_mat"]

