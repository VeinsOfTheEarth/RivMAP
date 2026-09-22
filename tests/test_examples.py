from __future__ import annotations

import numpy as np

from examples.multiyear_change import _post_1995_widening


def test_widening_diagnostic_handles_early_quick_run() -> None:
    widening, low_year, peak_year = _post_1995_widening(
        np.arange(1984, 1989), np.array([10.0, 9.0, 11.0, 8.0, 12.0])
    )
    assert np.isnan(widening)
    assert low_year is None
    assert peak_year is None


def test_widening_diagnostic_uses_post_low_peak() -> None:
    widening, low_year, peak_year = _post_1995_widening(
        np.array([1994, 1995, 1996, 1997, 1998]),
        np.array([80.0, 70.0, 60.0, 75.0, 72.0]),
    )
    assert widening == 15.0
    assert low_year == 1996
    assert peak_year == 1997
