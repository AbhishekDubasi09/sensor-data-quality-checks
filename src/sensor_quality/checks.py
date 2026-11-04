"""Plausibility checks for sensor and measurement time series.

Every check returns a boolean flag per sample. Nothing here drops or edits data:
flagged samples are kept, together with the reason, so that a person can decide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_to_reference(series: pd.Series, reference_t: np.ndarray) -> pd.Series:
    """Resample a series onto a shared reference time base by linear interpolation.

    The series index must be time in seconds. Duplicate timestamps keep the first value.
    Times outside the recorded range take the nearest recorded value (``np.interp`` clamps).
    """
    clean = series[~series.index.duplicated(keep="first")].sort_index()
    values = np.interp(reference_t, clean.index.to_numpy(dtype=float), clean.to_numpy(dtype=float))
    return pd.Series(values, index=reference_t, name=series.name)


def range_flags(series: pd.Series, low: float, high: float) -> pd.Series:
    """Flag values outside the physically possible range ``[low, high]``."""
    return (series < low) | (series > high)


def zscore_flags(
    series: pd.Series, window: int = 50, threshold: float = 3.0
) -> tuple[pd.Series, pd.Series]:
    """Flag samples far from their local mean, measured in local standard deviations.

    A centred rolling window is used instead of one global mean, so a slow and expected
    change of the signal is not flagged by itself. Where the local standard deviation is
    zero the z-score is undefined (NaN) and the sample is not flagged here; the frozen-value
    check covers that case.

    Returns:
        (flags, z) where ``z`` is the rolling z-score.
    """
    roll_mean = series.rolling(window, center=True, min_periods=10).mean()
    roll_std = series.rolling(window, center=True, min_periods=10).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return z.abs() > threshold, z


def frozen_flags(series: pd.Series, min_run: int = 8, tol: float = 1e-6) -> pd.Series:
    """Flag every sample of a stretch where the value does not change.

    A stretch counts when at least ``min_run`` consecutive samples differ from each other
    by no more than ``tol``. A sensor that stopped updating usually stays inside the
    plausible range, so the range and z-score checks can miss it.
    """
    values = series.to_numpy(dtype=float)
    flat_step = np.abs(np.diff(values, prepend=np.nan)) <= tol  # sample i equals sample i-1
    flags = np.zeros(len(values), dtype=bool)
    start = 0
    for i in range(1, len(values) + 1):
        run_continues = i < len(values) and flat_step[i]
        if not run_continues:
            if i - start >= min_run:
                flags[start:i] = True
            start = i
    return pd.Series(flags, index=series.index)


def cross_check_flags(a: pd.Series, b: pd.Series, limit: float) -> tuple[pd.Series, pd.Series]:
    """Flag samples where two signals of the same physical quantity disagree by more than ``limit``.

    A growing gap means one of the two measurement chains has a problem, not that the
    process itself changed. Both series must share the same index.

    Returns:
        (flags, deviation) where ``deviation`` is ``|a - b|``.
    """
    deviation = (a - b).abs()
    return deviation > limit, deviation
