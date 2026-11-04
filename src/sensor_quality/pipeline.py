"""Run all checks on two probes and keep every sample with the reasons it was flagged."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from sensor_quality.checks import (
    align_to_reference,
    cross_check_flags,
    frozen_flags,
    range_flags,
    zscore_flags,
)

CHECKS = ("range", "zscore", "frozen", "cross")


def evaluate(
    probe_a: pd.Series,
    probe_b: pd.Series,
    reference_t: npt.NDArray[np.float64],
    low: float,
    high: float,
    cross_limit: float,
    z_window: int = 50,
    z_threshold: float = 3.0,
    frozen_min_run: int = 8,
) -> pd.DataFrame:
    """Align both probes and return one row per reference sample.

    Columns: the aligned values, one boolean column per check and probe, the cross-check
    deviation, ``flagged`` (any check) and ``reasons`` (comma separated, empty if clean).
    No row is removed.
    """
    a = align_to_reference(probe_a, reference_t)
    b = align_to_reference(probe_b, reference_t)
    frame = pd.DataFrame({"t_s": reference_t, "probe_a": a.to_numpy(), "probe_b": b.to_numpy()})
    frame = frame.set_index("t_s", drop=False)
    a, b = frame["probe_a"], frame["probe_b"]

    for name, series in (("a", a), ("b", b)):
        frame[f"range_{name}"] = range_flags(series, low, high)
        z_flag, z = zscore_flags(series, z_window, z_threshold)
        frame[f"zscore_{name}"] = z_flag
        frame[f"z_{name}"] = z
        frame[f"frozen_{name}"] = frozen_flags(series, frozen_min_run)

    cross, deviation = cross_check_flags(a, b, cross_limit)
    frame["cross"] = cross
    frame["cross_deviation"] = deviation

    flag_columns = [
        c for c in frame.columns if c.split("_")[0] in CHECKS and frame[c].dtype == bool
    ]
    frame["flagged"] = frame[flag_columns].any(axis=1)
    frame["reasons"] = frame[flag_columns].apply(
        lambda row: ",".join(name for name, hit in row.items() if hit), axis=1
    )
    return frame.reset_index(drop=True)


def summarise(frame: pd.DataFrame) -> dict[str, float | int]:
    """Counts per check and the correlation of the two aligned probes."""
    summary: dict[str, float | int] = {
        "samples": len(frame),
        "flagged_samples": int(frame["flagged"].sum()),
        "flagged_percent": round(100.0 * float(frame["flagged"].mean()), 2),
        "correlation_a_b": round(float(frame["probe_a"].corr(frame["probe_b"])), 4),
    }
    for column in frame.columns:
        if column.split("_")[0] in CHECKS and frame[column].dtype == bool:
            summary[f"flags_{column}"] = int(frame[column].sum())
    return summary
