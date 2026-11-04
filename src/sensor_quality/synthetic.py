"""A synthetic test run with two temperature probes and two injected faults.

The data is generated, not measured. It exists so that the checks can be shown on a case
where the right answer is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 10.0
DURATION_S = 300.0

SPIKE_T_S = 150.0
SPIKE_SIZE_C = 40.0
FROZEN_START_S = 210.0
FROZEN_END_S = 216.0

PROBE_LOW_C = 0.0
PROBE_HIGH_C = 150.0


@dataclass(frozen=True)
class SyntheticRun:
    """Two probes on separate clocks, plus the true positions of the injected faults."""

    probe_a: pd.Series
    probe_b: pd.Series
    reference_t: np.ndarray
    spike_index_b: int
    frozen_slice_b: slice


def _true_temperature(t: np.ndarray) -> np.ndarray:
    """Slowly varying process: 85 C with a 120 s swing of +-12 C and a small drift."""
    return 85.0 + 12.0 * np.sin(2 * np.pi * t / 120.0) + 0.01 * t


def make_run(seed: int = 7) -> SyntheticRun:
    """Build the two-probe run.

    Probe A logs on a 10 Hz clock. Probe B logs at 10 Hz on its own clock that is offset by
    0.03 s and jitters by +-10 ms, so the two series only line up after resampling.
    Probe B gets two faults: a one-sample spike at 150 s and a stuck value from 210 s to 216 s.
    """
    rng = np.random.default_rng(seed)
    n = int(DURATION_S * SAMPLE_RATE_HZ)
    nominal_t = np.arange(n) / SAMPLE_RATE_HZ

    t_a = nominal_t
    a = _true_temperature(t_a) + rng.normal(0.0, 0.3, n)

    t_b = np.sort(nominal_t + 0.03 + rng.uniform(-0.01, 0.01, n))
    b = _true_temperature(t_b) + rng.normal(0.0, 0.3, n)

    spike_index = int(np.argmin(np.abs(t_b - SPIKE_T_S)))
    b[spike_index] += SPIKE_SIZE_C

    frozen = np.where((t_b >= FROZEN_START_S) & (t_b < FROZEN_END_S))[0]
    frozen_slice = slice(int(frozen[0]), int(frozen[-1]) + 1)
    b[frozen_slice] = b[frozen[0]]

    return SyntheticRun(
        probe_a=pd.Series(a, index=t_a, name="probe_a"),
        probe_b=pd.Series(b, index=t_b, name="probe_b"),
        reference_t=nominal_t,
        spike_index_b=spike_index,
        frozen_slice_b=frozen_slice,
    )
