import numpy as np
import pandas as pd

from sensor_quality.checks import (
    align_to_reference,
    cross_check_flags,
    frozen_flags,
    range_flags,
    zscore_flags,
)


def test_align_interpolates_linearly_onto_the_reference_clock() -> None:
    series = pd.Series([0.0, 10.0, 20.0], index=[0.0, 1.0, 2.0], name="x")
    aligned = align_to_reference(series, np.array([0.5, 1.5]))
    assert aligned.tolist() == [5.0, 15.0]
    assert aligned.name == "x"


def test_align_keeps_the_first_of_duplicate_timestamps() -> None:
    series = pd.Series([1.0, 99.0, 3.0], index=[0.0, 0.0, 1.0])
    aligned = align_to_reference(series, np.array([0.0]))
    assert aligned.tolist() == [1.0]


def test_range_flags_only_values_outside_the_limits() -> None:
    flags = range_flags(pd.Series([-1.0, 0.0, 50.0, 150.0, 151.0]), 0.0, 150.0)
    assert flags.tolist() == [True, False, False, False, True]


def test_zscore_flags_a_spike_but_not_a_smooth_ramp() -> None:
    ramp = pd.Series(np.linspace(0.0, 100.0, 300))
    assert not zscore_flags(ramp)[0].any()

    noisy = pd.Series(np.random.default_rng(1).normal(0.0, 0.2, 300))
    noisy.iloc[150] += 20.0
    flags, z = zscore_flags(noisy)
    assert flags.iloc[150]
    assert (
        z.abs().idxmax() == 150
    )  # a 3 sigma rule also flags a few ordinary noise samples by chance


def test_zscore_does_not_flag_when_local_std_is_zero() -> None:
    flags, z = zscore_flags(pd.Series(np.full(100, 5.0)))
    assert not flags.any()
    assert z.isna().all()


def test_frozen_flags_the_whole_stretch_and_only_long_enough_ones() -> None:
    values = [1.0, 2.0] + [3.0] * 8 + [4.0] + [5.0] * 7 + [6.0]
    flags = frozen_flags(pd.Series(values), min_run=8)
    assert flags.iloc[2:10].all()  # eight equal samples are a stretch
    assert flags.sum() == 8  # seven equal samples are not
    assert not flags.iloc[11:18].any()


def test_frozen_flags_a_stretch_that_runs_to_the_end() -> None:
    flags = frozen_flags(pd.Series([1.0, 2.0] + [9.0] * 10), min_run=8)
    assert flags.iloc[2:].all()
    assert not flags.iloc[:2].any()


def test_cross_check_flags_disagreement_above_the_limit() -> None:
    a = pd.Series([10.0, 10.0, 10.0])
    b = pd.Series([10.5, 15.0, 16.0])
    flags, deviation = cross_check_flags(a, b, limit=5.0)
    assert flags.tolist() == [False, False, True]
    assert deviation.tolist() == [0.5, 5.0, 6.0]
