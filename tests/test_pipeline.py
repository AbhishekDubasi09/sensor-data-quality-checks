import numpy as np

from sensor_quality import demo, synthetic
from sensor_quality.pipeline import summarise


def test_generator_is_deterministic_for_a_seed() -> None:
    a = synthetic.make_run(3)
    b = synthetic.make_run(3)
    assert np.array_equal(a.probe_b.to_numpy(), b.probe_b.to_numpy())
    assert not np.array_equal(a.probe_b.to_numpy(), synthetic.make_run(4).probe_b.to_numpy())


def test_probe_clocks_really_differ() -> None:
    run = synthetic.make_run()
    assert not np.allclose(run.probe_a.index.to_numpy(), run.probe_b.index.to_numpy())


def test_no_sample_is_dropped_and_every_flag_has_a_reason() -> None:
    frame, run = demo.run()
    assert len(frame) == len(run.reference_t)
    assert (frame.loc[frame["flagged"], "reasons"] != "").all()
    assert (frame.loc[~frame["flagged"], "reasons"] == "").all()


def test_both_injected_faults_are_found() -> None:
    frame, run = demo.run()
    spike_t = run.probe_b.index[run.spike_index_b]
    spike_row = int(np.argmin(np.abs(frame["t_s"].to_numpy() - spike_t)))
    window = frame.iloc[spike_row - 1 : spike_row + 2]
    assert window["flagged"].any()
    assert window["cross"].any()

    frozen_mask = (frame["t_s"] >= synthetic.FROZEN_START_S + 0.5) & (
        frame["t_s"] < synthetic.FROZEN_END_S - 0.5
    )
    assert frame.loc[frozen_mask, "frozen_b"].all()


def test_probe_a_is_never_frozen_or_out_of_range() -> None:
    frame, _ = demo.run()
    assert not frame["frozen_a"].any()
    assert not frame["range_a"].any()


def test_summary_counts_match_the_frame() -> None:
    frame, _ = demo.run()
    summary = summarise(frame)
    assert summary["samples"] == len(frame) == 3000
    assert summary["flagged_samples"] == int(frame["flagged"].sum())
    assert summary["flags_frozen_b"] == int(frame["frozen_b"].sum())
