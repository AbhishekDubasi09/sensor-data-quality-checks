"""The command line tool: reads a user CSV, writes flags, fails clearly on bad input."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensor_quality import cli


def _write_csv(path: Path, spike_row: int | None = 150, nan_rows: tuple[int, ...] = ()) -> None:
    t = np.arange(300) / 10.0
    a = 85.0 + 5.0 * np.sin(2 * np.pi * t / 20.0) + np.random.default_rng(1).normal(0, 0.2, 300)
    b = a + np.random.default_rng(2).normal(0, 0.2, 300)
    if spike_row is not None:
        b[spike_row] += 40.0
    frame = pd.DataFrame({"time_s": t, "temp_a": a, "temp_b": b})
    for row in nan_rows:
        frame.loc[row, "temp_b"] = np.nan
    frame.to_csv(path, index=False)


def _run(csv: Path, out: Path, *extra: str) -> int:
    return cli.main(
        [
            str(csv),
            "--time-col", "time_s",
            "--a-col", "temp_a",
            "--b-col", "temp_b",
            "--low", "0",
            "--high", "150",
            "--cross-limit", "5",
            "--out", str(out),
            *extra,
        ]
    )  # fmt: skip


def test_finds_an_injected_spike_and_writes_all_files(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv)
    assert _run(csv, tmp_path / "out") == 0
    flagged = pd.read_csv(tmp_path / "out" / "flagged_samples.csv")
    assert (flagged["t_s"].between(14.9, 15.1)).any()
    everything = pd.read_csv(tmp_path / "out" / "all_samples_with_flags.csv")
    assert len(everything) == 300  # nothing is dropped
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_rows"] == 300
    assert summary["missing_a"] == 0


def test_missing_values_are_counted_and_do_not_crash(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv, spike_row=None, nan_rows=(40, 41, 200))
    assert _run(csv, tmp_path / "out") == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["missing_b"] == 3
    assert summary["samples"] == 300


def test_unknown_column_gives_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv)
    code = cli.main(
        [str(csv), "--time-col", "nope", "--a-col", "temp_a", "--b-col", "temp_b"]
        + ["--low", "0", "--high", "150", "--cross-limit", "5", "--out", str(tmp_path / "o")]
    )
    assert code == 2
    assert "column 'nope' not found" in capsys.readouterr().err


def test_low_must_be_below_high(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = tmp_path / "data.csv"
    _write_csv(csv)
    code = cli.main(
        [str(csv), "--time-col", "time_s", "--a-col", "temp_a", "--b-col", "temp_b"]
        + ["--low", "150", "--high", "0", "--cross-limit", "5", "--out", str(tmp_path / "o")]
    )
    assert code == 2
    assert "--low must be smaller" in capsys.readouterr().err


def test_too_few_rows_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = tmp_path / "short.csv"
    pd.DataFrame({"time_s": [0, 1, 2], "temp_a": [1, 2, 3], "temp_b": [1, 2, 3]}).to_csv(
        csv, index=False
    )
    assert _run(csv, tmp_path / "out") == 2
    assert "at least 20 rows" in capsys.readouterr().err
