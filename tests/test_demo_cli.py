"""The demo command writes its files, and its numbers match the ones published in results/."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from sensor_quality import demo

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_demo(out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["demo", "--out", str(out)])
    demo.main()


def test_demo_writes_all_output_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_demo(tmp_path, monkeypatch)
    for name in (
        "all_samples_with_flags.csv",
        "flagged_samples.csv",
        "summary.json",
        "flagged_run.png",
    ):
        assert (tmp_path / name).is_file(), name


def test_flagged_csv_has_only_flagged_rows_with_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_demo(tmp_path, monkeypatch)
    everything = pd.read_csv(tmp_path / "all_samples_with_flags.csv")
    flagged = pd.read_csv(tmp_path / "flagged_samples.csv")
    assert len(everything) == 3000  # nothing is dropped
    assert len(flagged) == int(everything["flagged"].sum())
    assert flagged["reasons"].notna().all()


def test_summary_matches_the_published_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README quotes results/summary.json, so a change in the checks must update it."""
    _run_demo(tmp_path, monkeypatch)
    fresh = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    published = json.loads((REPO_ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    assert fresh == published
