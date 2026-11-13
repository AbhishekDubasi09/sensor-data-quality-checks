"""Run the plausibility checks on your own CSV file.

    python -m sensor_quality.cli data.csv --time-col t --a-col probe_a --b-col probe_b \\
        --low 0 --high 150 --cross-limit 5 --out my_results

The CSV needs one time column in seconds and two columns that measure the same quantity.
Probe B is interpolated onto the time stamps of probe A. Missing values (NaN) cannot be
interpolated, so they are left out of the interpolation and counted in the summary.
Every row of the time column stays in the output with its flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_quality.pipeline import evaluate, summarise


def load_probes(
    path: Path, time_col: str, a_col: str, b_col: str
) -> tuple[pd.Series, pd.Series, np.ndarray, dict[str, int]]:
    """Read the CSV and return probe A, probe B, the reference time base and missing counts."""
    frame = pd.read_csv(path)
    for column in (time_col, a_col, b_col):
        if column not in frame.columns:
            raise ValueError(f"column '{column}' not found, available: {list(frame.columns)}")
    numeric = frame[[time_col, a_col, b_col]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(subset=[time_col]).sort_values(time_col)
    numeric = numeric.drop_duplicates(subset=[time_col], keep="first")
    if len(numeric) < 20:
        raise ValueError("need at least 20 rows with a valid time stamp")
    reference_t = numeric[time_col].to_numpy(dtype=float)
    missing = {
        "input_rows": len(frame),
        "missing_a": int(numeric[a_col].isna().sum()),
        "missing_b": int(numeric[b_col].isna().sum()),
    }
    probe_a = pd.Series(
        numeric[a_col].to_numpy(dtype=float), index=numeric[time_col].to_numpy(), name="probe_a"
    ).dropna()
    probe_b = pd.Series(
        numeric[b_col].to_numpy(dtype=float), index=numeric[time_col].to_numpy(), name="probe_b"
    ).dropna()
    if probe_a.empty or probe_b.empty:
        raise ValueError("a probe column has no numeric values")
    return probe_a, probe_b, reference_t, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("csv", type=Path, help="input CSV file")
    parser.add_argument("--time-col", required=True, help="column with time in seconds")
    parser.add_argument("--a-col", required=True, help="column of probe A")
    parser.add_argument("--b-col", required=True, help="column of probe B")
    parser.add_argument("--low", type=float, required=True, help="lowest physically possible value")
    parser.add_argument(
        "--high", type=float, required=True, help="highest physically possible value"
    )
    parser.add_argument(
        "--cross-limit", type=float, required=True, help="largest accepted gap between A and B"
    )
    parser.add_argument("--z-window", type=int, default=50, help="rolling window in samples")
    parser.add_argument("--z-threshold", type=float, default=3.0, help="z-score limit")
    parser.add_argument(
        "--frozen-run", type=int, default=8, help="equal samples that count as stuck"
    )
    parser.add_argument("--out", type=Path, default=Path("sensor_check_results"))
    args = parser.parse_args(argv)

    if args.low >= args.high:
        print("error: --low must be smaller than --high", file=sys.stderr)
        return 2
    try:
        probe_a, probe_b, reference_t, missing = load_probes(
            args.csv, args.time_col, args.a_col, args.b_col
        )
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    frame = evaluate(
        probe_a,
        probe_b,
        reference_t,
        low=args.low,
        high=args.high,
        cross_limit=args.cross_limit,
        z_window=args.z_window,
        z_threshold=args.z_threshold,
        frozen_min_run=args.frozen_run,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "all_samples_with_flags.csv", index=False)
    frame[frame["flagged"]].to_csv(args.out / "flagged_samples.csv", index=False)
    summary: dict[str, float | int] = {**missing, **summarise(frame)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
