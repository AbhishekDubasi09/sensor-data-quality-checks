"""Run the checks on the synthetic two-probe run and write the results.

python -m sensor_quality.demo --out results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sensor_quality import synthetic
from sensor_quality.pipeline import evaluate, summarise

CROSS_LIMIT_C = 5.0


def run(seed: int = 7) -> tuple[pd.DataFrame, synthetic.SyntheticRun]:
    """Generate the run and evaluate it. Returns the per-sample frame and the run."""
    data = synthetic.make_run(seed)
    frame = evaluate(
        data.probe_a,
        data.probe_b,
        data.reference_t,
        low=synthetic.PROBE_LOW_C,
        high=synthetic.PROBE_HIGH_C,
        cross_limit=CROSS_LIMIT_C,
    )
    return frame, data


def plot(frame: pd.DataFrame, path: Path) -> None:
    """Both probes on the shared clock, with the flagged samples marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=140)
    ax.plot(frame["t_s"], frame["probe_a"], lw=1.0, label="probe A", color="#1f5eff")
    ax.plot(frame["t_s"], frame["probe_b"], lw=1.0, label="probe B", color="#e07b00")
    hits = frame[frame["flagged"]]
    ax.scatter(
        hits["t_s"], hits["probe_b"], s=14, color="#c62828", zorder=3, label="flagged sample"
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("temperature (C)")
    ax.set_title("Two probes on one clock, flagged samples marked (synthetic data)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    frame, _ = run(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "all_samples_with_flags.csv", index=False)
    frame[frame["flagged"]].to_csv(args.out / "flagged_samples.csv", index=False)
    summary = summarise(frame)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(frame, args.out / "flagged_run.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
