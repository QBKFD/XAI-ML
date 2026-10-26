"""Draw the README figures from the benchmark results.

python scripts/make_figures.py
python scripts/make_figures.py --results results/benchmark --out docs/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_rules import plotting  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=ROOT / "results" / "benchmark")
    p.add_argument("--thesis", type=Path, default=ROOT / "results" / "thesis" / "performance.csv")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    args = p.parse_args()

    summary_path = args.results / "summary.csv"
    if not summary_path.exists():
        sys.exit(f"{summary_path} not found, run scripts/run_benchmark.py first")
    summary = pd.read_csv(summary_path)

    written = [
        plotting.accuracy_heatmap(summary, args.out / "accuracy_heatmap.png"),
        plotting.accuracy_heatmap(summary, args.out / "roc_auc_heatmap.png", metric="roc_auc"),
        plotting.accuracy_vs_rules(summary, args.out / "accuracy_vs_rules.png"),
        plotting.rule_set_shape(summary, args.out / "rule_set_shape.png"),
    ]
    if args.thesis.exists():
        written.append(
            plotting.thesis_vs_reproduced(
                summary, pd.read_csv(args.thesis), args.out / "thesis_vs_reproduced.png"
            )
        )
    for path in written:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
