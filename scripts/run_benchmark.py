"""Run the comparison of rule learners and baselines.

Examples
--------
Full run as configured (5 seeds, 11 datasets, ~1-2 hours on a laptop, most of it IDS):

    python scripts/run_benchmark.py

Quick check on two datasets with one seed:

    python scripts/run_benchmark.py --datasets cells titanic --seeds 42

Writes ``raw_results.csv`` (one row per dataset/model/seed), ``summary.csv``
(mean and std over seeds) and ``rules/<dataset>__<model>.txt`` to the output
directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # allow running without `pip install -e .`

from xai.experiment import run_benchmark, summarise  # noqa: E402
from xai.models import MODELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "benchmark.yaml")
    p.add_argument("--datasets", nargs="+", help="override the datasets in the config")
    p.add_argument("--models", nargs="+", choices=list(MODELS), help="override the models in the config")
    p.add_argument("--seeds", nargs="+", type=int, help="override the seeds in the config")
    p.add_argument("--time-limit", type=float, help="seconds per fit (0 = no limit)")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def pin_hash_seed() -> None:
    """Re-run this script with PYTHONHASHSEED=0.

    pyIDS iterates over sets of rules, whose order depends on string hashes;
    without a fixed hash seed IDS picks slightly different rules in every
    process even with the same random seed.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def main() -> int:
    pin_hash_seed()
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = yaml.safe_load(args.config.read_text())
    datasets = args.datasets or cfg["datasets"]
    models = args.models or cfg["models"]
    seeds = args.seeds or cfg["seeds"]
    limit = cfg.get("time_limit") if args.time_limit is None else args.time_limit

    for key in models:
        missing = MODELS[key].missing_dependencies()
        if missing:
            logging.warning(
                "%s will be skipped, missing: %s (see scripts/install_legacy_deps.sh)",
                key,
                ", ".join(missing),
            )

    start = time.time()
    raw = run_benchmark(
        datasets,
        models,
        seeds,
        params=cfg.get("params"),
        seconds=limit or None,
        test_size=cfg.get("test_size", 0.2),
        out_dir=args.out,
    )
    summary = summarise(raw)
    summary.to_csv(args.out / "summary.csv", index=False)

    counts = raw["status"].value_counts().to_dict()
    logging.info("done in %.1f min: %s", (time.time() - start) / 60, counts)
    logging.info("results written to %s", args.out)
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
