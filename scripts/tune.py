"""Hyperparameter tuning for RuleOpt (grid search) and IDS (lambda weights).

Mirrors section 3.2.3 of the thesis. For each dataset the seed-42 training
split is used for tuning; the test split is only touched once at the end to
compare the default and tuned model.

    python scripts/tune.py ruleopt --datasets cells glass titanic cars
    python scripts/tune.py ids --datasets cells glass titanic cars --max-iterations 1

IDS tuning refits the model many times (coordinate ascent over seven weights),
so it is slow: roughly 50-100 fits per dataset per iteration.

Output: ``results/tuning/<model>.csv`` with the chosen parameters and the
test metrics before and after tuning.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai_rules.experiment import evaluate, make_split  # noqa: E402
from xai_rules.metrics import classification_metrics  # noqa: E402
from xai_rules.models import IDSModel, RuleOptModel  # noqa: E402

log = logging.getLogger("tune")

# Thesis grid (section 3.2.3) restricted to the parameters ruleopt>=1.2 still accepts.
RULEOPT_GRID = {
    "rule_cost": ["gini", "length"],
    "max_rmp_calls": [3, 6],
    "max_depth": [2, 4, 10],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 4],
    "threshold": [1e-4, 1e-2],
}
REPORT = [
    "accuracy",
    "f1",
    "roc_auc",
    "n_rules",
    "avg_rule_width",
    "frac_overlap",
    "frac_uncovered",
    "frac_classes",
]


def tune_ruleopt(dataset: str, seed: int, folds: int) -> dict:
    split = make_split(dataset, seed)
    X_tr, _ = split.view("numeric")
    search = GridSearchCV(
        RuleOptModel(random_state=seed),
        RULEOPT_GRID,
        scoring="accuracy",
        cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
        n_jobs=1,
        error_score=np.nan,
    )
    search.fit(X_tr, split.y_train)
    default = RuleOptModel(random_state=seed).fit(X_tr, split.y_train)
    return {
        "dataset": dataset,
        "best_params": search.best_params_,
        "cv_accuracy": search.best_score_,
        "before": evaluate(default, split),
        "after": evaluate(search.best_estimator_, split),
    }


def tune_ids(dataset: str, seed: int, low: int, high: int, max_iterations: int, precision: int) -> dict:
    from pyids.model_selection.coordinate_ascent import CoordinateAscent

    split = make_split(dataset, seed)
    X_tr, _ = split.view("categorical")
    # Validation part of the training split, used to score each lambda vector.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, split.y_train, test_size=0.25, random_state=seed, stratify=split.y_train
    )
    n_evals = 0

    def objective(lambda_dict: dict) -> float:
        nonlocal n_evals
        n_evals += 1
        lambdas = tuple(float(lambda_dict[f"l{i}"]) for i in range(1, 8))
        model = IDSModel(lambdas=lambdas).fit(X_fit, y_fit)
        auc = classification_metrics(
            y_val, model.predict(X_val), model.predict_proba(X_val), list(model.classes_)
        )["roc_auc"]
        log.debug("%s %s -> AUC %.3f", dataset, [round(v) for v in lambdas], auc)
        return 0.0 if np.isnan(auc) else auc

    ascent = CoordinateAscent(
        func=objective,
        func_args_ranges={f"l{i}": (low, high) for i in range(1, 8)},
        ternary_search_precision=precision,
        max_iterations=max_iterations,
    )
    # The starting point is drawn at random; seed it so tuning is repeatable.
    random.seed(seed)
    np.random.seed(seed)
    best = ascent.fit()  # list of the seven weights, in l1..l7 order
    lambdas = tuple(float(v) for v in best)

    default = IDSModel().fit(X_tr, split.y_train)
    tuned = IDSModel(lambdas=lambdas).fit(X_tr, split.y_train)
    return {
        "dataset": dataset,
        "best_params": {f"l{i}": round(v, 1) for i, v in enumerate(lambdas, 1)},
        "n_evaluations": n_evals,
        "before": evaluate(default, split),
        "after": evaluate(tuned, split),
    }


def to_rows(result: dict) -> list[dict]:
    rows = []
    for stage in ("before", "after"):
        row = {"dataset": result["dataset"], "stage": stage}
        row.update({k: result[stage][k] for k in REPORT})
        rows.append(row)
    return rows


def pin_hash_seed() -> None:
    """Re-run this script with PYTHONHASHSEED=0.

    pyIDS iterates over Python sets, so without a fixed hash seed IDS can pick
    slightly different rules in every process even with the same random seed.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def main() -> int:
    pin_hash_seed()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", choices=["ruleopt", "ids"])
    p.add_argument("--datasets", nargs="+", default=["cells", "glass", "titanic", "cars"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5, help="CV folds for RuleOpt")
    p.add_argument(
        "--lambda-range",
        nargs=2,
        type=int,
        default=[1, 400],
        metavar=("LOW", "HIGH"),
        help="search range for each IDS weight (thesis: 1-400)",
    )
    p.add_argument("--max-iterations", type=int, default=1, help="coordinate ascent passes for IDS")
    p.add_argument("--precision", type=int, default=50, help="ternary search precision for IDS")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "tuning")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.out.mkdir(parents=True, exist_ok=True)

    results, rows = [], []
    for ds in args.datasets:
        start = time.time()
        if args.model == "ruleopt":
            res = tune_ruleopt(ds, args.seed, args.folds)
        else:
            res = tune_ids(ds, args.seed, *args.lambda_range, args.max_iterations, args.precision)
        res["seconds"] = round(time.time() - start, 1)
        log.info(
            "%s: %s  test acc %.3f -> %.3f, rules %s -> %s (%.0fs)",
            ds,
            res["best_params"],
            res["before"]["accuracy"],
            res["after"]["accuracy"],
            res["before"]["n_rules"],
            res["after"]["n_rules"],
            res["seconds"],
        )
        results.append(res)
        rows += to_rows(res)
        # Write after each dataset so a long IDS run keeps partial results.
        pd.DataFrame(rows).to_csv(args.out / f"{args.model}.csv", index=False)
        (args.out / f"{args.model}_params.json").write_text(
            json.dumps({r["dataset"]: r["best_params"] for r in results}, indent=2, default=str)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
