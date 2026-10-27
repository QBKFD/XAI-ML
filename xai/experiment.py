"""Benchmark loop: datasets x models x random splits.

Every (dataset, model, seed) combination is isolated: a model that is not
applicable, crashes or runs out of time is recorded with its status and the
run continues with the next one.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from xai.datasets import DatasetSpec, load_dataset
from xai.encoding import make_encoder
from xai.metrics import classification_metrics, interpretability_metrics
from xai.models import MODELS, NotApplicable, get_model
from xai.rules import format_rules

log = logging.getLogger(__name__)


class TimeLimitExceeded(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float | None) -> Iterator[None]:
    """Raise ``TimeLimitExceeded`` after ``seconds`` (POSIX main thread only).

    Pure-Python code is interrupted promptly; a long call into C (an LP solve,
    for instance) is interrupted once it returns. Elsewhere this is a no-op.
    """
    usable = seconds and hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def handler(signum, frame):
        raise TimeLimitExceeded(f"exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class Split:
    """One train/test split of a dataset, with encoded views built on demand."""

    spec: DatasetSpec
    seed: int
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    _views: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)

    def view(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if name not in self._views:
            enc = make_encoder(name, self.spec)
            self._views[name] = (enc.fit_transform(self.X_train), enc.transform(self.X_test))
        return self._views[name]

    @property
    def classes(self) -> list[str]:
        return sorted(self.y_train.unique())


def make_split(dataset: str, seed: int, test_size: float = 0.2, data_dir=None) -> Split:
    X, y, spec = load_dataset(dataset, data_dir)
    # Stratify so small classes (glass has 9 rows of one type) appear in both parts.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return Split(spec, seed, X_tr, X_te, y_tr, y_te)


def evaluate(model, split: Split) -> dict[str, Any]:
    """Fit-free evaluation of an already fitted model on the split's test part."""
    _, X_te = split.view(model.view)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    row = classification_metrics(split.y_test, pred, proba, list(model.classes_))
    row.update(interpretability_metrics(model.rules_, X_te, split.classes))
    return row


def run_one(
    split: Split,
    model_key: str,
    params: dict[str, Any] | None = None,
    seconds: float | None = 300,
) -> tuple[dict[str, Any], Any]:
    """Fit and evaluate one model. Returns ``(result_row, fitted_model_or_None)``."""
    row: dict[str, Any] = {
        "dataset": split.spec.name,
        "model": model_key,
        "seed": split.seed,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "n_classes": len(split.classes),
    }
    model = None
    try:
        model = get_model(model_key, **(params or {}))
        model.check_applicable(len(split.X_train), len(split.classes))
        X_tr, _ = split.view(model.view)
        row["n_features"] = X_tr.shape[1]
        start = time.perf_counter()
        with time_limit(seconds):
            model.fit(X_tr, split.y_train)
        row["fit_seconds"] = time.perf_counter() - start
        row.update(evaluate(model, split))
        row["status"] = "ok"
    except NotApplicable as exc:
        row.update(status="skipped", message=str(exc))
        model = None
    except TimeLimitExceeded as exc:
        row.update(status="timeout", message=str(exc))
        model = None
    except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
        row.update(status="failed", message=f"{type(exc).__name__}: {exc}")
        log.debug("%s on %s failed:\n%s", model_key, split.spec.name, traceback.format_exc())
        model = None
    return row, model


def run_benchmark(
    datasets: list[str],
    models: list[str],
    seeds: list[int],
    params: dict[str, dict[str, dict[str, Any]]] | None = None,
    seconds: float | None = 300,
    test_size: float = 0.2,
    out_dir: str | Path | None = None,
    data_dir=None,
) -> pd.DataFrame:
    """Run every combination and return one row per (dataset, model, seed).

    ``params`` maps ``dataset -> model -> kwargs``; the key ``"default"`` applies
    to all datasets and is overridden by dataset-specific entries. When
    ``out_dir`` is given, results are written after every fit (so an
    interrupted run keeps what it has) and the rules of the first seed are
    saved as text.
    """
    params = params or {}
    unknown = set(models) - set(MODELS)
    if unknown:
        raise KeyError(f"Unknown models: {sorted(unknown)}")
    out = Path(out_dir) if out_dir else None
    if out:
        (out / "rules").mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in datasets:
        for seed in seeds:
            split = make_split(dataset, seed, test_size, data_dir)
            for key in models:
                kwargs = {**params.get("default", {}).get(key, {}), **params.get(dataset, {}).get(key, {})}
                row, model = run_one(split, key, kwargs, seconds)
                rows.append(row)
                _log_row(row)
                if out is not None:
                    pd.DataFrame(rows).to_csv(out / "raw_results.csv", index=False)
                    if model is not None and model.rules_ is not None and seed == seeds[0]:
                        _save_rules(out / "rules" / f"{dataset}__{key}.txt", model, split)
    return pd.DataFrame(rows)


def _log_row(row: dict[str, Any]) -> None:
    if row["status"] == "ok":
        log.info(
            "%-10s %-14s seed=%-3s acc=%.3f auc=%.3f rules=%s  %.1fs",
            row["dataset"],
            row["model"],
            row["seed"],
            row["accuracy"],
            row["roc_auc"],
            "-" if np.isnan(row["n_rules"]) else int(row["n_rules"]),
            row["fit_seconds"],
        )
    else:
        log.info(
            "%-10s %-14s seed=%-3s %s (%s)",
            row["dataset"],
            row["model"],
            row["seed"],
            row["status"],
            row.get("message", ""),
        )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _save_rules(path: Path, model, split: Split) -> None:
    kind = "rule list (first matching rule applies)" if model.ordered else "rule set (unordered)"
    header = (
        f"# {model.display_name} on {split.spec.name}, seed {split.seed}\n"
        f"# {kind}; {_plural(sum(not r.is_default for r in model.rules_), 'rule')}\n\n"
    )
    path.write_text(header + format_rules(model.rules_, model.ordered) + "\n")


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "n_rules",
    "avg_rule_width",
    "frac_overlap",
    "frac_uncovered",
    "frac_classes",
    "fit_seconds",
]


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds for successful runs."""
    ok = raw[raw["status"] == "ok"]
    cols = [c for c in METRIC_COLUMNS if c in ok.columns]
    grouped = ok.groupby(["dataset", "model"])[cols]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=0).add_suffix("_std"))
    summary["n_runs"] = ok.groupby(["dataset", "model"]).size()
    return summary.reset_index()
