"""Predictive and interpretability metrics.

Interpretability metrics follow the definitions from the IDS paper
(Lakkaraju et al., 2016) that were used in the thesis. They are computed on the
non-default rules only, using each rule's own antecedent coverage. For rule
lists (MDL, CORELS) that means a rule is treated as if it stood on its own,
which is how the thesis applied the IDS metrics to MDL.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xai.rules import Rule


def classification_metrics(
    y_true, y_pred, proba: np.ndarray | None = None, classes: list[str] | None = None
) -> dict[str, float]:
    """Accuracy, balanced accuracy, macro precision/recall/F1 and ROC AUC.

    Macro averaging is used for every dataset so binary and multi-class
    results are on the same footing. ROC AUC is one-vs-rest, weighted by class
    support, for multi-class problems; it is NaN when the model gives no
    probabilities or the test split misses a class.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        out = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
    out["roc_auc"] = _roc_auc(y_true, proba, classes)
    return out


def _roc_auc(y_true: np.ndarray, proba: np.ndarray | None, classes: list[str] | None) -> float:
    if proba is None or classes is None:
        return float("nan")
    proba = np.asarray(proba, dtype=float)
    present = set(y_true)
    if len(present) < 2 or not present <= set(classes):
        return float("nan")
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y_true == classes[1], proba[:, 1]))
        # Renormalise in case a model returns rows that do not sum to one.
        proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
        return float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=list(classes))
        )
    except ValueError:
        return float("nan")


def interpretability_metrics(
    rules: list[Rule] | None, X: pd.DataFrame, classes: list[str]
) -> dict[str, float]:
    """IDS-style interpretability metrics for a learned rule model.

    ``n_rules``            number of non-default rules
    ``avg_rule_width``     mean number of features per rule
    ``frac_overlap``       mean pairwise overlap between rules, as a share of rows
    ``frac_uncovered``     share of rows not covered by any rule
    ``frac_classes``       share of classes predicted by at least one rule
                           (including the default rule, as in pyIDS)
    """
    nan = float("nan")
    if rules is None:
        return dict.fromkeys(
            ["n_rules", "avg_rule_width", "frac_overlap", "frac_uncovered", "frac_classes"], nan
        )

    body = [r for r in rules if not r.is_default]
    n = len(X)
    k = len(body)
    predicted = {str(r.prediction) for r in rules}

    if k == 0 or n == 0:
        return {
            "n_rules": float(k),
            "avg_rule_width": 0.0,
            "frac_overlap": 0.0,
            "frac_uncovered": 1.0 if n else nan,
            "frac_classes": len(predicted & set(classes)) / len(classes),
        }

    cover = np.vstack([r.covers(X) for r in body]).astype(np.int32)  # k x n
    if k > 1:
        pair_overlap = cover @ cover.T  # rows covered by both i and j
        upper = pair_overlap[np.triu_indices(k, 1)].sum()
        frac_overlap = 2.0 / (k * (k - 1)) * upper / n
    else:
        frac_overlap = 0.0

    return {
        "n_rules": float(k),
        "avg_rule_width": float(np.mean([r.width for r in body])),
        "frac_overlap": float(frac_overlap),
        "frac_uncovered": float(1.0 - cover.any(axis=0).mean()),
        "frac_classes": len(predicted & set(classes)) / len(classes),
    }
