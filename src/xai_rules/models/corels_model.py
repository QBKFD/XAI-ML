"""CORELS: certifiably optimal rule lists (binary features, binary target).

Angelino, Larus-Stone, Alabi, Seltzer, Rudin (2018). Package: ``corels``
(https://github.com/corels/pycorels). The PyPI source release does not build
on Python >= 3.11; ``scripts/install_legacy_deps.sh`` applies a two-line fix.

CORELS only outputs hard predictions. To get a score for ROC AUC, each rule is
given the share of positive training rows it captures (Laplace smoothed),
the usual way to read probabilities off a rule list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xai_rules.models.base import RuleModel
from xai_rules.rules import Condition, Rule


class CORELSModel(RuleModel):
    key = "corels"
    display_name = "CORELS"
    view = "binary"
    ordered = True
    binary_only = True
    requires = ("corels",)

    def __init__(self, c: float = 0.01, n_iter: int = 10000, max_card: int = 2, min_support: float = 0.01):
        self.c = c
        self.n_iter = n_iter
        self.max_card = max_card
        self.min_support = min_support

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from corels import CorelsClassifier

        self.model_ = CorelsClassifier(
            c=self.c,
            n_iter=self.n_iter,
            max_card=self.max_card,
            min_support=self.min_support,
            verbosity=[],
        )
        Xb = X.to_numpy(dtype=np.uint8)
        yb = (y.to_numpy() == self.classes_[1]).astype(np.uint8)
        self.model_.fit(Xb, yb, features=list(X.columns), prediction_name="positive")

        # Per-rule positive rate on the training data, in list order.
        captured = self._capture(X)
        self.rule_scores_ = np.array(
            [
                (yb[captured == i].sum() + 1) / ((captured == i).sum() + 2)
                for i in range(len(self._raw_rules()))
            ]
        )

    def _raw_rules(self):
        return self.model_.rl().rules

    def _antecedent_mask(self, X: pd.DataFrame, antecedents) -> np.ndarray:
        if antecedents == [0]:
            return np.ones(len(X), dtype=bool)
        mask = np.ones(len(X), dtype=bool)
        cols = X.columns
        for a in antecedents:
            values = X[cols[abs(a) - 1]].to_numpy()
            mask &= values == (1 if a > 0 else 0)
        return mask

    def _capture(self, X: pd.DataFrame) -> np.ndarray:
        """Index of the rule that classifies each row (first match wins)."""
        out = np.full(len(X), -1)
        for i, rule in enumerate(self._raw_rules()):
            hit = (out == -1) & self._antecedent_mask(X, rule["antecedents"])
            out[hit] = i
        return out

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.asarray(self.model_.predict(X.to_numpy(dtype=np.uint8))).astype(bool)
        return np.where(pred, self.classes_[1], self.classes_[0])

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = self.rule_scores_[self._capture(X)]
        return np.column_stack([1 - p1, p1])

    def _extract_rules(self, X: pd.DataFrame) -> list[Rule]:
        cols = list(X.columns)
        rules = []
        for rule in self._raw_rules():
            pred = self.classes_[1] if bool(rule["prediction"]) else self.classes_[0]
            if rule["antecedents"] == [0]:
                rules.append(Rule(conditions=[], prediction=str(pred)))
                continue
            conds = [Condition(cols[abs(a) - 1], "==", 1 if a > 0 else 0) for a in rule["antecedents"]]
            rules.append(Rule(conditions=conds, prediction=str(pred)))
        return rules
