"""RuleOpt (RUG): rule generation with linear programming and column generation.

Lumadjeng, Röber, Akyüz, Birbil (2023). Package: https://github.com/sametcopur/ruleopt

Compared with the thesis this uses ruleopt >= 1.2, which ships the open-source
HiGHS solver (the thesis used OR-Tools) and no longer forwards
``max_features``, ``max_leaf_nodes`` or ``min_impurity_decrease`` to the
underlying trees. The remaining hyperparameters keep their meaning.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from xai_rules.models.base import RuleModel
from xai_rules.rules import Condition, Rule

RULE_COSTS = ("gini", "length")


class RuleOptModel(RuleModel):
    key = "ruleopt"
    display_name = "RuleOpt"
    view = "numeric"
    ordered = False
    requires = ("ruleopt",)

    def __init__(
        self,
        max_depth: int = 3,
        max_rmp_calls: int = 10,
        rule_cost: str = "gini",
        threshold: float = 1e-6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = 42,
    ):
        self.max_depth = max_depth
        self.max_rmp_calls = max_rmp_calls
        self.rule_cost = rule_cost
        self.threshold = threshold
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _make(self):
        from ruleopt import RUGClassifier
        from ruleopt.rule_cost import Gini, Length

        if self.rule_cost not in RULE_COSTS:
            raise ValueError(f"rule_cost must be one of {RULE_COSTS}, got {self.rule_cost!r}")
        cost = Gini() if self.rule_cost == "gini" else Length()
        return RUGClassifier(
            rule_cost=cost,
            max_depth=self.max_depth,
            max_rmp_calls=self.max_rmp_calls,
            threshold=self.threshold,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # RuleOpt requires labels 0..k-1.
        self._index = {c: i for i, c in enumerate(self.classes_)}
        self.model_ = self._make()
        self.model_.fit(X.to_numpy(dtype=float), y.map(self._index).to_numpy())

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X.to_numpy(dtype=float))

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        # RuleOpt's own decision (weighted vote of the rules that fire) is
        # used for accuracy, as in the thesis. On multi-class data it can
        # differ from the argmax of predict_proba, which only feeds ROC AUC.
        return self.classes_[np.asarray(self.model_.predict(X.to_numpy(dtype=float)), dtype=int)]

    def _extract_rules(self, X: pd.DataFrame) -> list[Rule]:
        names = list(X.columns)
        rules = []
        for r in self.model_.decision_rules_.values():
            conds = []
            for feat, bounds in sorted(r.to_dict().items()):
                lb, ub = float(bounds["lb"]), float(bounds["ub"])
                if not math.isinf(lb):
                    conds.append(Condition(names[feat], ">", lb))
                if not math.isinf(ub):
                    conds.append(Condition(names[feat], "<=", ub))
            rules.append(
                Rule(conditions=conds, prediction=str(self.classes_[int(r.label)]), weight=float(r.weight))
            )
        rules.sort(key=lambda r: -(r.weight or 0.0))
        return rules
