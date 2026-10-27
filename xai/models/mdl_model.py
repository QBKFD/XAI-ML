"""MDL rule lists (CLASSY).

Proença & van Leeuwen (2020), "Interpretable multiclass classification by
MDL-based rule lists". Package: https://github.com/HMProenca/RuleList

The method has no tuning parameters in the usual sense: the rule list is
chosen by the minimum description length principle. The arguments below only
limit the search (beam width, maximum rule length, number of cut points).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from xai import compat
from xai.models.base import RuleModel, quiet
from xai.rules import Condition, Rule


class MDLModel(RuleModel):
    key = "mdl"
    display_name = "MDL"
    view = "numeric"
    ordered = True
    requires = ("rulelist", "gmpy2", "numba")

    def __init__(
        self,
        max_depth: int = 5,
        beam_width: int = 100,
        min_support: int = 1,
        n_cutpoints: int = 5,
        discretization: str = "static",
    ):
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_support = min_support
        self.n_cutpoints = n_cutpoints
        self.discretization = discretization

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        compat.apply()
        from rulelist import RuleList

        self.model_ = RuleList(
            task="prediction",
            target_model="categorical",
            max_depth=self.max_depth,
            beam_width=self.beam_width,
            min_support=self.min_support,
            n_cutpoints=self.n_cutpoints,
            discretization=self.discretization,
        )
        with quiet():
            self.model_.fit(X.reset_index(drop=True), y.reset_index(drop=True))
        stats = self.model_._rulelist.default_rule_statistics
        self._target = next(iter(stats.usage_per_class))
        lib_order = [str(c) for c in stats.usage_per_class[self._target]]
        self._order = [lib_order.index(c) for c in self.classes_]

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.asarray(self.model_.predict_proba(X.reset_index(drop=True)), dtype=float)
        return proba[:, self._order]

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(X.reset_index(drop=True))).astype(str)

    def _prediction_of(self, statistics) -> str:
        counts = statistics.usage_per_class[self._target]
        return str(max(counts, key=counts.get))

    def _extract_rules(self, X: pd.DataFrame) -> list[Rule]:
        rl = self.model_._rulelist
        rules = []
        for subgroup in rl.subgroups:
            conds = []
            for item in subgroup.pattern:
                kw = item.activation_function.keywords
                name = kw["attribute_name"]
                if "category" in kw:
                    conds.append(Condition(name, "==", kw["category"]))
                    continue
                lo, hi = float(kw["minval"]), float(kw["maxval"])
                if not math.isinf(lo):
                    conds.append(Condition(name, ">=", lo))
                if not math.isinf(hi):
                    conds.append(Condition(name, "<", hi))
            rules.append(
                Rule(
                    conditions=conds,
                    prediction=self._prediction_of(subgroup.statistics),
                    extra={"usage": int(subgroup.usage)},
                )
            )
        rules.append(Rule(conditions=[], prediction=self._prediction_of(rl.default_rule_statistics)))
        return rules
