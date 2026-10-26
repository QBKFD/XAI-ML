"""Tree baselines: Random Forest, CART and a C4.5-style tree.

All three run with library defaults, as in the thesis. CART and the C4.5-style
tree are also converted to rules (one rule per leaf), which makes their size
comparable to the rule learners.

About C4.5: the thesis used the ``c45-decision-tree`` package. That package
accumulates the split information across attributes instead of resetting it
for each one, and it treats every distinct numeric value as a separate
category. Both distort the gain ratio, which explains its very low score on
``glass`` (0.34 accuracy). Here C4.5 is approximated by a scikit-learn tree
with the entropy criterion and C4.5's default minimum of two cases per leaf.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from xai_rules.models.base import RuleModel
from xai_rules.rules import Condition, Rule


def tree_to_rules(tree: DecisionTreeClassifier, feature_names, classes) -> list[Rule]:
    t = tree.tree_
    rules: list[Rule] = []

    def walk(node: int, conds: list[Condition]) -> None:
        left, right = t.children_left[node], t.children_right[node]
        if left == right:  # leaf
            counts = t.value[node][0]
            rules.append(
                Rule(
                    conditions=list(conds),
                    prediction=str(classes[int(np.argmax(counts))]),
                    extra={"n_samples": int(t.n_node_samples[node])},
                )
            )
            return
        name = str(feature_names[t.feature[node]])
        thr = float(t.threshold[node])
        walk(left, conds + [Condition(name, "<=", thr)])
        walk(right, conds + [Condition(name, ">", thr)])

    walk(0, [])
    return rules


class _SklearnTreeModel(RuleModel):
    def _make(self):
        raise NotImplementedError

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model_ = self._make()
        self.model_.fit(X.to_numpy(dtype=float), y.to_numpy())

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model_.predict_proba(X.to_numpy(dtype=float))
        # sklearn orders classes the same way (sorted), but be explicit.
        order = [list(self.model_.classes_).index(c) for c in self.classes_]
        return proba[:, order]

    def _extract_rules(self, X: pd.DataFrame):
        return tree_to_rules(self.model_, X.columns, self.model_.classes_)


class CARTModel(_SklearnTreeModel):
    key = "cart"
    display_name = "CART"

    def __init__(self, random_state: int | None = 42, max_depth: int | None = None):
        self.random_state = random_state
        self.max_depth = max_depth

    def _make(self):
        return DecisionTreeClassifier(random_state=self.random_state, max_depth=self.max_depth)


class C45Model(_SklearnTreeModel):
    key = "c45"
    display_name = "C4.5"

    def __init__(self, random_state: int | None = 42, min_samples_leaf: int = 2):
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf

    def _make(self):
        return DecisionTreeClassifier(
            criterion="entropy",
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )


class RandomForestModel(_SklearnTreeModel):
    key = "random_forest"
    display_name = "Random Forest"
    is_rule_model = False

    def __init__(self, random_state: int | None = 42, n_estimators: int = 100, n_jobs: int | None = None):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def _make(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs
        )

    def _extract_rules(self, X):
        return None
