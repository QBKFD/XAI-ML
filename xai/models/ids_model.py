"""Interpretable Decision Sets (IDS).

Lakkaraju, Bach, Leskovec (2016). Implementation: pyIDS by Jiří Filip,
https://github.com/jirifilip/pyIDS (installed from GitHub, see
``scripts/install_legacy_deps.sh``).

IDS works on discrete data: candidate rules are mined as class association
rules, and the seven-term objective (size, length, overlap, coverage, ...) is
optimised with smooth local search. The weights ``lambdas`` of the seven terms
are tuned with coordinate ascent in ``scripts/tune.py``.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from xai import compat
from xai.models.base import RuleModel, quiet
from xai.rules import Condition, Rule

_TARGET = "__target__"


class IDSModel(RuleModel):
    key = "ids"
    display_name = "IDS"
    view = "categorical"
    ordered = False
    # pyIDS keeps a full cover cache per rule; above this size it ran out of
    # memory on an 8 GB machine during the thesis.
    max_train_rows = 5000
    requires = ("pyids", "pyarc", "fim")

    def __init__(
        self,
        rule_cutoff: int = 50,
        lambdas: tuple[float, ...] = (1, 1, 1, 1, 1, 1, 1),
        algorithm: str = "SLS",
        random_seed: int | None = 5,
    ):
        self.rule_cutoff = rule_cutoff
        self.lambdas = lambdas
        self.algorithm = algorithm
        self.random_seed = random_seed

    def _frame(self, X: pd.DataFrame, y=None):
        from pyarc.qcba.data_structures import QuantitativeDataFrame

        df = X.astype(str).reset_index(drop=True)
        df[_TARGET] = np.asarray(y) if y is not None else self.classes_[0]
        df[_TARGET] = df[_TARGET].astype(str)
        return df, QuantitativeDataFrame(df)

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        compat.apply()
        from pyids.algorithms.ids import IDS
        from pyids.algorithms.ids_classifier import mine_CARs

        if len(self.lambdas) != 7:
            raise ValueError("IDS needs exactly seven lambda weights")
        df, qdf = self._frame(X, y)
        if self.random_seed is not None:
            # pyIDS draws from the global generators. Together with the hash
            # patch in compat.py and a fixed PYTHONHASHSEED (set by the
            # scripts) this makes IDS runs repeatable.
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        with quiet():
            cars = mine_CARs(df, self.rule_cutoff, sample=False)
            self.model_ = IDS(algorithm=self.algorithm)
            self.model_.fit(
                quant_dataframe=qdf,
                class_association_rules=cars,
                lambda_array=list(self.lambdas),
                random_seed=self.random_seed,
            )

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        _, qdf = self._frame(X)
        with quiet():
            return np.asarray(self.model_.predict(qdf)).astype(str)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # pyIDS gives the confidence of the rule that fired (or of the default
        # rule). Spread the remaining mass evenly over the other classes.
        _, qdf = self._frame(X)
        with quiet():
            labels = np.asarray(self.model_.clf.predict(qdf)).astype(str)
            conf = np.asarray(self.model_.clf.predict_proba(qdf), dtype=float)
        k = len(self.classes_)
        proba = np.repeat(((1 - conf) / (k - 1))[:, None], k, axis=1)
        idx = np.searchsorted(self.classes_, labels)
        proba[np.arange(len(labels)), idx] = conf
        return proba

    def _extract_rules(self, X: pd.DataFrame) -> list[Rule]:
        rules = []
        for r in self.model_.clf.rules:
            car = r.car
            conds = [Condition(attr, "==", val) for attr, val in car.antecedent]
            rules.append(
                Rule(
                    conditions=conds,
                    prediction=str(car.consequent.value),
                    extra={"confidence": float(car.confidence), "support": float(car.support)},
                )
            )
        rules.sort(key=lambda r: -r.extra["support"])
        rules.append(Rule(conditions=[], prediction=str(self.model_.clf.default_class)))
        return rules
