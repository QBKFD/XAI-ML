"""Common interface for all classifiers in the comparison."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from xai_rules.rules import Rule, simplify_binary_conditions


class NotApplicable(Exception):
    """Raised when a model cannot be used on a dataset (e.g. CORELS on multi-class)."""


class RuleModel(ClassifierMixin, BaseEstimator):
    """Base class: string labels in, string labels out.

    Subclasses implement ``_fit``, ``_predict_proba`` and, if they produce
    rules, ``_extract_rules``. Class probabilities are always returned with
    columns in the order of ``classes_``.
    """

    #: Short id used on the command line and in result files.
    key: ClassVar[str] = ""
    #: Name used in tables and figures.
    display_name: ClassVar[str] = ""
    #: Which encoding from :mod:`xai_rules.encoding` the model expects.
    view: ClassVar[str] = "numeric"
    #: ``True`` for rule lists (first matching rule wins).
    ordered: ClassVar[bool] = False
    #: Whether the model produces a readable rule set at all.
    is_rule_model: ClassVar[bool] = True
    binary_only: ClassVar[bool] = False
    #: Larger training sets are skipped (time / memory limits).
    max_train_rows: ClassVar[int | None] = None
    #: Python modules that must be importable.
    requires: ClassVar[tuple[str, ...]] = ()

    # -- availability -------------------------------------------------------

    @classmethod
    def missing_dependencies(cls) -> list[str]:
        return [m for m in cls.requires if importlib.util.find_spec(m) is None]

    @classmethod
    def check_applicable(cls, n_train: int, n_classes: int) -> None:
        missing = cls.missing_dependencies()
        if missing:
            raise NotApplicable(f"missing package(s): {', '.join(missing)}")
        if cls.binary_only and n_classes > 2:
            raise NotApplicable("binary classification only")
        if cls.max_train_rows is not None and n_train > cls.max_train_rows:
            raise NotApplicable(f"training set larger than {cls.max_train_rows} rows")

    # -- sklearn API --------------------------------------------------------

    def fit(self, X: pd.DataFrame, y) -> RuleModel:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} expects a pandas DataFrame")
        y = pd.Series(np.asarray(y).astype(str), index=X.index, name="target")
        self.classes_ = np.array(sorted(y.unique()))
        if len(self.classes_) < 2:
            raise ValueError("Need at least two classes to fit a classifier")
        self.check_applicable(len(X), len(self.classes_))
        self.feature_names_in_ = np.array(X.columns)
        self._fit(X, y)
        self.rules_ = self._extract_rules(X) if self.is_rule_model else None
        if self.rules_ is not None:
            self.rules_ = simplify_binary_conditions(self.rules_, X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.asarray(self._predict_proba(X[list(self.feature_names_in_)]), dtype=float)
        if proba.shape != (len(X), len(self.classes_)):
            raise RuntimeError(f"{self.key}: predict_proba returned shape {proba.shape}")
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.asarray(self._predict(X[list(self.feature_names_in_)])).astype(str)
        unknown = set(pred) - set(self.classes_)
        if unknown:
            raise RuntimeError(f"{self.key}: predicted unknown labels {sorted(unknown)}")
        return pred

    # -- to implement -------------------------------------------------------

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """Override when the library's own decision differs from argmax(proba)."""
        return self.classes_[np.asarray(self._predict_proba(X), dtype=float).argmax(axis=1)]

    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _extract_rules(self, X: pd.DataFrame) -> list[Rule] | None:
        return None


def one_hot(labels, classes) -> np.ndarray:
    """Hard predictions as a probability matrix."""
    labels = np.asarray(labels).astype(str)
    return (labels[:, None] == np.asarray(classes)[None, :]).astype(float)


@contextlib.contextmanager
def quiet():
    """Silence libraries that print progress to stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield
