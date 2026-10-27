"""Feature encodings for the different rule learners.

The algorithms in this study expect different inputs:

* ``numeric``      -- RuleOpt, MDL and the tree baselines. Ordered categories
                      become integers, unordered ones are one-hot encoded.
* ``categorical``  -- IDS mines association rules, so every feature must be a
                      discrete value. Numeric columns are cut into intervals.
* ``binary``       -- CORELS only accepts 0/1 features. Built on top of the
                      categorical view, one indicator per category.

All encoders learn their categories and cut points from the training split and
then apply them unchanged to the test split, so no test information leaks into
the preprocessing. (In the thesis notebooks discretisation was done on the full
file before splitting.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from xai.datasets import DatasetSpec

VIEWS = ("numeric", "categorical", "binary")


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _fmt(v: float) -> str:
    return f"{v:g}"


class NumericEncoder:
    """Ordinal-encode ordered categories, one-hot encode unordered ones."""

    def __init__(self, spec: DatasetSpec | None = None):
        self.spec = spec

    def fit(self, X: pd.DataFrame) -> NumericEncoder:
        ordinal = self.spec.ordinal if self.spec else {}
        self.plan_: list[tuple[str, str, Any]] = []
        for col in X.columns:
            if col in ordinal:
                order = [str(v) for v in ordinal[col]]
                unseen = set(X[col].astype(str)) - set(order)
                if unseen:
                    raise ValueError(f"Column {col!r} has values {sorted(unseen)} not in its ordinal order")
                self.plan_.append((col, "ordinal", {v: i for i, v in enumerate(order)}))
            elif _is_numeric(X[col]):
                self.plan_.append((col, "passthrough", None))
            else:
                cats = sorted(X[col].astype(str).unique())
                self.plan_.append((col, "onehot", cats))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for col, kind, info in self.plan_:
            if kind == "ordinal":
                # Values outside the known order get -1 rather than failing.
                out[col] = X[col].astype(str).map(info).fillna(-1).astype(float)
            elif kind == "passthrough":
                out[col] = X[col].astype(float)
            else:
                values = X[col].astype(str)
                for cat in info:
                    out[f"{col}={cat}"] = (values == cat).astype(float)
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class Discretizer:
    """Turn every column into a string category.

    Numeric columns with few distinct values are kept as categories. Others are
    cut at fixed points from the dataset spec, or else at training quantiles.
    """

    def __init__(self, spec: DatasetSpec | None = None, n_bins: int = 3, max_levels: int = 5):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.spec = spec
        self.n_bins = n_bins
        self.max_levels = max_levels

    def fit(self, X: pd.DataFrame) -> Discretizer:
        fixed = self.spec.bins if self.spec else {}
        self.edges_: dict[str, list[float] | None] = {}
        for col in X.columns:
            s = X[col]
            if not _is_numeric(s):
                self.edges_[col] = None
            elif col in fixed:
                self.edges_[col] = sorted(float(v) for v in fixed[col])
            elif s.nunique() <= self.max_levels:
                self.edges_[col] = None
            else:
                qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                edges = sorted(set(np.round(s.quantile(qs).to_numpy(dtype=float), 6)))
                # Drop cut points that would leave the first or last bin empty.
                edges = [e for e in edges if s.min() < e < s.max()]
                if not edges:
                    # Skewed column (e.g. mostly zeros): the quantiles collapse
                    # onto the minimum. Split "at the minimum" vs "above it".
                    edges = [float(s.min())]
                self.edges_[col] = edges
        return self

    @staticmethod
    def _labels(edges: list[float]) -> list[str]:
        labels = [f"<={_fmt(edges[0])}"]
        labels += [f"({_fmt(lo)},{_fmt(hi)}]" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
        labels.append(f">{_fmt(edges[-1])}")
        return labels

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, edges in self.edges_.items():
            s = X[col]
            if edges is None:
                if _is_numeric(s) and (s.dropna() % 1 == 0).all():
                    out[col] = s.astype("Int64").astype(str)
                else:
                    out[col] = s.astype(str)
            else:
                bins = [-np.inf, *edges, np.inf]
                cut = pd.cut(s.astype(float), bins=bins, labels=self._labels(edges))
                out[col] = cut.astype(str)
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class BinaryEncoder:
    """One 0/1 column per category of a discretised frame.

    Two-level features get a single indicator so CORELS does not see the same
    information twice.
    """

    def __init__(self, spec: DatasetSpec | None = None, n_bins: int = 3):
        self.discretizer = Discretizer(spec, n_bins=n_bins)

    def fit(self, X: pd.DataFrame) -> BinaryEncoder:
        D = self.discretizer.fit_transform(X)
        self.categories_: dict[str, list[str]] = {}
        for col in D.columns:
            cats = sorted(D[col].unique())
            self.categories_[col] = cats[1:] if len(cats) == 2 else cats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        D = self.discretizer.transform(X)
        out = {}
        for col, cats in self.categories_.items():
            for cat in cats:
                out[f"{col}={cat}"] = (D[col] == cat).astype(np.uint8)
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def make_encoder(view: str, spec: DatasetSpec | None = None):
    if view == "numeric":
        return NumericEncoder(spec)
    if view == "categorical":
        return Discretizer(spec)
    if view == "binary":
        return BinaryEncoder(spec)
    raise ValueError(f"Unknown view {view!r}, expected one of {VIEWS}")
