"""A small, library-independent representation of learned rules.

Every rule learner in this project returns its rules in a different format.
They are converted to :class:`Rule` objects so that rule counts, rule widths,
coverage and overlap are computed by the same code for every method.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: Any

    def __post_init__(self):
        if self.op not in _OPS:
            raise ValueError(f"Unsupported operator {self.op!r}")

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        col = X[self.feature]
        if self.op in ("==", "!="):
            numeric_value = isinstance(self.value, (int, float, np.number)) and not isinstance(
                self.value, bool
            )
            if numeric_value and pd.api.types.is_numeric_dtype(col):
                return _OPS[self.op](col.to_numpy(dtype=float), float(self.value))
            # Otherwise compare as text: discretised data is stored as strings.
            return _OPS[self.op](col.astype(str).to_numpy(), str(self.value))
        return _OPS[self.op](col.to_numpy(dtype=float), float(self.value))

    def __str__(self) -> str:
        v = self.value
        # Indicator columns from the binary encoding are named "feature=value".
        if self.op == "==" and "=" in self.feature and str(v) in ("0", "1"):
            return self.feature if str(v) == "1" else f"NOT {self.feature}"
        # Interval categories from the discretiser: "<=2.5", ">4.5", "(1.5,4.5]".
        text = str(v)
        if self.op == "==" and text.startswith(("<=", ">")):
            return f"{self.feature} {text[:2] if text.startswith('<=') else '>'} {text.lstrip('<=>')}"
        if self.op == "==" and text.startswith("(") and text.endswith("]") and "," in text:
            lo, hi = text[1:-1].split(",", 1)
            return f"{lo} < {self.feature} <= {hi}"
        if isinstance(v, float):
            v = f"{v:g}"
        return f"{self.feature} {self.op} {v}"


@dataclass
class Rule:
    """``IF all conditions THEN prediction``.

    A rule with no conditions is a default rule (it covers everything).
    """

    conditions: list[Condition]
    prediction: str
    weight: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        return not self.conditions

    @property
    def width(self) -> int:
        """Number of distinct features the rule tests.

        ``2 < x <= 5`` counts once, which matches how RuleOpt and pyIDS report
        rule length.
        """
        return len({c.feature for c in self.conditions})

    def covers(self, X: pd.DataFrame) -> np.ndarray:
        out = np.ones(len(X), dtype=bool)
        for cond in self.conditions:
            out &= cond.mask(X)
        return out

    def __str__(self) -> str:
        if self.is_default:
            body = "DEFAULT"
        else:
            body = "IF " + " AND ".join(str(c) for c in self.conditions)
        text = f"{body} THEN {self.prediction}"
        if self.weight is not None:
            text += f"  (weight {self.weight:.3g})"
        return text


def simplify_binary_conditions(rules: list[Rule], X: pd.DataFrame) -> list[Rule]:
    """Rewrite threshold tests on 0/1 columns as a single equality.

    Learners that treat one-hot columns as numbers produce conditions such as
    ``x >= 0 AND x < 1``; this turns them into ``x == 0``. A condition that
    every 0/1 value satisfies is dropped.
    """
    binary = {
        c
        for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c]) and set(np.unique(X[c].to_numpy())) <= {0, 1}
    }
    out = []
    for rule in rules:
        keep, by_feature = [], {}
        for cond in rule.conditions:
            if cond.feature in binary and cond.op not in ("==", "!="):
                by_feature.setdefault(cond.feature, []).append(cond)
            else:
                keep.append(cond)
        for feat, conds in by_feature.items():
            allowed = [v for v in (0, 1) if all(_OPS[c.op](v, float(c.value)) for c in conds)]
            if len(allowed) == 1:
                keep.append(Condition(feat, "==", allowed[0]))
            elif not allowed:  # contradictory; keep the original tests
                keep.extend(conds)
        out.append(Rule(keep, rule.prediction, rule.weight, rule.extra))
    return out


def format_rules(rules: list[Rule], ordered: bool) -> str:
    """Readable text for a rule set (unordered) or rule list (ordered)."""
    lines = []
    for i, rule in enumerate(rules):
        if rule.is_default:
            word = "ELSE" if ordered else "OTHERWISE"
            lines.append(f"{word} PREDICT {rule.prediction}" if rules[:i] else f"PREDICT {rule.prediction}")
        elif ordered and i > 0:
            lines.append("ELSE " + str(rule))
        else:
            lines.append(str(rule))
    return "\n".join(lines)
