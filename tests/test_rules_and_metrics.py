import math

import numpy as np
import pandas as pd
import pytest

from xai_rules.metrics import classification_metrics, interpretability_metrics
from xai_rules.rules import Condition, Rule, format_rules

X = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})


def test_condition_operators():
    assert Condition("a", ">", 2).mask(X).tolist() == [False, False, True, True]
    assert Condition("a", "<=", 2).mask(X).tolist() == [True, True, False, False]
    assert Condition("b", "==", "x").mask(X).tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        Condition("a", "~", 1)


def test_rule_width_counts_features_not_conditions():
    rule = Rule([Condition("a", ">", 1), Condition("a", "<=", 3), Condition("b", "==", "x")], "pos")
    assert rule.width == 2
    assert rule.covers(X).tolist() == [False, False, True, False]


def test_default_rule_covers_everything():
    assert Rule([], "neg").covers(X).all()


def test_indicator_conditions_print_cleanly():
    assert str(Condition("Gender=male", "==", 1)) == "Gender=male"
    assert str(Condition("Gender=male", "==", 0)) == "NOT Gender=male"


def test_format_rule_list():
    text = format_rules([Rule([Condition("a", ">", 2)], "p"), Rule([], "n")], ordered=True)
    assert text.splitlines() == ["IF a > 2 THEN p", "ELSE PREDICT n"]


def test_interpretability_metrics_by_hand():
    rules = [
        Rule([Condition("a", ">", 1)], "pos"),  # rows 1,2,3
        Rule([Condition("b", "==", "x")], "neg"),  # rows 0,2
        Rule([Condition("a", ">", 10)], "pos"),  # none
        Rule([], "neg"),  # default, ignored
    ]
    m = interpretability_metrics(rules, X, ["neg", "pos", "other"])
    assert m["n_rules"] == 3
    assert m["avg_rule_width"] == 1
    # overlaps: (r1,r2)=1 row, (r1,r3)=0, (r2,r3)=0 -> 2/(3*2) * 1/4
    assert math.isclose(m["frac_overlap"], 2 / 6 * 1 / 4)
    assert m["frac_uncovered"] == 0.0
    assert math.isclose(m["frac_classes"], 2 / 3)


def test_interpretability_metrics_without_rules():
    m = interpretability_metrics(None, X, ["a", "b"])
    assert all(np.isnan(v) for v in m.values())


def test_classification_metrics_binary_and_multiclass():
    y = np.array(["a", "b", "a", "b"])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    m = classification_metrics(y, ["a", "b", "a", "b"], proba, ["a", "b"])
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0

    y3 = np.array(["a", "b", "c", "a", "b", "c"])
    p3 = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    m3 = classification_metrics(y3, y3, p3, ["a", "b", "c"])
    assert m3["f1"] == 1.0 and m3["roc_auc"] == 1.0


def test_roc_auc_is_nan_without_probabilities():
    assert np.isnan(classification_metrics(["a", "b"], ["a", "b"])["roc_auc"])


def test_format_rule_set_default():
    text = format_rules([Rule([Condition("a", ">", 2)], "p"), Rule([], "n")], ordered=False)
    assert text.splitlines() == ["IF a > 2 THEN p", "OTHERWISE PREDICT n"]


def test_interval_categories_print_naturally():
    assert str(Condition("NormNucl", "==", "<=2.5")) == "NormNucl <= 2.5"
    assert str(Condition("Clump", "==", ">6.5")) == "Clump > 6.5"
    assert str(Condition("Clump", "==", "(4.5,6.5]")) == "4.5 < Clump <= 6.5"


def test_binary_threshold_conditions_are_collapsed():
    from xai_rules.rules import simplify_binary_conditions

    Xb = pd.DataFrame({"g=m": [0, 1, 0, 1], "age": [10.0, 20.0, 30.0, 40.0]})
    rules = [
        Rule([Condition("g=m", ">=", 0), Condition("g=m", "<", 1), Condition("age", ">", 15)], "p"),
        Rule([Condition("g=m", ">", 0.5)], "q"),
        Rule([Condition("g=m", ">=", 0)], "r"),  # always true -> dropped
    ]
    out = simplify_binary_conditions(rules, Xb)
    assert [str(r) for r in out] == [
        "IF age > 15 AND NOT g=m THEN p",
        "IF g=m THEN q",
        "DEFAULT THEN r",
    ]
    for before, after in zip(rules, out, strict=True):
        assert (before.covers(Xb) == after.covers(Xb)).all()


def test_equality_on_float_column_with_int_value():
    Xf = pd.DataFrame({"f": [0.0, 1.0, 1.0]})
    assert Condition("f", "==", 1).mask(Xf).tolist() == [False, True, True]
