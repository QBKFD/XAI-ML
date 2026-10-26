"""Every model honours the common interface. Slow learners are marked."""

import numpy as np
import pandas as pd
import pytest

from xai_rules.experiment import make_split, run_one
from xai_rules.models import MODELS, NotApplicable, get_model

FAST = ["ruleopt", "cart", "c45", "random_forest", "corels"]
SLOW = ["mdl", "ids"]


def _require(key):
    missing = MODELS[key].missing_dependencies()
    if missing:
        pytest.skip(f"{key} not installed ({', '.join(missing)})")


@pytest.fixture(scope="module")
def titanic_split():
    return make_split("titanic", seed=42)


def _check_fitted(model, split):
    X_tr, X_te = split.view(model.view)
    model.fit(X_tr, split.y_train)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)

    assert list(model.classes_) == sorted(split.y_train.unique())
    assert len(pred) == len(X_te) and set(pred) <= set(model.classes_)
    assert proba.shape == (len(X_te), len(model.classes_))
    assert np.all(proba >= -1e-9) and np.all(proba <= 1 + 1e-9)
    # Sanity check only: 67% of titanic passengers died. IDS with default
    # weights hovers around that, the other models reach about 76%.
    assert (pred == split.y_test.to_numpy()).mean() > 0.60
    if model.is_rule_model:
        assert model.rules_ and all(r.prediction in model.classes_ for r in model.rules_)
    else:
        assert model.rules_ is None


@pytest.mark.parametrize("key", FAST)
def test_fast_models(key, titanic_split):
    _require(key)
    _check_fitted(get_model(key), titanic_split)


@pytest.mark.slow
@pytest.mark.parametrize("key", SLOW)
def test_slow_models(key, titanic_split):
    _require(key)
    _check_fitted(get_model(key), titanic_split)


@pytest.mark.slow
def test_ids_metrics_match_pyids(titanic_split):
    """Our metric code reproduces pyIDS's own numbers for an IDS model."""
    _require("ids")
    from pyarc.qcba.data_structures import QuantitativeDataFrame

    from xai_rules.metrics import interpretability_metrics

    model = get_model("ids", rule_cutoff=20)
    X_tr, X_te = titanic_split.view("categorical")
    model.fit(X_tr, titanic_split.y_train)
    df = X_te.astype(str).reset_index(drop=True)
    df["__target__"] = titanic_split.y_test.to_numpy()
    theirs = model.model_.score_interpretability_metrics(QuantitativeDataFrame(df))
    ours = interpretability_metrics(model.rules_, X_te, titanic_split.classes)

    assert ours["n_rules"] == theirs["ruleset_length"]
    assert ours["avg_rule_width"] == pytest.approx(theirs["average_rule_width"])
    assert ours["frac_overlap"] == pytest.approx(theirs["fraction_overlap"])
    assert ours["frac_uncovered"] == pytest.approx(theirs["fraction_uncovered"])
    assert ours["frac_classes"] == pytest.approx(theirs["fraction_classes"])


def test_model_with_rules_from_cart_reproduces_its_predictions(titanic_split):
    model = get_model("cart")
    X_tr, X_te = titanic_split.view("numeric")
    model.fit(X_tr, titanic_split.y_train)
    # Leaves partition the space: each row is covered by exactly one rule,
    # and that rule's label is the tree's prediction.
    cover = np.vstack([r.covers(X_te) for r in model.rules_])
    assert (cover.sum(axis=0) == 1).all()
    labels = np.array([r.prediction for r in model.rules_])[cover.argmax(axis=0)]
    assert (labels == model.predict(X_te)).all()


def test_corels_is_skipped_on_multiclass():
    _require("corels")
    row, model = run_one(make_split("glass", seed=0), "corels")
    assert row["status"] == "skipped" and model is None
    with pytest.raises(NotApplicable):
        get_model("corels").fit(pd.DataFrame({"a": [0, 1, 1]}), ["x", "y", "z"])


def test_run_one_records_failures_instead_of_raising(titanic_split):
    row, model = run_one(titanic_split, "ruleopt", {"rule_cost": "entropy"})
    assert row["status"] == "failed" and "rule_cost" in row["message"]
    assert model is None


def test_unknown_model():
    with pytest.raises(KeyError, match="ruleopt"):
        get_model("xgboost")


@pytest.mark.slow
def test_ids_is_repeatable(titanic_split):
    """Same data, same seed, same rules (needs the pyarc hash patch in compat)."""
    _require("ids")
    from xai_rules.rules import format_rules

    X_tr, _ = titanic_split.view("categorical")
    a = get_model("ids", rule_cutoff=20).fit(X_tr, titanic_split.y_train)
    b = get_model("ids", rule_cutoff=20).fit(X_tr, titanic_split.y_train)
    assert format_rules(a.rules_, False) == format_rules(b.rules_, False)
