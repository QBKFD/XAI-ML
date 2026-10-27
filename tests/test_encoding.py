import numpy as np
import pandas as pd
import pytest

from xai.datasets import get_spec
from xai.encoding import BinaryEncoder, Discretizer, NumericEncoder, make_encoder


def test_numeric_encoder_orders_and_one_hots():
    spec = get_spec("titanic")
    X = pd.DataFrame(
        {"Passenger_Cat": ["crew", "1st_class"], "Age_Cat": ["adult", "child"], "Gender": ["male", "female"]}
    )
    out = NumericEncoder(spec).fit_transform(X)
    assert list(out["Age_Cat"]) == [1.0, 0.0]
    assert list(out["Gender"]) == [1.0, 0.0]
    assert {"Passenger_Cat=crew", "Passenger_Cat=1st_class"} <= set(out.columns)
    assert out.dtypes.eq(float).all()


def test_numeric_encoder_unseen_category():
    train = pd.DataFrame({"c": ["a", "b"]})
    enc = NumericEncoder().fit(train)
    out = enc.transform(pd.DataFrame({"c": ["z"]}))
    assert out.to_numpy().sum() == 0  # unseen value -> all indicators off


def test_numeric_encoder_rejects_value_outside_order():
    spec = get_spec("cars")
    X = pd.DataFrame({"buying": ["cheap"]})
    with pytest.raises(ValueError, match="buying"):
        NumericEncoder(spec).fit(X)


def test_discretizer_learns_cut_points_from_training_data_only():
    train = pd.DataFrame({"v": np.arange(100.0)})
    test = pd.DataFrame({"v": [1000.0, -1000.0]})
    disc = Discretizer(n_bins=3).fit(train)
    edges = disc.edges_["v"]
    assert len(edges) == 2 and edges[0] < edges[1] < 99
    out = disc.transform(test)
    assert out["v"].tolist() == [f">{edges[1]:g}", f"<={edges[0]:g}"]


def test_discretizer_uses_fixed_bins_and_keeps_small_integers():
    spec = get_spec("cells")
    X = pd.DataFrame({"Mit": [1, 2, 10], "Other": [1, 2, 3]})
    out = Discretizer(spec).fit_transform(X)
    assert out["Mit"].tolist() == ["<=1.5", ">1.5", ">1.5"]
    assert out["Other"].tolist() == ["1", "2", "3"]


def test_binary_encoder_single_column_for_two_levels():
    X = pd.DataFrame({"g": ["m", "f", "m"], "c": ["a", "b", "c"]})
    out = BinaryEncoder().fit_transform(X)
    assert list(out.columns) == ["g=m", "c=a", "c=b", "c=c"]
    assert set(np.unique(out.to_numpy())) <= {0, 1}


def test_make_encoder_rejects_unknown_view():
    with pytest.raises(ValueError):
        make_encoder("ordinal")


def test_discretizer_handles_mostly_zero_columns():
    # Like glass's Ba column: tertiles are all 0, values above are rare.
    train = pd.DataFrame({"ba": [0.0] * 90 + [0.3, 0.6, 1.1, 1.5, 2.0, 0.9, 0.2, 0.7, 1.2, 3.1]})
    out = Discretizer(n_bins=3).fit_transform(train)
    assert set(out["ba"]) == {"<=0", ">0"}
