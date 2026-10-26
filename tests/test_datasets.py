import pytest

from xai_rules.datasets import DATASETS, dataset_summary, get_spec, load_dataset

# Row counts from Table 1 of the thesis.
EXPECTED_ROWS = {
    "cells": 699,
    "cars": 1728,
    "glass": 214,
    "tictactoe": 958,
    "titanic": 1761,
    "doctor": 714,
    "sonar": 208,
    "ionosphere": 315,
    "compas": 7214,
    "liver": 345,
    "drybeans": 13611,
}


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_every_dataset_loads(name):
    X, y, spec = load_dataset(name)
    assert len(X) == len(y) == EXPECTED_ROWS[name]
    assert spec.target not in X.columns
    assert y.nunique() >= 2
    assert y.isna().sum() == 0
    assert X.isna().sum().sum() == 0


def test_tictactoe_keeps_first_row():
    # The raw file has no header; reading it with a header used to drop a game.
    X, _, _ = load_dataset("tictactoe")
    assert list(X.iloc[0]) == ["x", "x", "x", "x", "o", "o", "x", "o", "o"]


def test_target_labels_are_readable():
    _, y, _ = load_dataset("cells")
    assert set(y) == {"benign", "malignant"}


def test_unknown_dataset_lists_known_names():
    with pytest.raises(KeyError, match="cells"):
        get_spec("iris")


def test_summary_has_one_row_per_dataset():
    assert len(dataset_summary()) == len(DATASETS)
