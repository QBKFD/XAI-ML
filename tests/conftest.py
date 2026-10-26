import pandas as pd
import pytest

from xai_rules.datasets import load_dataset


@pytest.fixture(scope="session")
def titanic():
    X, y, spec = load_dataset("titanic")
    return X, y, spec


@pytest.fixture
def toy():
    """Small numeric frame where class 'b' is exactly x > 5."""
    X = pd.DataFrame({"x": [1, 2, 3, 4, 6, 7, 8, 9] * 5, "z": [0, 1] * 20})
    y = pd.Series(["b" if v > 5 else "a" for v in X["x"]])
    return X, y
