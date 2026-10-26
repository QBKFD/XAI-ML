import numpy as np
import pandas as pd

from xai_rules import compat


def test_apply_is_idempotent_and_restores_removed_names():
    compat.apply()
    compat.apply()
    assert np.NINF == -np.inf
    assert hasattr(pd.DataFrame, "iteritems")
    assert np.nanquantile([1.0, 2.0, 3.0], 0.5, interpolation="midpoint") == 2.0
    # The patched function still works without the old keyword.
    assert np.quantile([1.0, 3.0], 0.5) == 2.0
