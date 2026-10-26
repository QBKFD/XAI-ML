"""Compatibility patches for unmaintained rule-learning packages.

``rulelist`` (MDL) and ``pyids``/``pyarc`` (IDS) were last released in 2020-2021
and call pandas/numpy APIs that have since been removed. During the thesis this
was solved by pinning ``pandas==1.5.3``. That pin no longer installs cleanly on
recent Python versions, so instead we restore the handful of removed names.

Every patch only *adds* a missing attribute; nothing that exists is replaced,
except the ``interpolation=`` keyword on the quantile functions, which is
translated to its modern name ``method=``. Calling :func:`apply` more than once
is safe.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

_APPLIED = False

# Aliases removed in NumPy 2.0 that the legacy packages still reference.
_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "object": object,
    "NINF": -np.inf,
    "PINF": np.inf,
    "Inf": np.inf,
    "Infinity": np.inf,
    "infty": np.inf,
    "NaN": np.nan,
    "NAN": np.nan,
    "float_": np.float64,
    "int_": np.int64,
    "product": np.prod,
    "cumproduct": np.cumprod,
    "alltrue": np.all,
    "sometrue": np.any,
}

_QUANTILE_FUNCS = ("quantile", "nanquantile", "percentile", "nanpercentile")


def _accept_interpolation_kwarg(func):
    if getattr(func, "_xai_rules_patched", False):
        return func

    @functools.wraps(func)
    def wrapper(*args, interpolation=None, **kwargs):
        if interpolation is not None:
            kwargs.setdefault("method", interpolation)
        return func(*args, **kwargs)

    wrapper._xai_rules_patched = True
    return wrapper


def apply() -> None:
    """Install the patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    for cls in (pd.DataFrame, pd.Series):
        if not hasattr(cls, "iteritems"):
            cls.iteritems = cls.items

    for name, value in _NUMPY_ALIASES.items():
        # numpy raises from a module-level __getattr__ for removed names, so
        # check the module dict instead of using hasattr().
        if name not in np.__dict__:
            setattr(np, name, value)

    for name in _QUANTILE_FUNCS:
        setattr(np, name, _accept_interpolation_kwarg(getattr(np, name)))

    _APPLIED = True
