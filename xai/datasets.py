"""Dataset registry and loading.

Each dataset is described once by a :class:`DatasetSpec`. ``load_dataset``
returns the raw features and the target; turning those into the inputs a
specific algorithm needs (numeric, discretised, binary) is done afterwards by
:mod:`xai.encoding`, fitted on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

_LEVELS_4 = ["low", "med", "high", "vhigh"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    file: str
    target: str
    description: str
    source: str
    column_names: list[str] | None = None
    # Ordered categories for features that have a natural order. Used by the
    # numeric encoding; unordered string columns are one-hot encoded instead.
    ordinal: dict[str, list[Any]] = field(default_factory=dict)
    # Fixed cut points for discretisation. Columns without an entry are
    # discretised with training-set quantiles.
    bins: dict[str, list[float]] = field(default_factory=dict)
    # Optional relabelling of the target, purely for readable rules.
    target_labels: dict[Any, str] = field(default_factory=dict)
    in_thesis_tables: bool = True


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        DatasetSpec(
            name="cells",
            file="cells.csv",
            target="Class",
            description="Breast cancer cytology, benign vs. malignant",
            source="UCI Breast Cancer Wisconsin (Original)",
            target_labels={2: "benign", 4: "malignant"},
            # Cut points chosen in the thesis (notebooks/thesis/preprocessing.ipynb).
            bins={
                "UnifSize": [1.5, 4.5],
                "MargAdh": [1.5, 3.5],
                "Clump": [4.5, 6.5],
                "BlandChrom": [2.5, 3.5],
                "UnifShape": [1.5, 4.5],
                "NormNucl": [2.5, 9.5],
                "Mit": [1.5],
                "BareNuc": [1.5, 3.765, 5.5],
                "SingEpiSize": [2.5, 3.5],
            },
        ),
        DatasetSpec(
            name="cars",
            file="cars.csv",
            target="class",
            description="Car acceptability from price and technical features",
            source="UCI Car Evaluation",
            ordinal={
                "buying": _LEVELS_4,
                "maint": _LEVELS_4,
                "doors": ["2", "3", "4", "5more"],
                "persons": ["2", "4", "more"],
                "lug_boot": ["small", "med", "big"],
                "safety": ["low", "med", "high"],
            },
        ),
        DatasetSpec(
            name="glass",
            file="glass.csv",
            target="Type",
            description="Glass type from chemical composition",
            source="UCI Glass Identification",
            column_names=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"],
        ),
        DatasetSpec(
            name="tictactoe",
            file="tictactoe.csv",
            target="outcome",
            description="Tic-tac-toe end games, does x win",
            source="UCI Tic-Tac-Toe Endgame",
            column_names=[
                "top_left",
                "top_middle",
                "top_right",
                "middle_left",
                "middle_middle",
                "middle_right",
                "bottom_left",
                "bottom_middle",
                "bottom_right",
                "outcome",
            ],
        ),
        DatasetSpec(
            name="titanic",
            file="titanic.csv",
            target="Died",
            description="Titanic passengers, survived vs. died",
            source="Titanic (version distributed with pyIDS)",
            ordinal={"Age_Cat": ["child", "adult"], "Gender": ["female", "male"]},
            target_labels={0: "survived", 1: "died"},
        ),
        DatasetSpec(
            name="doctor",
            file="doctor.csv",
            target="Number of Doctors Visited",
            description="Number of doctors visited by older adults",
            source="UCI National Poll on Healthy Aging",
        ),
        DatasetSpec(
            name="sonar",
            file="sonar.csv",
            target="class",
            description="Sonar returns, mine vs. rock",
            source="UCI Connectionist Bench (Sonar)",
        ),
        DatasetSpec(
            name="ionosphere",
            file="ionosphere.csv",
            target="class",
            description="Radar returns, good vs. bad (pre-discretised)",
            source="UCI Ionosphere",
        ),
        DatasetSpec(
            name="compas",
            file="compas.csv",
            target="Recidivate-Within-Two-Years",
            description="Two-year recidivism (pre-binarised)",
            source="ProPublica COMPAS",
            in_thesis_tables=False,
        ),
        DatasetSpec(
            name="liver",
            file="liver.csv",
            target="y",
            description="Liver disorders (pre-binarised)",
            source="UCI Liver Disorders",
            in_thesis_tables=False,
        ),
        DatasetSpec(
            name="drybeans",
            file="drybeans.csv",
            target="Class",
            description="Dry bean variety from shape measurements",
            source="UCI Dry Bean",
        ),
    ]
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset {name!r}. Known datasets: {known}") from None


def load_dataset(
    name: str, data_dir: str | Path | None = None
) -> tuple[pd.DataFrame, pd.Series, DatasetSpec]:
    """Load a dataset by name.

    Returns ``(X, y, spec)``. ``X`` keeps the original column types; string
    columns stay strings. Rows with a missing target are dropped.
    """
    spec = get_spec(name)
    path = Path(data_dir or DEFAULT_DATA_DIR) / spec.file
    if not path.exists():
        raise FileNotFoundError(f"Data file for {name!r} not found at {path}")

    if spec.column_names is not None:
        df = pd.read_csv(path, header=None, names=spec.column_names)
    else:
        df = pd.read_csv(path)

    if spec.target not in df.columns:
        raise ValueError(f"{name}: target column {spec.target!r} not in {path.name}")

    df = df.dropna(subset=[spec.target]).reset_index(drop=True)
    y = df.pop(spec.target)
    if spec.target_labels:
        unknown = set(y.unique()) - set(spec.target_labels)
        if unknown:
            raise ValueError(f"{name}: unexpected target values {sorted(unknown)}")
        y = y.map(spec.target_labels)
    y = y.astype(str).rename("target")

    # Object columns are read as strings; normalise whitespace once here.
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    return df, y, spec


def dataset_summary(data_dir: str | Path | None = None) -> pd.DataFrame:
    """One row per dataset: size, number of features and classes, imbalance."""
    rows = []
    for name in DATASETS:
        X, y, spec = load_dataset(name, data_dir)
        counts = y.value_counts()
        rows.append(
            {
                "dataset": name,
                "rows": len(X),
                "features": X.shape[1],
                "classes": y.nunique(),
                "minority_share": round(counts.min() / counts.sum(), 3),
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)
