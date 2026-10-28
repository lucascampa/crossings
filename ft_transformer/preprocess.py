"""Fold-safe preprocessing that keeps numeric and categorical features apart.

The team's MLP one-hot encodes everything into one matrix. The FT-Transformer
instead needs continuous values and integer category codes, so the matrix
produced here is laid out as ``[numeric | missing indicators | category codes]``.
The last ``len(categorical_features)`` columns are category codes; unseen
categories are encoded as -1 (the model shifts codes by one so -1 gets its own
token). Missing categories are their own level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, QuantileTransformer

ID_COLUMNS = ["crossingid", "asof_year"]
TARGET_COLUMNS = ["incident", "severity_target"]
# The final dataset stores its categorical predictors after the numeric ones,
# starting at this column (same convention as collab-sol's train_mlp.py).
FIRST_CATEGORICAL_COLUMN = "reportingagencytypecode"
MISSING_LEVEL = "__missing__"

# Predictors deliberately left out of the FT-Transformer, with the reason.
# Keep the "Features not used" table in README.md in step with this dict; every
# run also records it in run.json.
EXCLUDED_FEATURES = {
    "latitude": "Near-unique per crossing, so it acts like an ID and invites memorization.",
    "longitude": "Near-unique per crossing, so it acts like an ID and invites memorization.",
    "latlongsourcecode": (
        "Records how the coordinates were obtained (1 = Actual, 2 = Estimated), not "
        "anything about crash risk. Null in ~23% of sample rows, which have ~40x lower "
        "incident rate: likely a record-quality proxy."
    ),
    "B01001_E001": "ACS total population, duplicate of B01003_E001 (kept as the one total-population column).",
    "B03002_E001": "ACS total population, duplicate of B01003_E001 (kept as the one total-population column).",
}


def split_feature_columns(all_columns: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) predictor names from the dataset schema.

    The numeric/categorical boundary is positional, as in collab-sol's MLP script.
    Columns in ``EXCLUDED_FEATURES`` are removed after that split.
    """

    features = [c for c in all_columns if c not in ID_COLUMNS + TARGET_COLUMNS]
    start = features.index(FIRST_CATEGORICAL_COLUMN)
    numeric = [c for c in features[:start] if c not in EXCLUDED_FEATURES]
    categorical = [c for c in features[start:] if c not in EXCLUDED_FEATURES]
    return numeric, categorical


def _as_labels(frame: pd.DataFrame) -> np.ndarray:
    """Cast categories to strings, with missing values as their own level."""

    values = pd.DataFrame(frame).astype(object)
    return values.where(values.notna(), MISSING_LEVEL).astype(str).to_numpy()


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    n_quantiles: int = 1000,
    random_state: int = 42,
) -> ColumnTransformer:
    """Build the unfitted transformer; fit it on training rows only."""

    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    QuantileTransformer(
                        n_quantiles=n_quantiles,
                        output_distribution="normal",
                        subsample=1_000_000,
                        random_state=random_state,
                    ),
                ),
                numeric_features,
            ),
            (
                "missing_indicators",
                MissingIndicator(features="missing-only", error_on_new=False),
                numeric_features,
            ),
            (
                "categorical",
                make_pipeline(
                    FunctionTransformer(_as_labels),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        dtype=np.float32,
                    ),
                ),
                categorical_features,
            ),
        ],
        sparse_threshold=0.0,
    )
