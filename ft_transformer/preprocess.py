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


def split_feature_columns(all_columns: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) predictor names from the dataset schema."""

    features = [c for c in all_columns if c not in ID_COLUMNS + TARGET_COLUMNS]
    start = features.index(FIRST_CATEGORICAL_COLUMN)
    return features[:start], features[start:]


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
