"""SHAP beeswarm plots for the team's final XGBoost models (no retraining).

Loads `model.joblib` + `preprocessor.joblib` from a finished run folder, pushes
rows through the saved preprocessor, and explains the booster with TreeExplainer.

Likelihood model: explains the 2024 rows (the held-out test year).
Severity model:   explains incident rows only (what it was trained on).
SHAP values are in the model's raw output: log-odds for likelihood, severity
points (1 = PDO, 2 = injury, 3 = fatal) for the regression.

Example (sample, local):
  uv run python scripts/shap_xgboost.py --task likelihood \
      --run-dir "results/google drive/likelihood_20261031/xgboost"
  uv run python scripts/shap_xgboost.py --task severity \
      --run-dir "results/google drive/severity_comparison_20261115_50trials/regression_xgboost"
Pass --dataset path/to/final_dataset.parquet for the full file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "collab-sol"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap
from collab_sol.features import split_risk_feature_columns


def load_xgb(path: Path):
    """joblib.load for the team's GPU-trained XGBoost pickles.

    Unpickling the full config+model buffer fails on CPU boxes ("input stream
    corrupted") even with the same xgboost version. The model section alone loads
    fine, so intercept Booster.__setstate__ and load only that.
    """
    import xgboost.core as xc

    original = xc.Booster.__setstate__

    def setstate(self, state):
        raw = bytes(state["handle"])
        key = b"L" + (5).to_bytes(8, "big") + b"Model"
        i = raw.find(key)
        if i < 0:  # not the combined format; use the normal path
            return original(self, state)
        self.handle = None
        self.__init__()  # fresh booster
        self.load_model(bytearray(raw[i + len(key):-1]))

    xc.Booster.__setstate__ = setstate
    try:
        return joblib.load(path)
    finally:
        xc.Booster.__setstate__ = original


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["likelihood", "severity"], required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=REPO / "final_dataset_sample_50k.parquet")
    p.add_argument("--year", type=int, default=2024, help="likelihood: year to explain")
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--max-display", type=int, default=20)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = args.out_dir or REPO / "results" / f"shap_{args.task}_xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_xgb(args.run_dir / "model.joblib")
    preprocessor = joblib.load(args.run_dir / "preprocessor.joblib")

    numeric, categorical = split_risk_feature_columns(pq.read_schema(args.dataset).names)
    cols = numeric + categorical
    frame = pd.read_parquet(args.dataset, columns=["asof_year", "incident", *cols])
    frame = frame[frame["asof_year"] == args.year] if args.task == "likelihood" else frame[frame["incident"] == 1]
    if len(frame) > args.max_rows:
        frame = frame.sample(args.max_rows, random_state=args.seed)
    frame[numeric] = frame[numeric].astype(np.float32)
    for c in categorical:
        frame[c] = frame[c].astype("category")
    print(f"Explaining {len(frame):,} rows ({args.task})")

    X = preprocessor.transform(frame[cols])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    names = list(preprocessor.get_feature_names_out())
    # ColumnTransformer prefixes: "numeric__x", "categorical__col_level"
    names = [re.sub(r"^(numeric|categorical)__", "", n) for n in names]
    assert X.shape[1] == len(names) == model.get_booster().num_features(), (
        X.shape, len(names), model.get_booster().num_features())

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    unit = "log-odds of incident" if args.task == "likelihood" else "severity points (1=PDO, 3=fatal)"
    explanation = shap.Explanation(values=sv, base_values=explainer.expected_value,
                                   data=X, feature_names=names)

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=args.max_display, show=False)
    plt.title(f"SHAP: {args.task} XGBoost ({len(frame):,} rows)", loc="left")
    plt.xlabel(f"SHAP value ({unit})", fontsize=9)
    plt.savefig(out_dir / "beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Importance rolled up to original columns (one-hot levels / missing flags summed).
    def source(name: str) -> str:
        if name.startswith("missingindicator_"):
            return name[len("missingindicator_"):]
        for c in categorical:
            if name == c or name.startswith(c + "_"):
                return c
        return name

    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(source(n), []).append(i)
    grouped = pd.Series({g: np.abs(sv[:, idx].sum(axis=1)).mean() for g, idx in groups.items()})
    encoded = pd.Series(np.abs(sv).mean(axis=0), index=names)
    pd.DataFrame({"mean_abs_shap": encoded}).sort_values("mean_abs_shap", ascending=False).to_csv(
        out_dir / "importance_by_encoded_feature.csv")
    grouped = grouped.sort_values(ascending=False)
    grouped.rename("mean_abs_shap").to_csv(out_dir / "importance_by_original_feature.csv")

    top = grouped.head(args.max_display)[::-1]
    plt.figure(figsize=(8, 0.35 * len(top) + 1.5))
    plt.barh(top.index, top.values)
    plt.xlabel(f"mean |SHAP| ({unit}), summed over encoded columns")
    plt.title(f"Top original features: {args.task} XGBoost")
    plt.tight_layout()
    plt.savefig(out_dir / "importance_bar.png", dpi=160)
    plt.close()
    print(grouped.head(15).to_string())
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
