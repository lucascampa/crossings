"""Metrics that collab-sol's report does not cover: ROC-AUC and calibration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


def save_calibration_report(output_dir: str | Path) -> dict[str, float]:
    """Read ``risk_scores.csv`` from a collab-sol report and add calibration files.

    Writes ``calibration.json`` (ROC-AUC, Brier score, Brier of the constant
    prevalence forecast) and ``reliability_plot.png``. A model trained with a
    positive-class weight is deliberately over-confident, so a worse Brier score
    than the prevalence baseline is expected; rankings are unaffected.
    """

    output = Path(output_dir)
    scores = pd.read_csv(output / "risk_scores.csv")
    labels = scores["actual"].to_numpy()
    risk = scores["risk_score"].to_numpy()
    prevalence = float(labels.mean())

    metrics = {
        "roc_auc": float(roc_auc_score(labels, risk)),
        "brier_score": float(brier_score_loss(labels, risk)),
        "brier_score_prevalence_baseline": float(
            brier_score_loss(labels, np.full(len(labels), prevalence))
        ),
        "mean_predicted_risk": float(risk.mean()),
        "prevalence": prevalence,
    }
    (output / "calibration.json").write_text(json.dumps(metrics, indent=2))

    # Quantile bins: with ~1% positives, equal-width bins are almost all empty.
    observed, predicted = calibration_curve(labels, risk, n_bins=10, strategy="quantile")
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(predicted, observed, marker="o", label="model")
    top = max(predicted.max(), observed.max())
    axis.plot([0, top], [0, top], linestyle="--", color="grey", label="perfectly calibrated")
    axis.set(
        xlabel="Mean predicted risk",
        ylabel="Observed incident rate",
        title="Reliability plot (decile bins)",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reliability_plot.png", dpi=150)
    plt.close(figure)
    return metrics
