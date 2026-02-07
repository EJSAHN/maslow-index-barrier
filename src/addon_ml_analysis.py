"""addon_ml_analysis.py

Machine-learning add-on analysis for the "Maslow Trap" / barrier manuscript.

Motivation
----------
Some reviewers may ask for a modern ML/AI benchmark alongside (or instead of)
classical GLM/logistic inference. This script provides a reproducible, drop-in
comparison using scikit-learn.

What this script does
---------------------
For each dataset (surface, regrowth, depth), it:
  1) Creates a binary target: severe := (score >= k)
  2) Fits multiple probabilistic classifiers with categorical one-hot encoding
  3) Evaluates out-of-sample performance with stratified K-fold CV
  4) Trains a final model on all data and outputs predicted probabilities for
     each unique condition, plus the corresponding barrier B = -ln(p)

Outputs (under <out_dir>/ml/)
----------------------------
tables/
  - ml_cv_metrics.csv
  - ml_predicted_conditions.csv
  - ml_feature_importance_permutation.csv   (optional; may be slow)
figures/
  - calibration_<dataset>.png

Dependencies
------------
Requires scikit-learn (not listed in the base requirements.txt for the public repo).
Install with:
    pip install scikit-learn

Run
---
From the repo root:
    python src/addon_ml_analysis.py
or:
    python src/addon_ml_analysis.py --kfold 5 --severe_threshold 4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance


# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_host_type(host: str) -> str:
    """Infer host type label used elsewhere in the repo."""
    host = str(host)
    if host.startswith("SH"):
        return "JG"  # Johnsongrass
    return "SB"  # Sorghum bicolor (annual)


def barrier_from_prob(p: float, min_p: float = 1e-12) -> float:
    p_adj = max(float(p), min_p)
    return -math.log(p_adj)


@dataclass
class DatasetSpec:
    name: str
    filename: str
    feature_cols: List[str]
    score_col: str = "score"


def build_model_zoo(random_state: int, rf_estimators: int = 300) -> Dict[str, object]:
    """Return a small model zoo of probabilistic classifiers.

    Parameters
    ----------
    random_state:
        Reproducibility seed.
    rf_estimators:
        Number of trees for the Random Forest model. Keep this moderate for
        reasonable runtime under cross-validation.
    """
    return {
        "logreg": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
        ),
        "rf": RandomForestClassifier(
            n_estimators=int(rf_estimators),
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "gboost": GradientBoostingClassifier(random_state=random_state),
    }


def make_pipeline(model, categorical_cols: List[str]) -> Pipeline:
    """OneHotEncode categoricals -> classifier."""
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            )
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("model", model)])


def cv_evaluate(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    kfold: int,
    random_state: int,
) -> Dict[str, float]:
    """Stratified K-fold CV; returns mean metrics across folds."""

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)

    aucs: List[float] = []
    ll: List[float] = []
    brier: List[float] = []
    acc: List[float] = []
    bacc: List[float] = []

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)

        # Some folds can become single-class in extreme imbalance; guard AUC/log-loss.
        try:
            aucs.append(float(roc_auc_score(yte, p)))
        except Exception:
            aucs.append(np.nan)

        try:
            ll.append(float(log_loss(yte, p, labels=[0, 1])))
        except Exception:
            ll.append(np.nan)

        brier.append(float(brier_score_loss(yte, p)))
        acc.append(float(accuracy_score(yte, yhat)))
        bacc.append(float(balanced_accuracy_score(yte, yhat)))

    def _nanmean(xs: Iterable[float]) -> float:
        arr = np.asarray(list(xs), dtype=float)
        if np.all(np.isnan(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    return {
        "roc_auc": _nanmean(aucs),
        "log_loss": _nanmean(ll),
        "brier": _nanmean(brier),
        "accuracy": _nanmean(acc),
        "balanced_accuracy": _nanmean(bacc),
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
    }


def save_calibration_plot(
    pipes: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    """Saves a calibration (reliability) curve plot."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for name, pipe in pipes.items():
        pipe.fit(X, y)
        CalibrationDisplay.from_estimator(pipe, X, y, n_bins=10, ax=ax, name=name)

    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def permutation_importance_table(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Permutation importance on the *original* categorical columns.

    We compute importance by permuting each raw column independently.
    This is slower than permuting one-hot columns but gives human-readable results.
    """

    pipe.fit(X, y)
    r = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="roc_auc",
    )
    out = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": r.importances_mean,
            "importance_sd": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return out


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Repository root (contains ./data)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: <base_dir>/outputs",
    )
    parser.add_argument(
        "--severe_threshold",
        type=float,
        default=4.0,
        help="Severe threshold k (default: 4.0) where severe := score >= k",
    )
    parser.add_argument("--kfold", type=int, default=5, help="# CV folds (default: 5)")
    parser.add_argument("--random_state", type=int, default=7, help="RNG seed")
    parser.add_argument(
        "--rf_estimators",
        type=int,
        default=300,
        help="RandomForest n_estimators (default: 300). Increase for stability, decrease for speed.",
    )
    parser.add_argument(
        "--compute_permutation_importance",
        action="store_true",
        help="Also compute permutation importance (can be slow).",
    )
    parser.add_argument(
        "--perm_repeats",
        type=int,
        default=20,
        help="Repeats for permutation importance (default: 20)",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_dir = os.path.join(base_dir, "outputs") if args.out_dir is None else os.path.abspath(args.out_dir)

    out_root = os.path.join(out_dir, "ml")
    out_tables = os.path.join(out_root, "tables")
    out_figures = os.path.join(out_root, "figures")
    ensure_dir(out_tables)
    ensure_dir(out_figures)

    data_dir = os.path.join(base_dir, "data")

    specs = [
        DatasetSpec(
            name="surface",
            filename="Figure2_SourceData.csv",
            feature_cols=["host", "isolate", "tissue", "surface_treatment"],
        ),
        DatasetSpec(
            name="regrowth",
            filename="Figure3_SourceData.csv",
            feature_cols=["host", "isolate", "round"],
        ),
        DatasetSpec(
            name="depth",
            filename="Figure4_SourceData.csv",
            feature_cols=["host_type", "accession", "isolate", "tissue"],
        ),
    ]

    zoo = build_model_zoo(random_state=args.random_state, rf_estimators=args.rf_estimators)

    cv_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []
    imp_rows: List[pd.DataFrame] = []

    for spec in specs:
        path = os.path.join(data_dir, spec.filename)
        df = pd.read_csv(path)

        if spec.name == "depth":
            # Align with the main pipeline labeling.
            df["host_type"] = df["accession"].apply(infer_host_type)

        # Build y
        y = (df[spec.score_col].astype(float) >= float(args.severe_threshold)).astype(int).to_numpy()

        # Build X (categorical)
        X = df[spec.feature_cols].copy()
        for c in X.columns:
            X[c] = X[c].astype(str)

        # Model evaluation
        pipes: Dict[str, Pipeline] = {}
        for model_name, model in zoo.items():
            pipes[model_name] = make_pipeline(model, categorical_cols=spec.feature_cols)

        for model_name, pipe in pipes.items():
            metrics = cv_evaluate(
                pipe,
                X,
                y,
                kfold=args.kfold,
                random_state=args.random_state,
            )
            metrics.update({"dataset": spec.name, "model": model_name})
            cv_rows.append(metrics)

        # Calibration plot (fit on full data; descriptive)
        save_calibration_plot(
            pipes,
            X,
            y,
            out_path=os.path.join(out_figures, f"calibration_{spec.name}.png"),
            title=f"Calibration curves ({spec.name}, severe >= {args.severe_threshold})",
        )

        # Final fit on all data -> condition-level predictions
        for model_name, pipe in pipes.items():
            pipe.fit(X, y)
            uniq = X.drop_duplicates().copy()
            uniq["pred_p_severe"] = pipe.predict_proba(uniq)[:, 1]
            uniq["pred_B"] = uniq["pred_p_severe"].apply(barrier_from_prob)
            uniq["dataset"] = spec.name
            uniq["model"] = model_name
            pred_rows.append(uniq)

        # Optional: permutation importance (on raw columns) for the best model by AUC
        if args.compute_permutation_importance:
            cv_df_tmp = pd.DataFrame([r for r in cv_rows if r["dataset"] == spec.name])
            best = cv_df_tmp.sort_values("roc_auc", ascending=False).iloc[0]["model"]
            best_pipe = pipes[str(best)]
            imp = permutation_importance_table(
                best_pipe,
                X,
                y,
                n_repeats=args.perm_repeats,
                random_state=args.random_state,
            )
            imp["dataset"] = spec.name
            imp["model"] = str(best)
            imp_rows.append(imp)

    cv_df = pd.DataFrame(cv_rows).sort_values(["dataset", "roc_auc"], ascending=[True, False])
    cv_df.to_csv(os.path.join(out_tables, "ml_cv_metrics.csv"), index=False)

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df.to_csv(os.path.join(out_tables, "ml_predicted_conditions.csv"), index=False)

    if imp_rows:
        imp_df = pd.concat(imp_rows, ignore_index=True)
        imp_df.to_csv(os.path.join(out_tables, "ml_feature_importance_permutation.csv"), index=False)

    print(f"[OK] Wrote: {os.path.join(out_tables, 'ml_cv_metrics.csv')}")
    print(f"[OK] Wrote: {os.path.join(out_tables, 'ml_predicted_conditions.csv')}")
    if imp_rows:
        print(f"[OK] Wrote: {os.path.join(out_tables, 'ml_feature_importance_permutation.csv')}")
    print(f"[OK] Calibration figures: {out_figures}")


if __name__ == "__main__":
    main()
