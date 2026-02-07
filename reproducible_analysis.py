#!/usr/bin/env python3
"""
Reproducible analysis (tables-only) for the Maslow Index / Defense Barrier manuscript.

This script reads the raw SourceData CSVs (Figure2–Figure5) and reproduces the
core *numeric* outputs used in the manuscript:

- Group-level severe-event probabilities: P_severe = P(score >= k)
- Jeffreys-type continuity correction for finite-sample / zero-event groups:
    P_severe_adj = (n_severe + 0.5) / (n_total + 1)
- Defense Barrier: B = -ln(P_severe_adj)
- GLM (binomial-logit) coefficient tables for the contrasts used in the paper
- GLM-predicted probabilities on the unique design grid of each dataset

No plotting / figure rendering is performed here.

Usage:
  python src/reproducible_analysis.py --data data --out outputs

Outputs:
  <out>/tables/*.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def barrier_from_p(p: float, eps: float = 1e-12) -> float:
    """Barrier transform: B = -ln(p) with numeric safety epsilon."""
    p = float(p)
    p = max(min(p, 1.0), eps)
    return -math.log(p)


def compute_group_metrics(
    df: pd.DataFrame,
    group_cols: List[str],
    score_col: str = "score",
    severe_threshold: int = 4,
) -> pd.DataFrame:
    """
    Compute finite-sample group summaries:
      - n_total, n_severe (score >= k)
      - P_severe (empirical proportion)
      - P_severe_adj (Jeffreys-type pseudocount: (n_severe+0.5)/(n_total+1))
      - B = -ln(P_severe_adj)
      - mean_score (Maslow index = E[score])
    """
    if score_col not in df.columns:
        raise ValueError(f"Missing score column '{score_col}'. Found: {df.columns.tolist()}")

    work = df.copy()
    work["_is_severe"] = (work[score_col] >= severe_threshold).astype(int)

    rows = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n_total = int(len(sub))
        n_severe = int(sub["_is_severe"].sum())
        p_severe = (n_severe / n_total) if n_total > 0 else np.nan

        # Jeffreys-type correction (same logic as run_maslow_barrier.py)
        p_severe_adj = ((n_severe + 0.5) / (n_total + 1.0)) if n_total > 0 else np.nan
        B = barrier_from_p(p_severe_adj) if n_total > 0 else np.nan

        mean_score = float(sub[score_col].mean()) if n_total > 0 else np.nan

        row = dict(zip(group_cols, keys))
        row.update(
            dict(
                n_total=n_total,
                n_severe=n_severe,
                P_severe=p_severe,
                P_severe_adj=p_severe_adj,
                B=B,
                mean_score=mean_score,
            )
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


def infer_host_type(accession: str) -> str:
    """
    Infer host life-history type from accession naming convention used in the datasets.

    Heuristic:
      - Accessions beginning with 'SH' are Johnsongrass (perennial).
      - Otherwise treated as Sorghum (annual).
    """
    if pd.isna(accession):
        return "Unknown"
    a = str(accession)
    return "Johnsongrass" if a.startswith("SH") else "Sorghum"


def fit_glm_and_export(
    df: pd.DataFrame,
    formula: str,
    design_cols: List[str],
    out_params_csv: str,
    out_pred_csv: str,
    severe_threshold: int = 4,
) -> None:
    """
    Fit a binomial GLM with logit link on a severe indicator (score >= k),
    export parameter table and predicted probabilities on the unique design grid.
    """
    work = df.copy()
    if "score" not in work.columns:
        raise ValueError("Expected 'score' column in dataframe.")
    work["is_severe"] = (work["score"] >= severe_threshold).astype(int)

    model = smf.glm(formula=formula, data=work, family=sm.families.Binomial()).fit()

    params = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    params.to_csv(out_params_csv, index=False)

    grid = work[design_cols].drop_duplicates().copy()
    grid["P_severe_pred"] = model.predict(grid)
    grid["B_pred"] = grid["P_severe_pred"].apply(barrier_from_p)
    grid.to_csv(out_pred_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Maslow Barrier reproducible analysis (tables-only).")
    ap.add_argument("--data", default="data", help="Path to data directory containing FigureX_SourceData.csv")
    ap.add_argument("--out", default="outputs", help="Output directory root")
    ap.add_argument("--severe-threshold", type=int, default=4, help="Severe threshold k (default: 4)")
    ap.add_argument("--write-manifest", action="store_true", help="Write a small JSON manifest into <out>/tables/")
    args = ap.parse_args()

    data_dir = args.data
    out_dir = args.out
    k = int(args.severe_threshold)

    tables_dir = os.path.join(out_dir, "tables")
    ensure_dir(tables_dir)

    # A) Surface integrity
    surface = pd.read_csv(os.path.join(data_dir, "Figure2_SourceData.csv"))
    surface_metrics = compute_group_metrics(
        surface,
        group_cols=["host", "tissue", "surface_treatment"],
        severe_threshold=k,
    )
    surface_metrics.to_csv(os.path.join(tables_dir, "surface_metrics.csv"), index=False)

    fit_glm_and_export(
        surface,
        formula="is_severe ~ C(host) + C(tissue) + C(surface_treatment)",
        design_cols=["host", "tissue", "surface_treatment"],
        out_params_csv=os.path.join(tables_dir, "glm_surface_params.csv"),
        out_pred_csv=os.path.join(tables_dir, "glm_surface_predicted.csv"),
        severe_threshold=k,
    )

    # B) Regrowth resilience
    regrowth = pd.read_csv(os.path.join(data_dir, "Figure3_SourceData.csv"))

    if "round" not in regrowth.columns:
        raise ValueError(f"Figure3_SourceData.csv missing 'round'. Columns: {regrowth.columns.tolist()}")

    # Guardrail: prevent spurious 'Round ?' category from a legacy missing value.
    if regrowth["round"].isna().any():
        regrowth["round"] = regrowth["round"].fillna(2)

    regrowth["round"] = regrowth["round"].astype(int)

    regrowth_metrics = compute_group_metrics(
        regrowth,
        group_cols=["host", "round"],
        severe_threshold=k,
    )
    regrowth_metrics.to_csv(os.path.join(tables_dir, "regrowth_metrics.csv"), index=False)

    fit_glm_and_export(
        regrowth,
        formula="is_severe ~ C(host) + C(round) + C(host):C(round)",
        design_cols=["host", "round"],
        out_params_csv=os.path.join(tables_dir, "glm_regrowth_params.csv"),
        out_pred_csv=os.path.join(tables_dir, "glm_regrowth_predicted.csv"),
        severe_threshold=k,
    )

    # C) Tissue depth (leaf vs rhizome)
    depth = pd.read_csv(os.path.join(data_dir, "Figure4_SourceData.csv"))

    if "accession" not in depth.columns:
        raise ValueError(f"Figure4_SourceData.csv missing 'accession'. Columns: {depth.columns.tolist()}")

    depth["host_type"] = depth["accession"].apply(infer_host_type)

    depth_metrics = compute_group_metrics(
        depth,
        group_cols=["host_type", "tissue"],
        severe_threshold=k,
    )
    depth_metrics.to_csv(os.path.join(tables_dir, "depth_metrics.csv"), index=False)

    fit_glm_and_export(
        depth,
        formula="is_severe ~ C(host_type) + C(tissue)",
        design_cols=["host_type", "tissue"],
        out_params_csv=os.path.join(tables_dir, "glm_depth_params.csv"),
        out_pred_csv=os.path.join(tables_dir, "glm_depth_predicted.csv"),
        severe_threshold=k,
    )

    # D) Mature stage population (GS6)
    gs6 = pd.read_csv(os.path.join(data_dir, "Figure5_SourceData.csv"))

    if "accession" not in gs6.columns:
        raise ValueError(f"Figure5_SourceData.csv missing 'accession'. Columns: {gs6.columns.tolist()}")

    gs6_metrics = compute_group_metrics(
        gs6,
        group_cols=["accession"],
        severe_threshold=k,
    )
    gs6_metrics.to_csv(os.path.join(tables_dir, "gs6_metrics.csv"), index=False)

    # Optional: manifest
    if args.write_manifest:
        manifest = {
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "platform": {
                "python": platform.python_version(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "inputs": {
                "data_dir": os.path.abspath(data_dir),
                "files": [
                    "Figure2_SourceData.csv",
                    "Figure3_SourceData.csv",
                    "Figure4_SourceData.csv",
                    "Figure5_SourceData.csv",
                ],
            },
            "params": {"severe_threshold": k},
            "outputs": {
                "tables_dir": os.path.abspath(tables_dir),
                "tables": [
                    "surface_metrics.csv",
                    "glm_surface_params.csv",
                    "glm_surface_predicted.csv",
                    "regrowth_metrics.csv",
                    "glm_regrowth_params.csv",
                    "glm_regrowth_predicted.csv",
                    "depth_metrics.csv",
                    "glm_depth_params.csv",
                    "glm_depth_predicted.csv",
                    "gs6_metrics.csv",
                ],
            },
        }
        with open(os.path.join(tables_dir, "analysis_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"[OK] Tables written to: {os.path.abspath(tables_dir)}")


if __name__ == "__main__":
    main()
