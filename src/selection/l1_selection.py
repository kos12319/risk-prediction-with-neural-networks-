from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.data.split import random_split, time_based_split
from src.features.preprocess import build_preprocessor, identify_feature_types
from src.selection.utils import aggregate_scores_by_group, get_feature_group_names


def _evaluate_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_inputs: List[str],
    target_col: str,
) -> float:
    X_train = train_df[feature_inputs]
    y_train = train_df[target_col]
    X_test = test_df[feature_inputs]
    y_test = test_df[target_col]

    num_cols, cat_cols = identify_feature_types(X_train)
    preproc = build_preprocessor(num_cols, cat_cols)
    Xtr = preproc.fit_transform(X_train)
    Xte = preproc.transform(X_test)

    clf = LogisticRegression(solver="saga", penalty="l2", max_iter=200, n_jobs=-1, class_weight="balanced")
    clf.fit(Xtr, y_train)
    y_prob = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y_test, y_prob))


def run_l1_selection(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    split_method: str = "time",
    time_col: str = "issue_d",
    test_size: float = 0.2,
    random_state: int = 42,
    missingness_threshold: float = 0.5,
    target_coverage: float = 0.99,
    C: float = 0.1,
    max_features: int | None = None,
    outdir: Path | None = None,
) -> Dict:
    # Filter features by missingness
    miss_rate = df[features].isna().mean()
    keep_features = [c for c in features if miss_rate.get(c, 0.0) <= missingness_threshold]

    # Split
    if split_method == "time":
        train_df, test_df = time_based_split(df, time_col=time_col, test_size=test_size)
    else:
        X = df[keep_features]
        y = df[target_col]
        X_train, X_test, y_train, y_test = random_split(
            X, y, test_size=test_size, random_state=random_state, stratify=True
        )
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

    # Full-set reference AUC
    auc_full = _evaluate_subset(train_df, test_df, keep_features, target_col)

    # Fit L1 logistic on encoded design for sparsity
    X_train = train_df[keep_features]
    y_train = train_df[target_col]
    num_cols, cat_cols = identify_feature_types(X_train)
    preproc = build_preprocessor(num_cols, cat_cols)
    Xtr = preproc.fit_transform(X_train)

    clf = LogisticRegression(
        solver="saga", penalty="l1", C=C, max_iter=300, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(Xtr, y_train)

    coefs = np.abs(clf.coef_).ravel()
    enc_names, group_for_col, order = get_feature_group_names(preproc)
    coef_by_group = aggregate_scores_by_group(coefs, group_for_col)

    ranking = sorted(((g, coef_by_group.get(g, 0.0)) for g in order), key=lambda x: x[1], reverse=True)

    # Incremental evaluation by adding one group at a time
    selected: List[str] = []
    curve_steps: List[Dict] = []
    target_auc = target_coverage * auc_full

    for feat, score in ranking:
        selected.append(feat)
        auc = _evaluate_subset(train_df, test_df, selected, target_col)
        curve_steps.append({"k": len(selected), "feature": feat, "auc": auc})
        if auc >= target_auc:
            break
        if max_features and len(selected) >= max_features:
            break

    results = {
        "method": "l1",
        "auc_full": auc_full,
        "target_coverage": target_coverage,
        "selected_features": selected,
        "steps": curve_steps,
        "ranking": [{"feature": f, "coef": s} for f, s in ranking],
    }

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "l1_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        # Plot curve
        if curve_steps:
            xs = [s["k"] for s in curve_steps]
            ys = [s["auc"] for s in curve_steps]
            plt.figure(figsize=(8, 5))
            plt.plot(xs, ys, marker="o", label="Subset AUC")
            plt.axhline(auc_full, color="gray", linestyle="--", label="Full AUC")
            plt.axhline(target_coverage * auc_full, color="green", linestyle=":", label=f"{int(target_coverage*100)}% target")
            plt.xlabel("# Features (original groups)")
            plt.ylabel("ROC AUC")
            plt.title("L1 Logistic Feature Selection Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "l1_auc_curve.png")
            plt.close()

        with open(outdir / "l1_ranking.csv", "w", encoding="utf-8") as f:
            f.write("feature,coef_abs\n")
            for ftr, sc in ranking:
                f.write(f"{ftr},{sc}\n")

    return results

