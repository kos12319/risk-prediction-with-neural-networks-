from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer



def get_feature_group_names(preproc: ColumnTransformer) -> Tuple[List[str], List[str], List[str]]:
    """
    Return:
      - full encoded feature names from the transformer
      - original group names for each encoded column (same length as first list)
      - unique original feature names in order of first appearance
    """
    enc_names = list(preproc.get_feature_names_out())
    group_for_col: List[str] = []
    order: List[str] = []

    # Introspect the ColumnTransformer structure for robust grouping
    # Assumes transformers named 'num' and 'cat' as built in this repo
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for name, trans, cols in preproc.transformers_:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)

    # Number of output columns per original
    # Numerical pipeline is imputer+scaler → one column per input feature
    for c in num_cols:
        group_for_col.append(c)
        if c not in order:
            order.append(c)

    # Categorical: pull categories_ from the fitted OneHotEncoder to know expansion
    try:
        cat_enc = preproc.named_transformers_["cat"].named_steps["onehot"]
        categories = list(cat_enc.categories_)
    except Exception:
        categories = [[] for _ in cat_cols]
    for c, cats in zip(cat_cols, categories):
        n_out = len(cats) if hasattr(cats, "__len__") else 0
        if n_out == 0:
            # Fallback to at least one column
            n_out = 1
        for _ in range(n_out):
            group_for_col.append(c)
        if c not in order:
            order.append(c)

    # Align group_for_col to enc_names length in case of ColumnTransformer changes
    if len(group_for_col) != len(enc_names):
        # Fallback: best-effort by trimming or padding last known group
        if len(group_for_col) > len(enc_names):
            group_for_col = group_for_col[: len(enc_names)]
        else:
            last = group_for_col[-1] if group_for_col else ""
            group_for_col.extend([last] * (len(enc_names) - len(group_for_col)))

    return enc_names, group_for_col, order


def aggregate_scores_by_group(
    scores: np.ndarray, group_for_col: List[str]
) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for s, g in zip(scores, group_for_col):
        agg[g] = agg.get(g, 0.0) + float(s)
    return agg
