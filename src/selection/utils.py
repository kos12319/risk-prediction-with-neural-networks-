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

    for name in enc_names:
        # Names look like 'num__feature' or 'cat__feature_category'
        if name.startswith("num__"):
            orig = name.split("__", 1)[1]
        elif name.startswith("cat__"):
            rest = name.split("__", 1)[1]
            # Split once on '_' to separate original feature from category
            orig = rest.split("_", 1)[0]
        else:
            # Fallback: treat entire name as original
            orig = name

        group_for_col.append(orig)
        if orig not in order:
            order.append(orig)

    return enc_names, group_for_col, order


def aggregate_scores_by_group(
    scores: np.ndarray, group_for_col: List[str]
) -> Dict[str, float]:
    agg: Dict[str, float] = {}
    for s, g in zip(scores, group_for_col):
        agg[g] = agg.get(g, 0.0) + float(s)
    return agg

