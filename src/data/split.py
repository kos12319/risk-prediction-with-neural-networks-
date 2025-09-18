from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    X, y, test_size: float = 0.2, random_state: int = 42, stratify: bool = True
):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )


def time_based_split(
    df: pd.DataFrame,
    time_col: str,
    test_size: float = 0.2,
    shuffle_within_groups: bool = False,
):
    """
    Split by time: earliest rows into train, most recent into test.
    Implementation: sort by `time_col` ascending and split by index at
    floor(n * (1 - test_size)). Returns (train_df, test_df).
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found for time-based split.")

    ordered = df.sort_values(time_col)
    n = len(ordered)
    split_idx = int(n * (1.0 - test_size))
    train_df = ordered.iloc[:split_idx].copy()
    test_df = ordered.iloc[split_idx:].copy()

    if shuffle_within_groups:
        train_df = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        test_df = test_df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    return train_df, test_df
