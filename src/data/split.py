from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


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
    logger.info(
        "Time-based split on %s -> train: %d rows, test: %d rows",
        time_col,
        train_df.shape[0],
        test_df.shape[0],
    )

    if shuffle_within_groups:
        train_df = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        test_df = test_df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    return train_df, test_df


@dataclass
class SplitResult:
    """Container for train/validation/test splits in DataFrame form."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: Optional[pd.DataFrame]
    y_val: Optional[pd.Series]
    X_test: pd.DataFrame
    y_test: pd.Series
    train_df: pd.DataFrame
    val_df: Optional[pd.DataFrame]
    test_df: pd.DataFrame


def train_val_test_split(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    method: str = "random",
    time_col: str = "issue_d",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> SplitResult:
    """Create train/val/test splits with consistent outputs.

    Validation is carved from the training subset prior to any preprocessing or
    resampling, matching the project's leakage safeguards. For time-based
    splits, validation rows come from the most recent portion of the training
    period to preserve chronological ordering.
    """

    method = (method or "random").lower()
    if method not in {"random", "time"}:
        raise ValueError(f"Unsupported split method '{method}'.")

    val_size = float(max(0.0, min(1.0, val_size)))
    test_size = float(max(0.0, min(1.0, test_size)))

    if method == "time":
        train_df_full, test_df = time_based_split(
            df,
            time_col=time_col,
            test_size=test_size,
        )

        if val_size > 0 and len(train_df_full) > 1:
            ordered_train = train_df_full.sort_values(time_col)
            n_val = int(round(len(ordered_train) * val_size))
            n_val = min(max(n_val, 0), len(ordered_train))
            if n_val > 0:
                val_df = ordered_train.iloc[-n_val:].copy()
                train_df = ordered_train.iloc[:-n_val].copy()
                logger.info(
                    "Validation carved from training period -> train: %d, val: %d",
                    train_df.shape[0],
                    val_df.shape[0],
                )
            else:
                val_df = None
                train_df = ordered_train.copy()
        else:
            train_df = train_df_full.copy()
            val_df = None

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        if val_df is not None:
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]
        else:
            X_val = None
            y_val = None

    else:
        X = df[feature_cols]
        y = df[target_col]
        stratify_y = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )
        train_df_full = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        if val_size > 0 and len(X_train) > 1:
            stratify_val = y_train if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                random_state=random_state,
                stratify=stratify_val,
            )
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            logger.info(
                "Random split produced train: %d, val: %d, test: %d rows",
                len(X_train),
                len(X_val),
                len(X_test),
            )
        else:
            X_val = None
            y_val = None
            train_df = train_df_full
            val_df = None

        if val_df is None:
            logger.info(
                "Random split produced train: %d, test: %d rows",
                len(X_train),
                len(X_test),
            )

    return SplitResult(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
