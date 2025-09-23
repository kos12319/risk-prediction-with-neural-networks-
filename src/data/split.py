from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


@dataclass
class TemporalFold:
    """Container for a single temporal k-fold split."""

    fold_id: int
    split: SplitResult
    train_range: Dict[str, Optional[str]]
    val_range: Dict[str, Optional[str]]
    test_range: Dict[str, Optional[str]]
    metadata: Dict[str, Any]


def _date_range(df: pd.DataFrame, time_col: str) -> Dict[str, Optional[str]]:
    if df.empty or time_col not in df.columns:
        return {"start": None, "end": None}
    series = pd.to_datetime(df[time_col], errors="coerce")
    series = series.dropna()
    if series.empty:
        return {"start": None, "end": None}
    return {"start": series.iloc[0].isoformat(), "end": series.iloc[-1].isoformat()}


def time_based_kfold_splits(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    time_col: str,
    n_folds: int,
    initial_train_fraction: float = 0.4,
    validation_fraction: float = 0.2,
    gap: int = 0,
    mode: str = "expanding",
    shuffle_within_folds: bool = False,
    random_state: Optional[int] = None,
) -> List[TemporalFold]:
    """Generate temporal k-fold splits using forward-chaining.

    Parameters
    ----------
    df : DataFrame
        Full dataset containing features and target.
    feature_cols : Sequence[str]
        Feature columns to extract into X matrices.
    target_col : str
        Target column name.
    time_col : str
        Name of the datetime column used for ordering.
    n_folds : int
        Number of temporal folds to generate. Must be >= 2.
    initial_train_fraction : float, optional
        Fraction of the dataset (0 < frac < 1) reserved for the initial
        training window before folds begin. Ensures the first fold has a
        non-empty training subset. Defaults to 0.4.
    validation_fraction : float, optional
        Fraction (0-1) of the fold's training subset carved out as the
        validation set (latest observations in the training period). Set to 0
        to disable validation for folds. Defaults to 0.2.
    gap : int, optional
        Number of rows to skip between the training and test periods of each
        fold (prevents leakage). Defaults to 0.
    mode : str, optional
        Temporal CV mode. Currently supports only "expanding". "rolling"
        raises ``ValueError`` until implemented.
    shuffle_within_folds : bool, optional
        When true, shuffles the training/validation subsets within each fold
        (after respecting chronological allocation). Defaults to False.
    random_state : Optional[int]
        Seed for the optional shuffling when ``shuffle_within_folds`` is true.

    Returns
    -------
    List[TemporalFold]
        One ``TemporalFold`` object per fold containing the ``SplitResult``
        and diagnostic metadata.
    """

    mode_normalized = (mode or "expanding").lower()
    if mode_normalized != "expanding":
        raise ValueError(f"Temporal CV mode '{mode}' is not supported yet; use 'expanding'.")

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")
    if not 2 <= int(n_folds):
        raise ValueError("Temporal k-fold requires at least 2 folds.")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    total_rows = len(ordered)
    if total_rows < 2:
        raise ValueError("Dataset must contain at least 2 rows for temporal k-fold split.")

    initial_train_fraction = float(max(0.0, min(1.0, initial_train_fraction)))
    validation_fraction = float(max(0.0, min(1.0, validation_fraction)))
    gap = int(max(0, gap))

    if initial_train_fraction <= 0.0:
        raise ValueError("initial_train_fraction must be > 0 for temporal k-fold splits.")

    initial_train_size = int(round(total_rows * initial_train_fraction))
    initial_train_size = max(1, min(initial_train_size, total_rows - n_folds))

    remaining = total_rows - initial_train_size
    if remaining < n_folds:
        raise ValueError(
            "Not enough rows to create the requested number of folds with the given initial "
            "training fraction. Reduce 'initial_train_fraction' or 'n_folds'."
        )

    base_fold_size = remaining // n_folds
    leftover = remaining % n_folds
    fold_sizes: List[int] = []
    for fold_idx in range(n_folds):
        size = base_fold_size + (1 if fold_idx < leftover else 0)
        fold_sizes.append(size)

    # Ensure no fold has zero test rows; merge with next fold if needed.
    for idx, size in enumerate(fold_sizes):
        if size > 0:
            continue
        # Borrow rows from subsequent folds when available.
        for jdx in range(idx + 1, len(fold_sizes)):
            if fold_sizes[jdx] > 1:
                fold_sizes[idx] += 1
                fold_sizes[jdx] -= 1
                break
        if fold_sizes[idx] <= 0:
            raise ValueError("Temporal fold computation produced an empty test segment.")

    folds: List[TemporalFold] = []
    cursor = initial_train_size
    rng = np.random.default_rng(random_state) if shuffle_within_folds else None

    for fold_id, test_size in enumerate(fold_sizes, start=1):
        test_start = cursor
        test_end = cursor + test_size
        if fold_id == len(fold_sizes):
            test_end = total_rows
        cursor = test_end

        train_end = max(0, test_start - gap)
        train_df_full = ordered.iloc[:train_end].copy()
        test_df = ordered.iloc[test_start:test_end].copy()

        if train_df_full.empty:
            raise ValueError(
                "Temporal fold generated an empty training subset. Adjust 'initial_train_fraction' "
                "or decrease the number of folds."
            )

        if validation_fraction > 0.0 and len(train_df_full) >= 2:
            n_val = int(round(len(train_df_full) * validation_fraction))
            n_val = max(1, min(n_val, len(train_df_full) - 1))
            val_df = train_df_full.iloc[-n_val:].copy()
            train_df = train_df_full.iloc[:-n_val].copy()
        else:
            val_df = None
            train_df = train_df_full

        if shuffle_within_folds and rng is not None:
            train_df = train_df.sample(frac=1.0, random_state=rng.integers(0, 1 << 32)).reset_index(drop=True)
            if val_df is not None:
                val_df = val_df.sample(frac=1.0, random_state=rng.integers(0, 1 << 32)).reset_index(drop=True)
            test_df = test_df.sample(frac=1.0, random_state=rng.integers(0, 1 << 32)).reset_index(drop=True)

        split = SplitResult(
            X_train=train_df[feature_cols],
            y_train=train_df[target_col],
            X_val=val_df[feature_cols] if val_df is not None else None,
            y_val=val_df[target_col] if val_df is not None else None,
            X_test=test_df[feature_cols],
            y_test=test_df[target_col],
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        fold_meta: Dict[str, Any] = {
            "fold": fold_id,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)) if val_df is not None else 0,
            "n_test": int(len(test_df)),
            "gap": gap,
        }

        folds.append(
            TemporalFold(
                fold_id=fold_id,
                split=split,
                train_range=_date_range(train_df_full, time_col),
                val_range=_date_range(val_df, time_col) if val_df is not None else {"start": None, "end": None},
                test_range=_date_range(test_df, time_col),
                metadata=fold_meta,
            )
        )

    return folds


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
