import pandas as pd

from src.data.split import time_based_kfold_splits, time_based_split


def test_time_based_split_monotonic():
    df = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "x": [1, 2, 3, 4],
        }
    )
    train, test = time_based_split(df, time_col="issue_d", test_size=0.5)
    assert train["issue_d"].max() <= test["issue_d"].min()


def test_time_based_kfold_splits_order_and_sizes():
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "issue_d": dates,
            "feature": range(len(dates)),
            "target": [0, 1] * 6,
        }
    )

    folds = time_based_kfold_splits(
        df,
        feature_cols=["feature"],
        target_col="target",
        time_col="issue_d",
        n_folds=3,
        initial_train_fraction=0.4,
        validation_fraction=0.2,
    )

    assert len(folds) == 3

    last_test_end = None
    prev_train_size = 0
    for fold in folds:
        train_df = fold.split.train_df
        val_df = fold.split.val_df
        test_df = fold.split.test_df

        # Train data should precede test data in time
        assert train_df["issue_d"].max() <= test_df["issue_d"].min()

        if val_df is not None and not val_df.empty:
            # Validation is carved from tail of train period
            assert train_df["issue_d"].max() <= val_df["issue_d"].min()

        if last_test_end is not None:
            assert last_test_end <= test_df["issue_d"].min()
        last_test_end = test_df["issue_d"].max()

        assert len(train_df) >= prev_train_size  # expanding window should grow
        prev_train_size = len(train_df)


def test_time_based_kfold_raises_on_too_many_folds():
    df = pd.DataFrame(
        {
            "issue_d": pd.date_range("2021-01-01", periods=5, freq="MS"),
            "feature": range(5),
            "target": [0, 1, 0, 1, 0],
        }
    )

    try:
        time_based_kfold_splits(
            df,
            feature_cols=["feature"],
            target_col="target",
            time_col="issue_d",
            n_folds=5,
            initial_train_fraction=0.2,
        )
    except ValueError:
        return
    assert False, "Expected ValueError when folds exceed available data"
