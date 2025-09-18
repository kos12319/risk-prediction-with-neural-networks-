import pandas as pd
from src.data.split import time_based_split


def test_time_based_split_monotonic():
    df = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "x": [1, 2, 3, 4],
        }
    )
    train, test = time_based_split(df, time_col="issue_d", test_size=0.5)
    assert train["issue_d"].max() <= test["issue_d"].min()

