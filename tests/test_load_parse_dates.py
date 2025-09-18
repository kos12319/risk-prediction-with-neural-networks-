import pandas as pd
from pathlib import Path
from src.data.load import LoadConfig, load_and_prepare


def test_load_includes_parse_dates(tmp_path: Path):
    p = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "issue_d": ["Jan-20", "Feb-20"],
            "earliest_cr_line": ["Jan-10", "Jan-15"],
            "loan_status": ["Fully Paid", "Charged Off"],
            "loan_amnt": [1000, 2000],
            "annual_inc": [50000, 60000],
        }
    )
    df.to_csv(p, index=False)
    cfg = LoadConfig(
        csv_path=str(p),
        target_col="loan_status",
        target_mapping={"Fully Paid": 1, "Charged Off": 0},
        parse_dates=["issue_d", "earliest_cr_line"],
        drop_leakage=False,
        leakage_cols=[],
        features=["loan_amnt", "annual_inc", "issue_d", "earliest_cr_line"],
    )
    out = load_and_prepare(cfg)
    # Dates parsed
    assert pd.api.types.is_datetime64_any_dtype(out["issue_d"])  # parsed
    assert pd.api.types.is_datetime64_any_dtype(out["earliest_cr_line"])  # parsed
    # Engineered credit_history_length should exist and be non-negative
    assert "credit_history_length" in out.columns
    assert (out["credit_history_length"].fillna(0) >= 0).all()

