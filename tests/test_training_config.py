from pathlib import Path

from src.training.config import load_config_with_extends


def test_load_config_with_extends_resolves_nested_chain():
    cfg = load_config_with_extends(Path("configs/provider_agnostic.yaml"))
    assert cfg["model"]["backend"] == "pytorch"
    features = cfg["data"].get("features", [])
    assert "loan_amnt" in features
    assert cfg["data"]["target_col"] == "loan_status"
