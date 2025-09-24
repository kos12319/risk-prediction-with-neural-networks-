from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

try:  # scikit-learn >=1.3
    from sklearn.preprocessing import TargetEncoder  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TargetEncoder = None  # type: ignore


logger = logging.getLogger(__name__)

def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=["number", "float", "int", "Int64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


@dataclass
class PreprocessingOptions:
    numerical_imputer: str = "median"
    numerical_scaler: str = "standard"
    categorical_imputer: str = "most_frequent"
    categorical_encoder: str = "one_hot"
    categorical_encoder_params: Mapping[str, Any] = field(default_factory=dict)
    sparse_threshold: float = 0.3


@dataclass
class PreprocessResult:
    transformer: ColumnTransformer
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    X_test: np.ndarray
    y_test: np.ndarray
    numerical_features: List[str]
    categorical_features: List[str]
    feature_names: List[str]


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, config: Optional[Mapping[str, Mapping[str, float]]]) -> None:
        # Store raw config for sklearn clone compatibility; derive copies during fit
        self.config = config
        self.columns_: List[str] = []
        self.clip_values_: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[np.ndarray] = None) -> "Winsorizer":
        df, _ = self._to_dataframe(X)
        self.columns_ = df.columns.tolist()
        clip_values: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        cfg = self.config or {}
        for col in self.columns_:
            spec = cfg.get(col)
            if not spec:
                continue
            spec_map = dict(spec)
            series = pd.to_numeric(df[col], errors="coerce")
            lower = spec_map.get("lower_value")
            upper = spec_map.get("upper_value")
            lower_q = spec_map.get("lower_quantile")
            upper_q = spec_map.get("upper_quantile")
            if lower_q is not None:
                try:
                    lower = float(series.quantile(float(lower_q)))
                except Exception:
                    lower = lower
            if upper_q is not None:
                try:
                    upper = float(series.quantile(float(upper_q)))
                except Exception:
                    upper = upper
            if lower is not None and pd.isna(lower):
                lower = None
            if upper is not None and pd.isna(upper):
                upper = None
            if lower is None and upper is None:
                continue
            clip_values[col] = (lower, upper)
        self.clip_values_ = clip_values
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        df, original_type = self._to_dataframe(X)
        if not self.clip_values_:
            return X
        for col, bounds in self.clip_values_.items():
            if col not in df.columns:
                continue
            lower, upper = bounds
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lower, upper=upper)
        if original_type == "dataframe":
            return df
        return df.to_numpy()

    def get_feature_names_out(self, input_features=None):  # type: ignore[override]
        if input_features is None:
            if self.columns_:
                return np.asarray(self.columns_, dtype=object)
            raise ValueError(
                "Winsorizer is not fitted yet; provide input_features or call fit before get_feature_names_out."
            )
        return np.asarray(list(input_features), dtype=object)

    def _to_dataframe(self, X: pd.DataFrame | np.ndarray) -> Tuple[pd.DataFrame, str]:
        if isinstance(X, pd.DataFrame):
            return X.copy(), "dataframe"
        arr = np.asarray(X)
        columns = self.columns_ if self.columns_ else [str(i) for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns), "array"


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    winsorize_cfg: Optional[Mapping[str, Mapping[str, float]]] = None,
    *,
    options: Optional[PreprocessingOptions] = None,
) -> ColumnTransformer:
    opts = options or PreprocessingOptions()
    winsor_cfg_filtered = (
        {col: winsorize_cfg[col] for col in numerical_cols if col in winsorize_cfg}
        if winsorize_cfg
        else {}
    )
    numerical_transformer = _build_numerical_pipeline(opts, winsor_cfg_filtered)
    categorical_transformer = _build_categorical_pipeline(opts)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        sparse_threshold=float(opts.sparse_threshold),
    )

    return preprocessor


def preprocess_tabular_data(
    split,
    *,
    winsorize_cfg: Optional[Mapping[str, Mapping[str, float]]] = None,
    preprocessing_cfg: Optional[Mapping[str, Any]] = None,
):
    """Fit the configured preprocessor on training data and transform splits."""

    opts = _resolve_options(preprocessing_cfg)
    num_cols, cat_cols = identify_feature_types(split.X_train)
    logger.info(
        "Preprocessing with %d numerical and %d categorical features",
        len(num_cols),
        len(cat_cols),
    )
    transformer = build_preprocessor(
        num_cols,
        cat_cols,
        winsorize_cfg=winsorize_cfg,
        options=opts,
    )

    X_train_proc = transformer.fit_transform(split.X_train)
    X_val_proc = transformer.transform(split.X_val) if split.X_val is not None else None
    X_test_proc = transformer.transform(split.X_test)
    logger.info(
        "Transformed feature matrices -> train: %s, val: %s, test: %s",
        X_train_proc.shape,
        X_val_proc.shape if X_val_proc is not None else None,
        X_test_proc.shape,
    )

    feature_names: List[str]
    if hasattr(transformer, "get_feature_names_out"):
        try:
            feature_names = list(transformer.get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]
    else:
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    return PreprocessResult(
        transformer=transformer,
        X_train=_to_dense(X_train_proc),
        y_train=np.asarray(split.y_train),
        X_val=_to_dense(X_val_proc) if X_val_proc is not None else None,
        y_val=np.asarray(split.y_val) if split.y_val is not None else None,
        X_test=_to_dense(X_test_proc),
        y_test=np.asarray(split.y_test),
        numerical_features=list(num_cols),
        categorical_features=list(cat_cols),
        feature_names=feature_names,
    )


def _resolve_options(cfg: Optional[Mapping[str, Any]] = None) -> PreprocessingOptions:
    if cfg is None:
        return PreprocessingOptions()

    numerical_cfg = cfg.get("numerical", {}) if isinstance(cfg, Mapping) else {}
    categorical_cfg = cfg.get("categorical", {}) if isinstance(cfg, Mapping) else {}

    return PreprocessingOptions(
        numerical_imputer=str(numerical_cfg.get("imputer", cfg.get("numerical_imputer", "median"))).lower(),
        numerical_scaler=str(numerical_cfg.get("scaler", cfg.get("numerical_scaler", "standard"))).lower(),
        categorical_imputer=str(categorical_cfg.get("imputer", cfg.get("categorical_imputer", "most_frequent"))).lower(),
        categorical_encoder=str(cfg.get("categorical_encoder", categorical_cfg.get("encoder", "one_hot"))).lower(),
        categorical_encoder_params=dict(categorical_cfg.get("encoder_params", cfg.get("categorical_encoder_params", {}))),
        sparse_threshold=float(cfg.get("sparse_threshold", 0.3)),
    )


def _build_numerical_pipeline(
    opts: PreprocessingOptions,
    winsor_cfg: Mapping[str, Mapping[str, float]],
) -> Pipeline:
    steps: List[Tuple[str, Any]] = []
    steps.append(("imputer", SimpleImputer(strategy=_map_numerical_imputer(opts.numerical_imputer))))
    if winsor_cfg:
        steps.append(("winsorizer", Winsorizer(winsor_cfg)))
    scaler = _map_numerical_scaler(opts.numerical_scaler)
    if scaler is not None:
        steps.append(("scaler", scaler))
    return Pipeline(steps=steps)


def _build_categorical_pipeline(opts: PreprocessingOptions) -> Pipeline:
    steps: List[Tuple[str, Any]] = []
    steps.append(("imputer", SimpleImputer(strategy=_map_categorical_imputer(opts.categorical_imputer))))
    encoder = _map_categorical_encoder(opts.categorical_encoder, opts.categorical_encoder_params)
    steps.append(("encoder", encoder))
    return Pipeline(steps=steps)


def _map_numerical_imputer(name: str) -> str:
    mapping = {
        "median": "median",
        "mean": "mean",
        "most_frequent": "most_frequent",
        "constant": "constant",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported numerical imputer '{name}'.")
    return mapping[name]


def _map_numerical_scaler(name: str):
    name = name.lower()
    if name in {"standard", "standardscaler"}:
        return StandardScaler()
    if name in {"minmax", "minmaxscaler"}:
        return MinMaxScaler()
    if name in {"robust", "robustscaler"}:
        return RobustScaler()
    if name in {"none", "identity", ""}:
        return None
    raise ValueError(f"Unsupported numerical scaler '{name}'.")


def _map_categorical_imputer(name: str) -> str:
    mapping = {
        "most_frequent": "most_frequent",
        "constant": "constant",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported categorical imputer '{name}'.")
    return mapping[name]


def _map_categorical_encoder(name: str, params: Mapping[str, Any]):
    name = name.lower()
    if name in {"one_hot", "onehot", "ohe"}:
        kwargs = {"handle_unknown": "ignore", "sparse_output": True}
        kwargs.update(params or {})
        return OneHotEncoder(**kwargs)
    if name in {"target", "target_encoder"}:
        if TargetEncoder is None:
            raise ImportError("TargetEncoder requires scikit-learn >= 1.3")
        return TargetEncoder(**(params or {}))
    raise ValueError(f"Unsupported categorical encoder '{name}'.")


def _to_dense(X):
    if X is None:
        return None
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def resolve_feature_inputs(
    df: pd.DataFrame,
    configured_features: Optional[Mapping[str, Any] | List[str]],
    target_col: str,
    time_columns: Optional[List[str]] = None,
    engineered_candidates: Optional[List[str]] = None,
) -> List[str]:
    """Resolve model input columns from config + engineered availability."""

    base_features: List[str]
    if configured_features is None:
        base_features = []
    elif isinstance(configured_features, Mapping):
        base_features = list(configured_features.keys())
    else:
        base_features = list(configured_features)

    engineered = engineered_candidates or [
        "credit_history_length",
        "income_to_loan_ratio",
        "fico_avg",
        "fico_spread",
    ]

    feature_pool = base_features + [col for col in engineered if col in df.columns]
    seen = set()
    resolved: List[str] = []
    time_cols = set(time_columns or [])

    for col in feature_pool:
        if col == target_col or col in time_cols or col not in df.columns:
            continue
        if col in seen:
            continue
        resolved.append(col)
        seen.add(col)

    return resolved
