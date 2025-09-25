from __future__ import annotations

import logging
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.history import SimpleHistory


logger = logging.getLogger(__name__)


def train_h2o(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np: Optional[np.ndarray],
    y_val_np: Optional[np.ndarray],
    X_test_np: np.ndarray,
    y_test_np: np.ndarray,
    feature_names: List[str],
    target_name: str,
    automl_cfg: Dict[str, Any],
    model_path: Path,
    run_dir: Path,
    run_id: str,
    pos_label: int,
) -> Tuple[Dict[str, Any], SimpleHistory]:
    import h2o
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    from h2o.automl import H2OAutoML
    try:
        from h2o.exceptions import H2ODependencyWarning  # type: ignore
    except Exception:  # pragma: no cover - older h2o versions
        H2ODependencyWarning = None  # type: ignore

    def _sanitize_token(value: str, *, default: str = "model") -> str:
        token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
        token = token.strip("_").lower()
        return token or default

    def _ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _is_enabled(cfg_entry: Any, *, default: bool = False) -> Tuple[bool, Dict[str, Any]]:
        if isinstance(cfg_entry, dict):
            enabled = cfg_entry.get("enabled")
            if enabled is None:
                enabled = True
            return bool(enabled), dict(cfg_entry)
        if cfg_entry is None:
            return default, {}
        return bool(cfg_entry), {}

    def _model_varimp_df(model) -> Optional[pd.DataFrame]:  # type: ignore[no-untyped-def]
        df: Optional[pd.DataFrame] = None
        try:
            vip = model.varimp(use_pandas=True)
            if vip is not None:
                df = pd.DataFrame(vip)
        except Exception:
            try:
                vip = model.varimp()
                if vip:
                    df = pd.DataFrame(vip, columns=["feature", "relative_importance", "scaled_importance", "percentage"])
            except Exception:
                df = None
        if df is None or df.empty:
            try:
                coef = model.coef_norm()
                if isinstance(coef, dict) and coef:
                    df = pd.DataFrame(
                        {
                            "feature": list(coef.keys()),
                            "relative_importance": [abs(float(v)) for v in coef.values()],
                        }
                    )
            except Exception:
                df = None
        if df is None or df.empty:
            return None
        df = df.copy()
        rename_map: Dict[str, str] = {}
        if "variable" in df.columns:
            rename_map["variable"] = "feature"
        if "relative_importance" not in df.columns:
            for candidate in ["importance", "scaled_importance", "percentage"]:
                if candidate in df.columns:
                    rename_map[candidate] = "relative_importance"
                    break
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        if "feature" not in df.columns or "relative_importance" not in df.columns:
            return None
        df = df[["feature", "relative_importance"]].dropna()
        try:
            df["relative_importance"] = pd.to_numeric(df["relative_importance"], errors="coerce")
        except Exception:
            pass

        df = df.dropna(subset=["relative_importance"])
        if df.empty:
            return None
        df = df.sort_values("relative_importance", ascending=False).reset_index(drop=True)
        return df

    def _verify_java_available() -> None:
        java_path = shutil.which("java")
        if not java_path:
            raise RuntimeError(
                "H2O AutoML requires the Java runtime (java executable not found). "
                "Install a JRE/JDK and ensure it is on PATH or set JAVA_HOME."
            )

        try:
            subprocess.run(
                [java_path, "-version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except PermissionError as exc:
            raise RuntimeError(
                "Unable to execute 'java -version'; the current environment blocks launching Java. "
                "Grant execution permission or run in an environment where Java is allowed."
            ) from exc
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or b"") + (exc.stderr or b"")
            snippet = output.decode(errors="ignore").strip().splitlines()
            detail = snippet[-1] if snippet else f"exit code {exc.returncode}"
            raise RuntimeError(
                "Java is installed but 'java -version' failed ({}). "
                "Ensure the JVM is functional and not blocked by sandbox restrictions before running H2O.".format(detail)
            ) from exc

    feature_list = [str(f) for f in feature_names] if len(feature_names) else [f"feature_{i}" for i in range(X_train_np.shape[1])]

    train_df = pd.DataFrame(X_train_np, columns=feature_list)
    train_df[target_name] = np.asarray(y_train_np).astype(int)

    val_df = None
    if X_val_np is not None and y_val_np is not None:
        val_df = pd.DataFrame(X_val_np, columns=feature_list)
        val_df[target_name] = np.asarray(y_val_np).astype(int)

    test_df = pd.DataFrame(X_test_np, columns=feature_list)
    test_df[target_name] = np.asarray(y_test_np).astype(int)
    logger.info(
        "Starting H2O AutoML training with train=%s, val=%s, test=%s",
        train_df.shape,
        None if val_df is None else val_df.shape,
        test_df.shape,
    )

    init_kwargs: Dict[str, Any] = {}
    nthreads_cfg = automl_cfg.get("nthreads")
    if nthreads_cfg is not None:
        try:
            init_kwargs["nthreads"] = int(nthreads_cfg)
        except Exception:
            pass
    max_mem_cfg = automl_cfg.get("max_mem_size")
    if max_mem_cfg:
        init_kwargs["max_mem_size"] = str(max_mem_cfg)

    log_dir_cfg = automl_cfg.get("log_dir")
    if log_dir_cfg:
        log_dir_path = Path(log_dir_cfg)
        if not log_dir_path.is_absolute():
            log_dir_path = run_dir / log_dir_path
        log_dir_path.mkdir(parents=True, exist_ok=True)
        init_kwargs["log_dir"] = log_dir_path.as_posix()
    else:
        default_log_dir = run_dir / "h2o_logs"
        default_log_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["log_dir"] = default_log_dir.as_posix()

    log_level_cfg = automl_cfg.get("log_level")
    allowed_levels = {"TRACE", "DEBUG", "INFO", "WARN", "ERRR", "FATA"}
    if log_level_cfg:
        level_val = str(log_level_cfg).upper()
        if level_val not in allowed_levels:
            logger.warning("Unsupported H2O log level '%s'; falling back to WARN", level_val)
            init_kwargs["log_level"] = "WARN"
        else:
            init_kwargs["log_level"] = level_val
    else:
        init_kwargs["log_level"] = "WARN"

    _verify_java_available()
    logger.info("Connecting to H2O with args=%s", init_kwargs)
    conn = h2o.init(**init_kwargs)
    try:
        h2o.remove_all()

        try:
            if bool(automl_cfg.get("progress", False)):
                h2o.show_progress()
            else:
                h2o.no_progress()
        except Exception as exc:
            logger.warning("Failed to toggle H2O progress display: %s", exc)

        if bool(automl_cfg.get("suppress_dependency_warnings", True)) and H2ODependencyWarning is not None:
            try:
                warnings.filterwarnings("ignore", category=H2ODependencyWarning)
            except Exception:
                pass

        train_hf = h2o.H2OFrame(train_df)
        val_hf = h2o.H2OFrame(val_df) if val_df is not None else None
        test_hf = h2o.H2OFrame(test_df)
        logger.info("Uploaded frames to H2O: train=%s, val=%s, test=%s", train_hf.shape, None if val_hf is None else val_hf.shape, test_hf.shape)

        pos_label_str = str(int(pos_label))
        for frame in [train_hf, val_hf, test_hf]:
            if frame is None:
                continue
            vec = frame[target_name].asfactor()
            try:
                levels_raw = vec.levels()[0]
            except Exception:
                levels_raw = []
            if pos_label_str in levels_raw and len(levels_raw) >= 2:
                reordered_levels = [lvl for lvl in levels_raw if lvl != pos_label_str]
                reordered_levels.append(pos_label_str)
                try:
                    vec = vec.set_levels(reordered_levels)
                except Exception:
                    pass
            frame[target_name] = vec

        aml_kwargs: Dict[str, Any] = {
            "project_name": automl_cfg.get("project_name") or f"h2o_automl_{run_id}",
            "max_runtime_secs": automl_cfg.get("max_runtime_secs"),
            "max_models": automl_cfg.get("max_models"),
            "stopping_metric": automl_cfg.get("stopping_metric"),
            "sort_metric": automl_cfg.get("sort_metric"),
            "balance_classes": automl_cfg.get("balance_classes"),
            "max_after_balance_size": automl_cfg.get("max_after_balance_size"),
            "class_sampling_factors": automl_cfg.get("class_sampling_factors"),
            "seed": automl_cfg.get("seed"),
            "nfolds": automl_cfg.get("nfolds"),
            "keep_cross_validation_models": automl_cfg.get("keep_cross_validation_models"),
            "keep_cross_validation_predictions": automl_cfg.get("keep_cross_validation_predictions"),
            "keep_cross_validation_fold_assignment": automl_cfg.get("keep_cross_validation_fold_assignment"),
            "verbosity": "info",
        }

        include_algos = automl_cfg.get("include_algos")
        if include_algos:
            aml_kwargs["include_algos"] = list(include_algos)
        exclude_algos = automl_cfg.get("exclude_algos")
        if exclude_algos:
            aml_kwargs["exclude_algos"] = list(exclude_algos)

        checkpoints_dir = automl_cfg.get("export_checkpoints_dir")
        if checkpoints_dir:
            cp_path = Path(checkpoints_dir)
            if not cp_path.is_absolute():
                cp_path = run_dir / cp_path
            cp_path.mkdir(parents=True, exist_ok=True)
            aml_kwargs["export_checkpoints_dir"] = cp_path.as_posix()

        aml_kwargs = {k: v for k, v in aml_kwargs.items() if v is not None}

        logger.info("Launching H2OAutoML with options=%s", aml_kwargs)
        aml = H2OAutoML(**aml_kwargs)
        aml.train(x=feature_list, y=target_name, training_frame=train_hf, validation_frame=val_hf)

        leader = aml.leader
        leaderboard_df = aml.leaderboard.as_data_frame()
        leaderboard_frames: Dict[str, pd.DataFrame] = {"default": leaderboard_df}
        extra_cols_cfg = automl_cfg.get("leaderboard_extra_columns")
        if extra_cols_cfg:
            try:
                lb_extra = aml.get_leaderboard(extra_columns=extra_cols_cfg)
                leaderboard_frames["extra"] = lb_extra.as_data_frame()
            except Exception as exc:
                logger.warning("Failed to fetch extended leaderboard columns: %s", exc)
        if automl_cfg.get("leaderboard_make_test"):
            test_extra = extra_cols_cfg or "ALL"
            try:
                lb_test = h2o.make_leaderboard(aml, leaderboard_frame=test_hf, extra_columns=test_extra)
                leaderboard_frames["test"] = lb_test.as_data_frame()
            except Exception as exc:
                logger.warning("Failed to build test leaderboard: %s", exc)
        logger.info("AutoML completed; %d models on leaderboard. Leader=%s", leaderboard_df.shape[0], getattr(leader, 'model_id', 'unknown'))
        leaderboard_path = run_dir / "h2o_leaderboard.csv"
        try:
            leaderboard_df.to_csv(leaderboard_path, index=False)
        except Exception:
            leaderboard_path = run_dir / "h2o_leaderboard.txt"
            leaderboard_path.write_text(leaderboard_df.to_string(index=False), encoding="utf-8")
        try:
            leaderboard_json_path = run_dir / "h2o_leaderboard.json"
            leaderboard_df.to_json(leaderboard_json_path, orient="records", indent=2)
        except Exception:
            pass
        for name, frame in leaderboard_frames.items():
            if name == "default":
                continue
            out_path = run_dir / f"h2o_leaderboard_{name}.csv"
            try:
                frame.to_csv(out_path, index=False)
            except Exception:
                try:
                    out_txt = out_path.with_suffix(".txt")
                    out_txt.write_text(frame.to_string(index=False), encoding="utf-8")
                except Exception as exc:
                    logger.warning("Failed to persist leaderboard '%s': %s", name, exc)

        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        per_family_records: List[Dict[str, Any]] = []
        partial_dependence_records: List[Dict[str, Any]] = []
        model_varimp_cache: Dict[str, pd.DataFrame] = {}

        explanation_plots_cfg = automl_cfg.get("explanation_plots") or {}

        leader_model_id = getattr(leader, "model_id", None)
        if leader_model_id and str(leader_model_id) not in model_varimp_cache:
            leader_varimp_df = _model_varimp_df(leader)
            if leader_varimp_df is not None:
                model_varimp_cache[str(leader_model_id)] = leader_varimp_df

        per_family_cfg_raw = explanation_plots_cfg.get("per_family_varimp")
        per_family_enabled, per_family_cfg = _is_enabled(per_family_cfg_raw)
        logger.info("Per-family varimp enabled=%s", per_family_enabled)
        if per_family_enabled:
            try:
                plot_dir = _ensure_dir(figures_dir / "comparison" / "per_family_varimp")
                csv_dir = _ensure_dir(run_dir / "varimp_per_family")
                top_k = int(per_family_cfg.get("top_k", 20) or 20)
                source_df = leaderboard_frames.get("test")
                if source_df is None:
                    source_df = leaderboard_df
                group_col = None
                for candidate in ("algo", "model_type", "model_category"):
                    if candidate in source_df.columns:
                        group_col = candidate
                        break
                if group_col and "model_id" in source_df.columns:
                    best_by_group: Dict[str, pd.Series] = {}
                    for _, row in source_df.iterrows():
                        algo_val = row.get(group_col)
                        if pd.isna(algo_val):
                            algo_val = "unknown"
                        key = str(algo_val).strip() or "unknown"
                        if key not in best_by_group:
                            best_by_group[key] = row
                    for algo_key, row in best_by_group.items():
                        model_id = str(row.get("model_id"))
                        if not model_id:
                            continue
                        try:
                            model_obj = h2o.get_model(model_id)
                        except Exception:
                            continue
                        varimp_df = model_varimp_cache.get(model_id)
                        if varimp_df is None:
                            varimp_df = _model_varimp_df(model_obj)
                            if varimp_df is None:
                                continue
                            model_varimp_cache[model_id] = varimp_df
                        top_df = varimp_df.head(max(top_k, 1)).copy()
                        if top_df.empty:
                            continue
                        algo_token = _sanitize_token(algo_key)
                        csv_path = csv_dir / f"varimp_{algo_token}.csv"
                        try:
                            top_df.to_csv(csv_path, index=False)
                        except Exception:
                            csv_path_txt = csv_dir / f"varimp_{algo_token}.txt"
                            try:
                                csv_path_txt.write_text(top_df.to_string(index=False), encoding="utf-8")
                                csv_path = csv_path_txt
                            except Exception:
                                continue
                        plot_path = plot_dir / f"varimp_{algo_token}.png"
                        try:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            plot_df = top_df.iloc[::-1]
                            ax.barh(plot_df["feature"], plot_df["relative_importance"], color="#1f77b4")
                            ax.set_title(f"Feature Importance — {algo_key}")
                            ax.set_xlabel("Relative Importance")
                            ax.set_ylabel("Feature")
                            ax.grid(True, axis="x", alpha=0.2)
                            plt.tight_layout()
                            fig.savefig(plot_path, dpi=150)
                            plt.close(fig)
                        except Exception:
                            plot_path = None
                        per_family_records.append(
                            {
                                "algo": algo_key,
                                "model_id": model_id,
                                "plot_path": plot_path.as_posix() if plot_path and plot_path.exists() else None,
                                "csv_path": csv_path.as_posix(),
                            }
                        )
                        logger.info(
                            "Stored per-family varimp for %s | model=%s | features=%d",
                            algo_key,
                            model_id,
                            int(top_df.shape[0]),
                        )
            except Exception as exc:
                logger.warning("Failed to generate per-family varimp artifacts: %s", exc)

        partial_cfg_raw = explanation_plots_cfg.get("partial_dependence")
        partial_enabled, partial_cfg = _is_enabled(partial_cfg_raw)
        logger.info("Partial dependence enabled=%s", partial_enabled)
        if partial_enabled:
            try:
                partial_plot_dir = _ensure_dir(figures_dir / "explanations" / "partial_dependence")
                partial_csv_dir = _ensure_dir(run_dir / "partial_dependence")
                nbins = int(partial_cfg.get("nbins", 20) or 20)
                include_na = bool(partial_cfg.get("include_na", False))
                ice_requested = bool(partial_cfg.get("ice", False))
                center_requested = bool(partial_cfg.get("center", False))
                data_choice = str(partial_cfg.get("data", "train")).lower()
                if data_choice == "validation" and val_hf is not None:
                    plot_frame = val_hf
                elif data_choice == "test":
                    plot_frame = test_hf
                else:
                    plot_frame = train_hf
                features_cfg = partial_cfg.get("features")
                if features_cfg:
                    features = [str(f) for f in features_cfg]
                else:
                    top_k = int(partial_cfg.get("top_k", 3) or 3)
                    leader_varimp = model_varimp_cache.get(str(leader_model_id))
                    if leader_varimp is not None and not leader_varimp.empty:
                        features = leader_varimp["feature"].head(max(top_k, 1)).tolist()
                    else:
                        features = feature_list[: max(top_k, 1)]
                features = [f for f in features if f in feature_list]
                if not features:
                    features = feature_list[:1]
                for feature in features:
                    feature_token = _sanitize_token(feature, default="feature")
                    plot_path = partial_plot_dir / f"partial_{feature_token}.png"
                    csv_path = partial_csv_dir / f"partial_{feature_token}.csv"
                    try:
                        pp_results = leader.partial_plot(
                            frame=plot_frame,
                            cols=[feature],
                            plot=True,
                            include_na=include_na,
                            nbins=nbins,
                            save_plot_path=plot_path.as_posix(),
                        )
                    except Exception as exc:
                        logger.warning("Failed to compute partial dependence for %s: %s", feature, exc)
                        continue
                    if ice_requested:
                        logger.info("ICE overlay requested but not available in H2O partial_plot; generated PDP only for %s", feature)
                    if center_requested:
                        logger.info("Centering requested for %s but not supported by H2O partial_plot; returned raw PDP", feature)
                    table_df = None
                    try:
                        if pp_results and isinstance(pp_results, list):
                            entry = pp_results[0]
                            table = entry.get("table") if isinstance(entry, dict) else None
                            if table is not None and hasattr(table, "as_data_frame"):
                                table_df = table.as_data_frame()
                    except Exception:
                        table_df = None
                    csv_path_final = None
                    if table_df is not None and not table_df.empty:
                        try:
                            table_df.to_csv(csv_path, index=False)
                            csv_path_final = csv_path
                        except Exception:
                            try:
                                csv_path.write_text(table_df.to_string(index=False), encoding="utf-8")
                                csv_path_final = csv_path
                            except Exception:
                                csv_path_final = None
                    partial_dependence_records.append(
                        {
                            "feature": feature,
                            "plot_path": plot_path.as_posix() if plot_path.exists() else None,
                            "csv_path": csv_path_final.as_posix() if csv_path_final and csv_path_final.exists() else None,
                        }
                    )
                    logger.info("Generated partial dependence for %s | csv=%s", feature, bool(csv_path_final))
            except Exception as exc:
                logger.warning("Failed to generate partial dependence plots: %s", exc)

        try:
            lb_plot_df = leaderboard_frames.get("test", leaderboard_df).copy()

            def _shorten_model_id(model_id: str) -> str:
                if not isinstance(model_id, str):
                    return str(model_id)
                if "_AutoML" in model_id:
                    return model_id.split("_AutoML", 1)[0]
                return model_id

            def _build_label_map(*frames: Optional[pd.DataFrame]) -> Dict[str, str]:
                label_map: Dict[str, str] = {}
                seen_labels: Dict[str, int] = {}
                for frame in frames:
                    if frame is None or "model_id" not in frame.columns:
                        continue
                    reset = frame.reset_index(drop=True)
                    for rank, row in reset.iterrows():
                        model_id = row.get("model_id")
                        if not isinstance(model_id, str):
                            model_id = str(model_id)
                        if not model_id or model_id in label_map:
                            continue
                        algo = None
                        for key in ("algo", "model_type", "model_category"):
                            if key in row and pd.notna(row[key]):
                                algo = str(row[key]).strip()
                                break
                        short_id = _shorten_model_id(model_id)
                        base_label = algo if algo else short_id
                        label = f"{rank + 1}. {base_label}"
                        if short_id not in base_label:
                            label = f"{label} [{short_id}]"
                        if label in seen_labels:
                            seen_labels[label] += 1
                            label = f"{label} #{seen_labels[label]}"
                        else:
                            seen_labels[label] = 1
                        label_map[model_id] = label
                return label_map

            def _normalize_algo_labels(df: pd.DataFrame) -> None:
                if "model_id" not in df.columns or "algo" not in df.columns:
                    return
                try:
                    model_id_series = df["model_id"].astype(str)
                except Exception:
                    return
                try:
                    algo_series = df["algo"].astype(str)
                except Exception:
                    algo_series = df["algo"]
                try:
                    xrt_mask = model_id_series.str.startswith("XRT")
                    if xrt_mask.any():
                        algo_series = algo_series.where(~xrt_mask, "XRT")
                except Exception:
                    pass
                df.loc[:, "algo"] = algo_series

            _normalize_algo_labels(lb_plot_df)
            for frame in leaderboard_frames.values():
                _normalize_algo_labels(frame)

            label_map = _build_label_map(leaderboard_df, leaderboard_frames.get("test"), leaderboard_frames.get("extra"))
            for col in lb_plot_df.columns:
                if col in {"model_id", "model_category", "algo", "model_type"}:
                    continue
                try:
                    lb_plot_df[col] = pd.to_numeric(lb_plot_df[col])
                except Exception:
                    pass
            fam_col: Optional[str] = None
            for c in ("model_category", "algo", "model_type"):
                if c in lb_plot_df.columns:
                    fam_col = c
                    break

            # Produce family-level leaderboard bars (best model per family)
            # This keeps only the main model families visible in bar charts.
            try:
                if fam_col is not None and "model_id" in lb_plot_df.columns:
                    def _plot_family_leaderboard(metric: str, higher_is_better: bool, title: str) -> None:
                        if metric not in lb_plot_df.columns:
                            return
                        tmp = lb_plot_df[["model_id", fam_col, metric]].dropna().copy()
                        try:
                            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
                        except Exception:
                            pass
                        tmp = tmp.dropna(subset=[metric])
                        if tmp.empty:
                            return
                        fam_winners = tmp.sort_values(metric, ascending=not higher_is_better).groupby(fam_col, as_index=False).first()
                        series = fam_winners.set_index(fam_col)[metric].sort_values(ascending=not higher_is_better)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        direction_txt = "higher is better" if higher_is_better else "lower is better"
                        arrow = "↑" if higher_is_better else "↓"
                        pretty_metric = "Average Precision (AP)" if metric == "aucpr" else metric.upper()
                        series.name = f"{pretty_metric} ({direction_txt} {arrow})"
                        series.plot(kind="barh", ax=ax, color="#1f77b4", legend=True)
                        ax.invert_yaxis()
                        ax.set_xlabel(pretty_metric)
                        ax.set_ylabel("Category")
                        ax.set_title(title)
                        ax.legend(loc="lower right", frameon=True)
                        plt.tight_layout()
                        suffix = "by_family"
                        filename = f"h2o_leaderboard_{metric}_{suffix}.png"
                        fig_path = figures_dir / filename
                        fig.savefig(fig_path, dpi=150)
                        plt.close(fig)
                    _plot_family_leaderboard("aucpr", True, "H2O Leaderboard — Average Precision (by family)")
            except Exception:
                pass

            # Extra: best model per category (algo/model_category)
            category_winners: Optional[pd.DataFrame] = None
            try:
                if fam_col is not None and "auc" in lb_plot_df.columns:
                    # Prefer test leaderboard if available to pick winners
                    src = leaderboard_frames.get("test", leaderboard_df).copy()
                    if fam_col not in src.columns:
                        src = leaderboard_df.copy()
                    src = src[["model_id", fam_col, "auc"]].dropna()
                    try:
                        src["auc"] = pd.to_numeric(src["auc"])
                    except Exception:
                        pass
                    category_winners = src.sort_values("auc", ascending=False).groupby(fam_col, as_index=False).first()
                    category_winners.index = [label_map.get(str(mid), str(mid)) for mid in category_winners["model_id"]]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    category_winners.set_index(fam_col)["auc"].sort_values(ascending=False).plot(kind="barh", ax=ax, color="#2ca02c")
                    ax.invert_yaxis()
                    ax.set_xlabel("AUC")
                    ax.set_ylabel("Category")
                    ax.set_title("H2O — Best Model per Category (AUC)")
                    plt.tight_layout()
                    fig.savefig(figures_dir / "h2o_best_per_category_auc.png", dpi=150)
                    plt.close(fig)
            except Exception:
                pass

            # Full leaderboard chart (all models ranked)
            try:
                if "auc" in lb_plot_df.columns:
                    lb_full = lb_plot_df.copy()
                    try:
                        lb_full["auc"] = pd.to_numeric(lb_full["auc"], errors="coerce")
                    except Exception:
                        pass
                    lb_full = lb_full.dropna(subset=["auc"])
                    total_models = len(lb_full)
                    plot_top_n = automl_cfg.get("leaderboard_plot_top_n")
                    if plot_top_n is not None:
                        try:
                            plot_top_n = int(plot_top_n)
                        except Exception:
                            plot_top_n = None
                    top_n = min(total_models, plot_top_n) if plot_top_n and plot_top_n > 0 else min(total_models, 20)
                    lb_full["display_label"] = [label_map.get(str(mid), str(mid)) for mid in lb_full["model_id"]]
                    lb_full = lb_full.sort_values("auc", ascending=False).head(top_n)
                    fig_height = max(6.0, 0.4 * len(lb_full))
                    fig, ax = plt.subplots(figsize=(12, fig_height))
                    ax.barh(lb_full["display_label"], lb_full["auc"], color="#1f77b4")
                    ax.invert_yaxis()
                    ax.set_xlabel("AUC")
                    ax.set_ylabel("Model")
                    ax.set_title(f"H2O Leaderboard — AUC (top {len(lb_full)} of {total_models} models)")
                    auc_max = float(lb_full["auc"].max()) if not lb_full.empty else 1.0
                    if np.isfinite(auc_max):
                        ax.set_xlim(0.0, min(1.0, auc_max * 1.01))
                    ax.grid(True, axis="x", alpha=0.2)
                    plt.tight_layout()
                    fig.savefig(figures_dir / "h2o_leaderboard_auc.png", dpi=150)
                    plt.close(fig)
            except Exception:
                pass

            # Prepare helper for probability extraction once class levels are known
            levels_raw = train_hf[target_name].levels()
            levels = levels_raw[0] if isinstance(levels_raw, list) and len(levels_raw) else []
            lvl_map = {str(lv): idx for idx, lv in enumerate(levels)}
            pos_label_str = str(int(pos_label))
            pos_idx_default = lvl_map.get(pos_label_str)
            if pos_idx_default is None:
                pos_idx_default = lvl_map.get(str(pos_label))
            if pos_idx_default is None and levels:
                # Fallback to last level (pos label was appended during preprocessing)
                pos_idx_default = len(levels) - 1
            if pos_idx_default is None:
                pos_idx_default = 1

            def extract_probabilities(preds_df: pd.DataFrame, default_idx: int) -> Tuple[np.ndarray, int, Optional[str]]:
                idx = int(default_idx)
                prob_col = f"p{idx}"
                if prob_col not in preds_df.columns:
                    candidates = [c for c in preds_df.columns if c.startswith("p")]
                    if candidates:
                        candidates_sorted = sorted(candidates)
                        prob_col = candidates_sorted[-1]
                        try:
                            idx = int(prob_col[1:])
                        except Exception:
                            pass
                probs = np.asarray(preds_df[prob_col]).astype(float)
                label_raw: Optional[str] = None
                if levels and 0 <= idx < len(levels):
                    label_raw = levels[idx]
                return probs, idx, label_raw

            y_true_test_bin = (test_df[target_name].astype(int).to_numpy() == int(pos_label)).astype(int)
            # Prefer category-level ROC/PR curves (one best model per family)
            # Fallback to top-N if categories unavailable
            curves_top_n = automl_cfg.get("leaderboard_curve_top_n", 0) or 0
            try:
                curves_top_n = int(curves_top_n)
            except Exception:
                curves_top_n = 0

            roc_curves: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
            pr_curves: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
            prob_cache: Dict[str, np.ndarray] = {}

            source_lb_for_curves = leaderboard_frames.get("test", leaderboard_df)
            # Try category winners first
            cat_col = None
            for c in ("model_category", "algo", "model_type"):
                if c in source_lb_for_curves.columns:
                    cat_col = c
                    break
            models_to_plot: List[Tuple[str, str]] = []
            if cat_col is not None and "auc" in source_lb_for_curves.columns:
                try:
                    tmp = source_lb_for_curves[["model_id", cat_col, "auc"]].dropna().copy()
                    tmp["auc"] = pd.to_numeric(tmp["auc"], errors="coerce")
                    winners_df = tmp.sort_values("auc", ascending=False).groupby(cat_col, as_index=False).first()
                    for _, row in winners_df.iterrows():
                        mid = str(row["model_id"])
                        family = str(row[cat_col])
                        label = f"{family} [{_shorten_model_id(mid)}]" if _shorten_model_id(mid) not in family else family
                        models_to_plot.append((mid, label))
                except Exception:
                    models_to_plot = []
            # If no category winners found, fallback to top-N
            if not models_to_plot and curves_top_n > 0 and "model_id" in source_lb_for_curves.columns:
                top_curve_models = source_lb_for_curves.head(curves_top_n)["model_id"].tolist()
                models_to_plot = [(m, label_map.get(m, m)) for m in top_curve_models]

            for model_id, label in models_to_plot:
                try:
                    candidate = h2o.get_model(model_id)
                    preds_candidate = candidate.predict(test_hf).as_data_frame()
                    probs_candidate, candidate_idx, _ = extract_probabilities(preds_candidate, pos_idx_default)
                    fpr, tpr, _ = roc_curve(y_true_test_bin, probs_candidate)
                    precision, recall, _ = precision_recall_curve(y_true_test_bin, probs_candidate)
                    roc_curves.append((model_id, label, fpr, tpr))
                    pr_curves.append((model_id, label, recall, precision))
                    prob_cache[model_id] = probs_candidate
                    pos_idx_default = candidate_idx
                except Exception:
                    continue

            if roc_curves:
                fig, ax = plt.subplots(figsize=(8, 6))
                for _, label, fpr, tpr in roc_curves:
                    ax.plot(fpr, tpr, label=label)
                ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("H2O Leaderboard — ROC Curves (Test)")
                ax.legend(loc="lower right", fontsize="small")
                ax.grid(True, alpha=0.2)
                plt.tight_layout()
                (figures_dir / "comparison").mkdir(parents=True, exist_ok=True)
                roc_path = figures_dir / "comparison" / "h2o_leaderboard_roc.png"
                fig.savefig(roc_path, dpi=150)
                plt.close(fig)

            if pr_curves:
                fig, ax = plt.subplots(figsize=(8, 6))
                baseline = y_true_test_bin.mean() if y_true_test_bin.size else 0.0
                ax.hlines(baseline, 0, 1, linestyle="--", color="#888888", linewidth=1, label="Baseline")
                for _, label, recall, precision in pr_curves:
                    ax.plot(recall, precision, label=label)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("H2O Leaderboard — Precision-Recall Curves (Test)")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.0])
                ax.legend(loc="lower left", fontsize="small")
                ax.grid(True, alpha=0.2)
                plt.tight_layout()
                pr_path = figures_dir / "comparison" / "h2o_leaderboard_pr.png"
                fig.savefig(pr_path, dpi=150)
                plt.close(fig)

            explanation_plots_cfg = automl_cfg.get("explanation_plots") or {}
            if explanation_plots_cfg.get("model_correlation", True):
                try:
                    corr_dir = figures_dir / "comparison"
                    corr_dir.mkdir(parents=True, exist_ok=True)
                    corr_path = corr_dir / "h2o_model_correlation.png"
                    corr_models: List[Tuple[str, str]] = []
                    for mid, label in models_to_plot:
                        probs = prob_cache.get(mid)
                        if probs is None or probs.size == 0:
                            continue
                        if not np.all(np.isfinite(probs)):
                            continue
                        corr_models.append((mid, label))
                    if len(corr_models) >= 2:
                        prob_df = pd.DataFrame(
                            {
                                lbl: prob_cache[mid]
                                for mid, lbl in corr_models
                            }
                        )
                        corr_matrix = prob_df.corr(method="pearson").fillna(0.0)
                        labels = [lbl for _, lbl in corr_models]
                        if not corr_matrix.empty:
                            fig_size = max(6.0, 0.7 * len(labels))
                            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                            upper_vals = corr_matrix.to_numpy()[np.triu_indices(len(labels), k=1)]
                            if upper_vals.size:
                                vmin = np.nanmin(upper_vals)
                            else:
                                vmin = np.nanmin(corr_matrix.to_numpy())
                            if not np.isfinite(vmin):
                                vmin = 0.0
                            vmax = 1.0
                            if vmin >= vmax:
                                vmin = vmax - 1e-3
                            im = ax.imshow(corr_matrix.to_numpy(), cmap="coolwarm", vmin=vmin, vmax=vmax)
                            ax.set_xticks(range(len(labels)))
                            ax.set_xticklabels(labels, rotation=45, ha="right")
                            ax.set_yticks(range(len(labels)))
                            ax.set_yticklabels(labels)
                            for i in range(len(labels)):
                                for j in range(len(labels)):
                                    ax.text(
                                        j,
                                        i,
                                        f"{corr_matrix.iat[i, j]:.3f}",
                                        ha="center",
                                        va="center",
                                        fontsize=8,
                                        color="black",
                                    )
                            ax.set_title("Model Correlation (per-algorithm representative)")
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                            plt.tight_layout()
                            fig.savefig(corr_path, dpi=150)
                            plt.close(fig)
                            corr_csv = run_dir / "h2o_model_correlation.csv"
                            corr_matrix.to_csv(corr_csv)
                        else:
                            aml.model_correlation_heatmap(test_hf, save_plot_path=corr_path.as_posix())
                    else:
                        aml.model_correlation_heatmap(test_hf, save_plot_path=corr_path.as_posix())
                except Exception as exc:
                    logger.warning("Failed to generate model correlation heatmap: %s", exc)
            if explanation_plots_cfg.get("varimp_heatmap", True):
                try:
                    heat_dir = figures_dir / "comparison"
                    heat_dir.mkdir(parents=True, exist_ok=True)
                    heat_path = heat_dir / "h2o_varimp_heatmap.png"
                    aml.varimp_heatmap(save_plot_path=heat_path.as_posix())
                except Exception as exc:
                    logger.warning("Failed to generate variable importance heatmap: %s", exc)
            # Extra: varimp heatmap for winners only (one best model per category)
            try:
                heat_dir = figures_dir / "comparison"
                heat_dir.mkdir(parents=True, exist_ok=True)
                winners_heat_path = heat_dir / "h2o_varimp_heatmap_winners.png"
                # Determine the category column present on the leaderboard
                src_lb = leaderboard_frames.get("test", leaderboard_df).copy()
                cat_col = None
                for c in ("model_category", "algo", "model_type"):
                    if c in src_lb.columns:
                        cat_col = c
                        break
                metric_col = "auc" if "auc" in src_lb.columns else None
                if cat_col is not None and metric_col is not None and "model_id" in src_lb.columns:
                    # Pick the top model per category by metric
                    tmp = src_lb[["model_id", cat_col, metric_col]].dropna()
                    try:
                        tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
                    except Exception:
                        pass
                    tmp = tmp.dropna(subset=[metric_col])
                    if not tmp.empty:
                        winners = tmp.sort_values(metric_col, ascending=False).groupby(cat_col, as_index=False).first()
                        # Collect normalized varimp vectors for each winner
                        winner_varimps: Dict[str, pd.Series] = {}
                        feature_union: List[str] = []
                        for _, row in winners.iterrows():
                            model_id = str(row.get("model_id"))
                            if not model_id:
                                continue
                            try:
                                model_obj = h2o.get_model(model_id)
                            except Exception:
                                continue
                            vip_df = _model_varimp_df(model_obj)
                            if vip_df is None or vip_df.empty:
                                continue
                            s = vip_df.set_index("feature")["relative_importance"].astype(float)
                            total = float(s.abs().sum()) or 0.0
                            if total > 0:
                                s = s / total
                            # Keep a reasonable number of top features per model
                            top_k = int(explanation_plots_cfg.get("varimp_top_k", 20) or 20)
                            s = s.sort_values(ascending=False).head(max(top_k, 1))
                            winner_varimps[model_id] = s
                            feature_union.extend(list(s.index))
                        feature_union = list(dict.fromkeys(feature_union))  # de-dup while preserving order
                        if winner_varimps and feature_union:
                            # Build matrix [features x models]
                            model_ids = list(winner_varimps.keys())
                            import numpy as _np
                            mat = _np.zeros((len(feature_union), len(model_ids)), dtype=float)
                            for j, mid in enumerate(model_ids):
                                s = winner_varimps[mid]
                                for i, feat in enumerate(feature_union):
                                    if feat in s.index:
                                        try:
                                            mat[i, j] = float(s.loc[feat])
                                        except Exception:
                                            mat[i, j] = 0.0
                            # Optionally reduce to a global top-N across all winners to keep the plot readable.
                            # Aggregate per-feature importance across winner models using mean (default) or max.
                            agg_mode = str(explanation_plots_cfg.get("varimp_winners_sort", "mean")).lower()
                            if agg_mode not in {"mean", "max"}:
                                agg_mode = "mean"
                            scores = mat.mean(axis=1) if agg_mode == "mean" else mat.max(axis=1)
                            global_top_k = int(explanation_plots_cfg.get("varimp_winners_top_k", 35) or 35)
                            if global_top_k > 0 and len(feature_union) > global_top_k:
                                keep_idx = _np.argsort(scores)[::-1][:global_top_k]
                                keep_idx = _np.sort(keep_idx)  # keep original order within selection for legibility
                                mat = mat[keep_idx, :]
                                feature_union = [feature_union[i] for i in keep_idx.tolist()]

                            # Plot with adaptive height to avoid cramped y-labels
                            row_height = float(explanation_plots_cfg.get("varimp_heatmap_row_height", 0.35) or 0.35)
                            fig_h = max(6.0, 2.0 + row_height * max(1, len(feature_union)))
                            fig, ax = plt.subplots(figsize=(12, fig_h))
                            im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r")
                            ax.set_yticks(range(len(feature_union)))
                            y_fontsize = int(explanation_plots_cfg.get("varimp_heatmap_fontsize", 9) or 9)
                            ax.set_yticklabels(feature_union, fontsize=y_fontsize)
                            labels = [label_map.get(mid, mid) for mid in model_ids]
                            ax.set_xticks(range(len(model_ids)))
                            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                            ax.set_xlabel("Model (category winners)")
                            ax.set_ylabel("Feature")
                            ax.set_title("Variable Importance Heatmap — Winners per Category")
                            cbar = fig.colorbar(im, ax=ax)
                            cbar.ax.set_ylabel("Relative importance (normalized)")
                            fig.tight_layout()
                            fig.savefig(winners_heat_path, dpi=200)
                            plt.close(fig)
            except Exception as exc:
                logger.warning("Failed to generate winners-only varimp heatmap: %s", exc)
            pareto_cfg = explanation_plots_cfg.get("pareto_front", True)
            if pareto_cfg:
                pareto_args: Dict[str, Any] = {}
                if isinstance(pareto_cfg, dict):
                    for key in ("x_metric", "y_metric"):
                        if key in pareto_cfg and pareto_cfg[key] is not None:
                            pareto_args[key] = pareto_cfg[key]
                try:
                    pareto_result = aml.pareto_front(test_frame=test_hf, **pareto_args)
                    pareto_dir = figures_dir / "comparison"
                    pareto_dir.mkdir(parents=True, exist_ok=True)
                    pareto_fig = None
                    try:
                        pareto_fig = pareto_result.figure()
                    except Exception:
                        pareto_fig = None
                    if pareto_fig is not None:
                        pareto_path = pareto_dir / "h2o_pareto_front.png"
                        pareto_fig.savefig(pareto_path, dpi=150)
                        plt.close(pareto_fig)
                    pareto_frame = getattr(pareto_result, "pareto_front", None)
                    if pareto_frame is None:
                        pareto_frame = getattr(pareto_result, "frame", None)
                    if pareto_frame is not None:
                        try:
                            pareto_df = pareto_frame.as_data_frame()
                            pareto_csv = run_dir / "h2o_pareto_front.csv"
                            pareto_df.to_csv(pareto_csv, index=False)
                        except Exception:
                            pass
                except Exception as exc:
                    logger.warning("Failed to generate H2O Pareto front plot: %s", exc)
        except Exception:
            pass

        preds_test = leader.predict(test_hf).as_data_frame()
        preds_val = leader.predict(val_hf).as_data_frame() if val_hf is not None else None
        logger.info("Generated predictions for test=%s%s", preds_test.shape, " and val=%s" % (preds_val.shape,) if preds_val is not None else "")

        levels_raw = train_hf[target_name].levels()
        levels = levels_raw[0] if isinstance(levels_raw, list) and len(levels_raw) else []
        lvl_map = {str(lv): idx for idx, lv in enumerate(levels)}
        pos_label_str = str(int(pos_label))
        pos_idx = lvl_map.get(pos_label_str)
        if pos_idx is None:
            pos_idx = lvl_map.get(str(pos_label))
        if pos_idx is None and levels:
            pos_idx = lvl_map.get(str(levels[0]), 0)
        prob_col = f"p{int(pos_idx)}"
        if prob_col not in preds_test.columns:
            prob_candidates = [c for c in preds_test.columns if c.startswith("p")]
            if prob_candidates:
                prob_col = prob_candidates[0]
                try:
                    pos_idx = int(prob_col[1:])
                except Exception:
                    pos_idx = int(pos_idx)

        try:
            prob_label_raw = levels[int(pos_idx)] if levels and 0 <= int(pos_idx) < len(levels) else pos_label_str
        except Exception:
            prob_label_raw = pos_label_str
        try:
            prob_label_value = int(float(prob_label_raw))
        except Exception:
            prob_label_value = prob_label_raw

        y_prob_test = np.asarray(preds_test[prob_col]).astype(float)
        y_prob_val = None
        if preds_val is not None:
            y_prob_val = np.asarray(preds_val[prob_col]).astype(float)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        saved_model_path = Path(h2o.save_model(model=leader, path=model_path.parent.as_posix(), force=True))
        final_model_path = saved_model_path
        if saved_model_path.is_dir():
            archive_base = model_path.with_suffix("")
            archive_file = shutil.make_archive(archive_base.as_posix(), "zip", root_dir=saved_model_path)
            final_model_path = Path(archive_file)
            try:
                shutil.rmtree(saved_model_path)
            except Exception:
                pass
        elif saved_model_path.suffix != model_path.suffix or saved_model_path.name != model_path.name:
            try:
                shutil.move(saved_model_path.as_posix(), model_path.as_posix())
                final_model_path = model_path
            except Exception:
                final_model_path = saved_model_path

        history = SimpleHistory([], [])
        leader_algo = getattr(leader, "algo", None)
        result: Dict[str, Any] = {
            "y_prob": y_prob_test,
            "y_prob_val": y_prob_val,
            "param_count": None,
            "device": "h2o",
            "device_info": {
                "selected": "h2o",
                "leader_id": getattr(leader, "model_id", None),
                "leader_algo": leader_algo,
            },
            "epochs_ran": 0,
            "epoch_stats": [],
            "leaderboard_path": leaderboard_path.as_posix(),
            "model_path": final_model_path.as_posix(),
            "leader_id": getattr(leader, "model_id", None),
            "leader_algo": leader_algo,
            "y_prob_label": prob_label_value,
            "per_family_varimp": per_family_records,
            "partial_dependence": partial_dependence_records,
        }
        logger.info(
            "Finished H2O AutoML | leader=%s (%s) | model_path=%s",
            result.get("leader_id"),
            leader_algo,
            final_model_path,
        )
        return result, history
    finally:
        try:
            conn.shutdown(prompt=False)
        except Exception:
            pass
