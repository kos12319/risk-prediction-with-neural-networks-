from __future__ import annotations

import logging
import shutil
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

        try:
            figures_dir = run_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            lb_plot_df = leaderboard_frames.get("test", leaderboard_df).copy()
            for col in lb_plot_df.columns:
                try:
                    lb_plot_df[col] = pd.to_numeric(lb_plot_df[col])
                except Exception:
                    pass
            top_models = lb_plot_df.head(10)
            metrics_to_plot = [
                ("auc", True, "H2O Leaderboard — AUC"),
                ("logloss", False, "H2O Leaderboard — Log Loss"),
                ("rmse", False, "H2O Leaderboard — RMSE"),
            ]
            for metric, higher_is_better, title in metrics_to_plot:
                if metric not in top_models.columns:
                    continue
                vals = top_models[metric].astype(float)
                if vals.isna().all():
                    continue
                order = vals.sort_values(ascending=not higher_is_better)
                fig, ax = plt.subplots(figsize=(10, 6))
                order.plot(kind="barh", ax=ax, color="#1f77b4")
                ax.invert_yaxis()
                ax.set_xlabel(metric.upper())
                ax.set_ylabel("Model ID")
                ax.set_title(title)
                plt.tight_layout()
                fig_path = figures_dir / f"h2o_leaderboard_{metric}.png"
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)

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
            curves_top_n = automl_cfg.get("leaderboard_curve_top_n", 5) or 5
            try:
                curves_top_n = int(curves_top_n)
            except Exception:
                curves_top_n = 5

            roc_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []
            pr_curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

            source_lb_for_curves = leaderboard_frames.get("test", leaderboard_df)
            if curves_top_n > 0 and "model_id" in source_lb_for_curves.columns:
                top_curve_models = source_lb_for_curves.head(curves_top_n)["model_id"].tolist()
                for model_id in top_curve_models:
                    try:
                        candidate = h2o.get_model(model_id)
                        preds_candidate = candidate.predict(test_hf).as_data_frame()
                        probs_candidate, candidate_idx, _ = extract_probabilities(preds_candidate, pos_idx_default)
                        fpr, tpr, _ = roc_curve(y_true_test_bin, probs_candidate)
                        precision, recall, _ = precision_recall_curve(y_true_test_bin, probs_candidate)
                        roc_curves.append((model_id, fpr, tpr))
                        pr_curves.append((model_id, recall, precision))
                        # Update default index for downstream use if column choice changed
                        pos_idx_default = candidate_idx
                    except Exception:
                        continue

            if roc_curves:
                fig, ax = plt.subplots(figsize=(8, 6))
                for model_id, fpr, tpr in roc_curves:
                    ax.plot(fpr, tpr, label=model_id)
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
                for model_id, recall, precision in pr_curves:
                    ax.plot(recall, precision, label=model_id)
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
