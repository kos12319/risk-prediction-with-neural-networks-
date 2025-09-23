from __future__ import annotations

import logging
import shutil
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
    from h2o.automl import H2OAutoML

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

    logger.info("Connecting to H2O with args=%s", init_kwargs)
    conn = h2o.init(**init_kwargs)
    try:
        h2o.remove_all()

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

        try:
            figures_dir = run_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            lb_plot_df = leaderboard_df.copy()
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
