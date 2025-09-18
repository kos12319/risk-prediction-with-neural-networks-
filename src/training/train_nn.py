from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import sys
import shutil
import json as _json
import hashlib
import os

# Apply safe env as early as possible to avoid BLAS/Accelerate crashes on import
try:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_PROC_BIND", "FALSE")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", ".cache")
    os.environ.setdefault("MPLCONFIGDIR", ".mplcache")
    import platform as _platform  # local import to avoid global dependency
    if _platform.system() == "Darwin" and _platform.machine() in {"arm64", "aarch64"}:
        os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")
except Exception:
    pass

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler

from sklearn.utils import check_random_state
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data.load import LoadConfig, load_and_prepare
from src.data.split import random_split, time_based_split
from src.eval.metrics import (
    compute_metrics_binary,
    plot_learning_curves,
    save_metrics,
    plot_roc_curve,
    plot_pr_curve,
    choose_threshold_youden,
    choose_threshold_f1,
    confusion_metrics_at_threshold,
)
from src.features.preprocess import build_preprocessor, identify_feature_types
from src.utils.seed import set_seed, make_torch_generator, make_worker_init_fn
import platform


def _collect_system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["machine"] = platform.machine()
        info["processor"] = platform.processor()
        info["platform"] = platform.platform()
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    # RAM (best-effort)
    try:
        import psutil  # type: ignore
        info["ram_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    # Torch device info
    try:
        import torch as _torch
        info["has_cuda"] = bool(_torch.cuda.is_available())
        info["cuda_version"] = getattr(_torch.version, "cuda", None)
        if _torch.cuda.is_available():
            try:
                info["gpu_name"] = _torch.cuda.get_device_name(0)
            except Exception:
                pass
        try:
            info["has_mps"] = bool(getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available())
        except Exception:
            pass
    except Exception:
        pass
    # Threads
    for envk in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        v = os.environ.get(envk)
        if v is not None:
            info[envk.lower()] = v
    return info


def _collect_env_metadata() -> Dict[str, Any]:
    """Collect lightweight environment and git metadata for logging."""
    info: Dict[str, Any] = {"env": {}, "git": {}}
    # Python
    try:
        import sys
        info["env"]["python"] = sys.version.split(" ")[0]
    except Exception:
        pass
    # Library versions
    try:
        import numpy as _np
        info["env"]["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd
        info["env"]["pandas"] = _pd.__version__
    except Exception:
        pass
    try:
        import sklearn as _sk
        info["env"]["scikit_learn"] = _sk.__version__
    except Exception:
        pass
    try:
        import imblearn as _im
        info["env"]["imbalanced_learn"] = _im.__version__
    except Exception:
        pass
    try:
        import matplotlib as _mpl
        info["env"]["matplotlib"] = _mpl.__version__
    except Exception:
        pass
    try:
        import torch as _torch
        info["env"]["torch"] = _torch.__version__
    except Exception:
        pass
    try:
        import yaml as _yaml
        ver = getattr(_yaml, "__version__", None)
        if ver:
            info["env"]["PyYAML"] = ver
    except Exception:
        pass
    try:
        import wandb as _wandb
        info["env"]["wandb"] = _wandb.__version__
    except Exception:
        pass

    # Git metadata (best-effort)
    try:
        import subprocess
        # commit hash (short)
        sha = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["git"]["commit"] = sha
        # dirty flag
        dirty = True
        try:
            subprocess.check_call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.check_call(["git", "diff", "--quiet", "--cached"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            dirty = False
        except Exception:
            dirty = True
        info["git"]["dirty"] = dirty
        # branch (optional)
        try:
            br = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            info["git"]["branch"] = br
        except Exception:
            pass
        # remote URL (origin)
        try:
            remote = subprocess.check_output(["git", "remote", "get-url", "origin"], stderr=subprocess.DEVNULL).decode().strip()
            info["git"]["remote"] = remote
            # Normalize a clickable commit URL for GitHub-style remotes
            try:
                commit_url: Optional[str] = None
                if remote.startswith("git@github.com:"):
                    path = remote.split(":", 1)[1]
                    if path.endswith(".git"):
                        path = path[:-4]
                    commit_url = f"https://github.com/{path}/commit/{sha}"
                elif remote.startswith("https://github.com/"):
                    path = remote.split("https://github.com/", 1)[1]
                    if path.endswith(".git"):
                        path = path[:-4]
                    commit_url = f"https://github.com/{path}/commit/{sha}"
                if commit_url:
                    info["git"]["commit_url"] = commit_url
                # Also a normalized HTTPS repo URL if GitHub
                if remote.startswith("git@github.com:"):
                    path = remote.split(":", 1)[1]
                    if path.endswith(".git"):
                        path = path[:-4]
                    info["git"]["repo_url"] = f"https://github.com/{path}"
                elif remote.startswith("https://github.com/"):
                    repo_https = remote[:-4] if remote.endswith(".git") else remote
                    info["git"]["repo_url"] = repo_https
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        pass
    return info


def _ensure_dirs(paths: List[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _to_dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_config_with_extends(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.get("extends")
    if extends:
        # Resolve base path relative to this config file
        base_candidate = cfg_path.parent / f"{extends}.yaml"
        base_path = base_candidate if base_candidate.exists() else Path(extends)
        base_cfg = _load_config_with_extends(base_path)
        # Child overrides base
        merged = _deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "extends"})
        return merged
    return cfg


class _SimpleHistory:
    """Lightweight history container for plotting curves."""

    def __init__(self, loss: List[float], val_loss: List[float]):
        self.history = {"loss": loss, "val_loss": val_loss}


def _train_with_pytorch(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np: Optional[np.ndarray],
    y_val_np: Optional[np.ndarray],
    X_test_np: np.ndarray,
    y_test_np: np.ndarray,
    model_cfg: Dict[str, Any],
    out_model_path: Path,
    random_state: int,
    pos_label_for_auc: int = 1,
) -> tuple[Dict[str, Any], _SimpleHistory]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split as torch_random_split
    from src.models.torch_nn import MLP as TorchMLP, focal_binary_loss as torch_focal_loss
    # Device selection with optional CPU override (FORCE_CPU=1)
    force_cpu = str(os.environ.get("FORCE_CPU", "")).lower() in {"1", "true", "yes"}
    if force_cpu:
        device = torch.device("cpu")
        try:
            # Keep threading modest for stability in constrained envs
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception:
            pass
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )

    # Build model
    model = TorchMLP(
        input_dim=X_train_np.shape[1],
        layers=model_cfg.get("layers", [256, 128, 64, 32]),
        dropout=model_cfg.get("dropout", [0.4, 0.3, 0.2, 0.2]),
        batchnorm=model_cfg.get("batchnorm", True),
    ).to(device)
    # Optional: allow external watchers (e.g., wandb.watch)
    try:
        on_watch = _train_with_pytorch.on_watch_model  # type: ignore[attr-defined]
    except Exception:
        on_watch = None
    if on_watch is not None:
        try:
            on_watch(model)
        except Exception:
            pass
    try:
        n_params = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        n_params = None  # type: ignore[assignment]

    loss_name = "focal" if model_cfg.get("focal", {}).get("enabled", False) else model_cfg.get("loss", "binary_crossentropy")
    if loss_name == "focal":
        criterion = torch_focal_loss(
            gamma=float(model_cfg.get("focal", {}).get("gamma", 2.0)),
            alpha=float(model_cfg.get("focal", {}).get("alpha", 0.25)),
        )
        use_logits = True
    else:
        # BCE with logits is numerically stable
        # Use reduction='none' so we can apply class weights if provided
        bce_none = torch.nn.BCEWithLogitsLoss(reduction="none")
        criterion = bce_none
        use_logits = True

    optimizer_name = model_cfg.get("optimizer", "adam").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = int(model_cfg.get("epochs", 30))
    batch_size = int(model_cfg.get("batch_size", 128))
    val_split = float(model_cfg.get("val_split", 0.2))
    patience = int(model_cfg.get("early_stopping_patience", 3) or 0)

    X_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)

    # Use provided validation set if available; otherwise split here deterministically
    if X_val_np is not None and y_val_np is not None:
        val_tensor_x = torch.tensor(X_val_np, dtype=torch.float32)
        val_tensor_y = torch.tensor(y_val_np, dtype=torch.float32)
        val_ds = TensorDataset(val_tensor_x, val_tensor_y)
        train_ds = ds
    else:
        n_total = len(ds)
        n_val = int(max(1, round(n_total * val_split)))
        n_train = n_total - n_val
        g = make_torch_generator(random_state)  # deterministic split
        train_ds, val_ds = torch_random_split(ds, [n_train, n_val], generator=g) if n_val > 0 else (ds, None)

    # DataLoaders with seeded workers and deterministic shuffles
    g_loader = make_torch_generator(random_state)
    worker_fn = make_worker_init_fn(random_state)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g_loader, worker_init_fn=worker_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=worker_fn) if val_ds is not None else None

    best_val = float("inf")
    best_state: Optional[dict[str, Any]] = None
    wait = 0
    tr_losses: List[float] = []
    va_losses: List[float] = []
    epoch_stats: List[Dict[str, Any]] = []

    # Optional: class weights (for BCE path). Compute auto if requested via model_cfg["_class_weight"] injected by caller.
    class_weight_cfg = model_cfg.get("_class_weight")
    use_weighted_bce = class_weight_cfg is not None and loss_name != "focal"
    if use_weighted_bce:
        # Expect dict {0: w0, 1: w1}
        w0 = float(class_weight_cfg.get(0, 1.0))
        w1 = float(class_weight_cfg.get(1, 1.0))

    for epoch in range(epochs):
        _ep_start = time.time()
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # Ensure target has shape (N, 1) to match logits
            yb = yb.view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if use_weighted_bce:
                loss_per = criterion(logits, yb)
                sw = yb * w1 + (1.0 - yb) * w0
                loss = (loss_per * sw).mean()
            else:
                # criterion returns per-sample loss; reduce to scalar
                loss = criterion(logits, yb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        tr_losses.append(epoch_loss)

        # Validation
        val_auc = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_targets = []
            val_logits_all = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    yb = yb.view(-1, 1)
                    logits = model(xb)
                    if use_weighted_bce:
                        loss_per = criterion(logits, yb)
                        sw = yb * w1 + (1.0 - yb) * w0
                        loss = (loss_per * sw).mean()
                    else:
                        loss = criterion(logits, yb).mean()
                    val_loss += loss.item() * xb.size(0)
                    # Collect for AUC
                    val_targets.append(yb.detach().cpu())
                    val_logits_all.append(logits.detach().cpu())
            val_loss /= len(val_loader.dataset)
            va_losses.append(val_loss)
            # Compute validation AUC aligned to configured positive class
            try:
                import numpy as _np
                from sklearn.metrics import roc_auc_score as _roc_auc_score
                vl = torch.cat(val_logits_all, dim=0).numpy().reshape(-1)
                vt = torch.cat(val_targets, dim=0).numpy().reshape(-1)
                vprob = 1.0 / (1.0 + _np.exp(-vl))
                # Align to pos_label_for_auc: if 0 => treat defaults as positive
                if int(pos_label_for_auc) == 0:
                    y_true_auc = (1 - vt).astype(int)
                    y_prob_auc = 1.0 - vprob
                else:
                    y_true_auc = vt.astype(int)
                    y_prob_auc = vprob
                val_auc = float(_roc_auc_score(y_true_auc, y_prob_auc))
            except Exception:
                val_auc = None

            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if patience and wait >= patience:
                    break
        else:
            va_losses.append(float("nan"))

        # Optional per-epoch callback (e.g., for W&B logging)
        try:
            from typing import cast as _cast
            on_epoch = _train_with_pytorch.on_epoch  # type: ignore[attr-defined]
        except Exception:
            on_epoch = None
        # Learning rate (first group)
        try:
            lr_val = float(optimizer.param_groups[0].get("lr", 0.0))
        except Exception:
            lr_val = None
        # Epoch duration
        _ep_time = time.time() - _ep_start
        # Accumulate stats
        epoch_stats.append({
            "epoch": int(epoch + 1),
            "loss": float(epoch_loss),
            "val_loss": float(va_losses[-1] if len(va_losses) else float("nan")),
            "val_auc": None if val_auc is None else float(val_auc),
            "lr": lr_val,
            "time_sec": float(_ep_time),
        })
        # Console line (captured by W&B Logs)
        try:
            if val_auc is None:
                print(f"epoch {epoch+1}/{epochs} loss={epoch_loss:.4f} val_loss={va_losses[-1]:.4f} lr={lr_val} time={_ep_time:.2f}s")
            else:
                print(f"epoch {epoch+1}/{epochs} loss={epoch_loss:.4f} val_loss={va_losses[-1]:.4f} val_auc={val_auc:.3f} lr={lr_val} time={_ep_time:.2f}s")
        except Exception:
            pass
        if on_epoch is not None:
            try:
                on_epoch(epoch=epoch + 1, loss=epoch_loss, val_loss=(va_losses[-1] if len(va_losses) else None), val_auc=val_auc, lr=lr_val, epoch_time_sec=_ep_time)
            except Exception:
                pass

    # Restore best weights if available
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on test (and validation if provided)
    model.eval()
    Xt = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(Xt).cpu().numpy().reshape(-1)
    y_prob = 1 / (1 + np.exp(-logits)) if use_logits else logits
    # Optional validation probabilities for threshold selection upstream
    y_prob_val = None
    if X_val_np is not None and y_val_np is not None:
        with torch.no_grad():
            Xv = torch.tensor(X_val_np, dtype=torch.float32).to(device)
            v_logits = model(Xv).cpu().numpy().reshape(-1)
            y_prob_val = 1 / (1 + np.exp(-v_logits)) if use_logits else v_logits
    # Save model
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    import torch as _torch
    _torch.save(model.state_dict(), out_model_path.as_posix())

    history = _SimpleHistory(tr_losses, va_losses)
    return {
        "y_prob": y_prob,
        "y_prob_val": y_prob_val,
        "param_count": n_params,
        "device": str(device),
        "epochs_ran": len(tr_losses),
        "epoch_stats": epoch_stats,
    }, history


def train_from_config(cfg_path: str | Path, notes: Optional[str] = None):
    cfg_path = Path(cfg_path)
    cfg = _load_config_with_extends(cfg_path)

    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    os_cfg = cfg.get("oversampling", {"enabled": True})
    model_cfg = cfg["model"]
    out_cfg = cfg["output"]
    eval_cfg = cfg.get("eval", {})
    training_cfg = cfg.get("training", {})
    tracking_cfg = cfg.get("tracking", {})

    # Determine single-run output directory strategy
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    single_run_dir_mode = bool(out_cfg.get("runs_root"))
    if single_run_dir_mode:
        # Consolidate all artifacts for this run into one gitignored folder
        run_dir = Path(out_cfg["runs_root"]).resolve() / run_id
        models_dir = run_dir
        reports_dir = run_dir
        figures_dir = run_dir / "figures"
        _ensure_dirs([run_dir, figures_dir])
    else:
        models_dir = Path(out_cfg["models_dir"]).resolve()
        reports_dir = Path(out_cfg["reports_dir"]).resolve()
        figures_dir = Path(out_cfg["figures_dir"]).resolve()
        _ensure_dirs([models_dir, reports_dir, figures_dir])

    # Load
    load_config = LoadConfig(
        csv_path=data_cfg["csv_path"],
        target_col=data_cfg["target_col"],
        target_mapping=data_cfg["target_mapping"],
        parse_dates=data_cfg.get("parse_dates", []),
        drop_leakage=data_cfg.get("drop_leakage", True),
        leakage_cols=data_cfg.get("leakage_cols", []),
        features=data_cfg.get("features", []),
    )

    t0 = time.time()
    t_load_start = t0
    df = load_and_prepare(load_config)
    t_load_end = time.time()

    # Select features for modeling; keep time cols only for time-split, not as predictors
    features = list(data_cfg.get("features", []))
    time_cols = set(data_cfg.get("parse_dates", []))

    # Always add engineered features if present
    for eng in ["credit_history_length", "income_to_loan_ratio", "fico_avg", "fico_spread"]:
        if eng in df.columns and eng not in features:
            features.append(eng)

    # Remove target and time columns from feature inputs
    feature_inputs = [
        c
        for c in features
        if c != data_cfg["target_col"] and c not in time_cols and c in df.columns
    ]

    # Split
    t_split_start = time.time()
    method = split_cfg.get("method", "random")
    if method == "time":
        time_col = split_cfg.get("time_col", "issue_d")
        train_df, test_df = time_based_split(df, time_col=time_col, test_size=split_cfg.get("test_size", 0.2))
        X_train = train_df[feature_inputs]
        y_train = train_df[data_cfg["target_col"]]
        X_test = test_df[feature_inputs]
        y_test = test_df[data_cfg["target_col"]]
    else:
        X = df[feature_inputs]
        y = df[data_cfg["target_col"]]
        X_train, X_test, y_train, y_test = random_split(
            X,
            y,
            test_size=split_cfg.get("test_size", 0.2),
            random_state=split_cfg.get("random_state", 42),
            stratify=True,
        )
        train_df = None
        test_df = None
    t_split_end = time.time()

    # Carve validation from training BEFORE preprocessing and oversampling
    from sklearn.model_selection import train_test_split as _tts
    val_split = float(model_cfg.get("val_split", 0.2))
    X_train_df, X_val_df, y_train_s, y_val_s = _tts(
        X_train,
        y_train,
        test_size=val_split if val_split > 0 else 0.0,
        random_state=int(split_cfg.get("random_state", 42)),
        stratify=y_train if val_split > 0 else None,
    ) if val_split > 0 else (X_train, None, y_train, None)

    # Build and fit preprocessor on the training subset only
    num_cols, cat_cols = identify_feature_types(X_train_df)
    preproc = build_preprocessor(num_cols, cat_cols)

    X_train_proc = preproc.fit_transform(X_train_df)
    X_val_proc = preproc.transform(X_val_df) if val_split > 0 else None
    X_test_proc = preproc.transform(X_test)

    # Oversample only the training subset to avoid leakage
    if os_cfg.get("enabled", True):
        ros = RandomOverSampler(random_state=split_cfg.get("random_state", 42))
        X_train_proc, y_train_s = ros.fit_resample(X_train_proc, y_train_s)

    # Convert to dense for downstream model training
    X_train_np = _to_dense(X_train_proc)
    X_val_np = _to_dense(X_val_proc) if X_val_proc is not None else None
    X_test_np = _to_dense(X_test_proc)
    y_train_np = np.asarray(y_train_s)
    y_val_np = np.asarray(y_val_s) if y_val_s is not None else None
    y_test_np = np.asarray(y_test)
    t_preproc_end = time.time()

    # Backend: PyTorch only
    model_backend = "pytorch"

    # Prepare output model filename/extension for PyTorch
    model_filename = out_cfg.get("model_filename", "loan_default_model.pt")
    if not str(model_filename).endswith(".pt"):
        model_filename = str(model_filename).rsplit(".", 1)[0] + ".pt"
    model_path = models_dir / model_filename

    # Optional class weights for BCE
    class_weight_cfg = training_cfg.get("class_weight")
    cw_resolved = None
    if class_weight_cfg is not None:
        if isinstance(class_weight_cfg, str) and class_weight_cfg.lower() == "auto":
            # compute from y_train distribution
            n = float(len(y_train_np))
            n1 = float((y_train_np == 1).sum())
            n0 = n - n1
            # Inverse frequency normalized to mean 1
            w0 = n / (2.0 * max(n0, 1.0))
            w1 = n / (2.0 * max(n1, 1.0))
            cw_resolved = {0: w0, 1: w1}
        elif isinstance(class_weight_cfg, dict):
            try:
                cw_resolved = {int(k): float(v) for k, v in class_weight_cfg.items()}
            except Exception:
                cw_resolved = None
        # Inject into model_cfg for the torch path helper
        if cw_resolved is not None:
            model_cfg = dict(model_cfg)
            model_cfg["_class_weight"] = cw_resolved

    # Optional: W&B initialization
    wandb_run = None
    wandb_enabled = False
    try:
        tracking_backend = str(tracking_cfg.get("backend", "none")).lower()
        if tracking_backend == "wandb" or (tracking_cfg.get("wandb", {}).get("enabled")):
            wandb_enabled = True
            import wandb  # type: ignore
            wb_cfg = tracking_cfg.get("wandb", {})
            mode = str(wb_cfg.get("mode", "online"))
            # Optional login via env var (preferred in headless/CI)
            try:
                api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WB_API_KEY")
                if api_key:
                    try:
                        wandb.login(key=api_key)
                    except Exception:
                        pass
            except Exception:
                pass
            if wb_cfg.get("ignore_globs"):
                # Space-separated patterns per W&B convention
                os.environ["WANDB_IGNORE_GLOBS"] = " ".join(wb_cfg.get("ignore_globs", []))
            # Derive default group/job_type for organization
            csv_base_init = Path(str(data_cfg.get("csv_path", ""))).stem
            split_method_init = split_cfg.get("method", "time")
            pos_label_init = eval_cfg.get("pos_label", 1)
            if isinstance(pos_label_init, str):
                pos_label_init = 0 if str(pos_label_init).lower() in {"default", "charged off", "charged_off"} else 1
            pos_tok_init = "co" if int(pos_label_init) == 0 else "fp"
            default_group = f"{csv_base_init}|{split_method_init}|{pos_tok_init}"
            # Template context available pre-training
            try:
                sha_init = _collect_env_metadata().get("git", {}).get("commit")
            except Exception:
                sha_init = None
            ctx_init = {
                "dataset": csv_base_init,
                "split": split_method_init,
                "pos": pos_tok_init,
                "sha": sha_init or "",
            }
            # Render group/job_type from templates if provided
            group_val = wb_cfg.get("group")
            if not group_val:
                tmpl = wb_cfg.get("group_template")
                if tmpl:
                    try:
                        group_val = str(tmpl).format(**ctx_init)
                    except Exception:
                        group_val = default_group
                else:
                    group_val = default_group
            job_type_val = wb_cfg.get("job_type")
            if not job_type_val:
                tmpl = wb_cfg.get("job_type_template")
                if tmpl:
                    try:
                        job_type_val = str(tmpl).format(**ctx_init)
                    except Exception:
                        job_type_val = "train"
                else:
                    job_type_val = "train"
            # Allow project/entity from env when not provided in config
            _entity_env = os.environ.get("WANDB_ENTITY") or os.environ.get("WB_ENTITY")
            _project_env = os.environ.get("WANDB_PROJECT")
            wandb_run = wandb.init(
                project=wb_cfg.get("project") or _project_env or "loan-risk-mlp",
                entity=wb_cfg.get("entity") or _entity_env or None,
                config=cfg,
                mode=mode,
                group=group_val,
                job_type=job_type_val,
                settings=wandb.Settings(code_dir=None),
            )
            # Define epoch as step and map metrics to it for clean charts
            try:
                wandb.define_metric("epoch")
                for _m in ["loss", "val_loss", "val_auc", "lr", "epoch_time_sec"]:
                    wandb.define_metric(_m, step_metric="epoch")
            except Exception:
                pass
            # Expose basic identifiers for downstream consumers (results/auto-pull)
            try:
                _wb = wandb.run
                wb_id = getattr(_wb, "id", None)
                wb_path = "/".join([p for p in (getattr(_wb, "entity", None), getattr(_wb, "project", None), getattr(_wb, "id", None)) if p])
                wb_url = getattr(_wb, "url", None)
            except Exception:
                wb_id = None
                wb_path = None
                wb_url = None
    except Exception:
        wandb_enabled = False
        wandb_run = None

    # Hook per-epoch logging if W&B is active
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            def _wb_on_epoch(epoch: int, loss: float, val_loss: Optional[float] = None, **kwargs: Any) -> None:
                data: Dict[str, Any] = {"epoch": int(epoch), "loss": float(loss)}
                if val_loss is not None and not (isinstance(val_loss, float) and np.isnan(val_loss)):
                    data["val_loss"] = float(val_loss)
                for k, v in (kwargs or {}).items():
                    if v is None:
                        continue
                    try:
                        data[k] = float(v) if isinstance(v, (int, float)) else v
                    except Exception:
                        pass
                wandb.log(data, step=int(epoch))

            # attach as attribute to avoid changing function signature broadly
            _train_with_pytorch.on_epoch = _wb_on_epoch  # type: ignore[attr-defined]
            # Optional: watch model gradients lightly
            def _wb_watch_model(m: Any) -> None:
                try:
                    wandb.watch(m, log="gradients", log_freq=max(1, int(model_cfg.get("epochs", 30)) // 10))
                except Exception:
                    pass
            _train_with_pytorch.on_watch_model = _wb_watch_model  # type: ignore[attr-defined]
        except Exception:
            pass

    t_train_start = time.time()
    # Seed Python/NumPy/Torch for reproducibility
    set_seed(int(split_cfg.get("random_state", 42)))

    # Resolve configured positive label early so epoch val_auc aligns to it
    _pos_cfg = eval_cfg.get("pos_label", 1)
    if isinstance(_pos_cfg, str):
        _pos_cfg = 0 if str(_pos_cfg).lower() in {"default", "charged off", "charged_off"} else 1
    pos_label_for_auc = int(_pos_cfg)

    result, history_obj = _train_with_pytorch(
        X_train_np,
        y_train_np,
        X_val_np,
        y_val_np,
        X_test_np,
        y_test_np,
        model_cfg,
        model_path,
        int(split_cfg.get("random_state", 42)),
        pos_label_for_auc,
    )
    y_prob = result["y_prob"]
    param_count = result.get("param_count") if isinstance(result, dict) else None
    device_used = result.get("device") if isinstance(result, dict) else None
    epochs_ran = result.get("epochs_ran") if isinstance(result, dict) else None
    t_train_end = time.time()

    # Evaluation controls
    pos_label_cfg = eval_cfg.get("pos_label", 1)
    if isinstance(pos_label_cfg, str):
        pos_label_cfg = 0 if pos_label_cfg.lower() in {"default", "charged off", "charged_off"} else 1

    if int(pos_label_cfg) == 1:
        y_true_pos_test = y_test_np.astype(int)
        y_prob_pos_test = y_prob
        pos_label_name = "positive=1 (Fully Paid)"
    else:
        # Treat defaults as positive (label=0 in data mapping)
        y_true_pos_test = (1 - y_test_np).astype(int)
        y_prob_pos_test = 1.0 - y_prob
        pos_label_name = "positive=default (Charged Off)"

    # Threshold strategy
    thr_cfg = eval_cfg.get("threshold", {})
    strategy = str(thr_cfg.get("strategy", "fixed")).lower()
    # Choose threshold on validation and apply to test
    # If no explicit validation set was provided, fall back to test (legacy behavior)
    threshold: float
    y_prob_val = result.get("y_prob_val") if isinstance(result, dict) else None
    if y_prob_val is not None and y_val_np is not None:
        if int(pos_label_cfg) == 1:
            y_true_pos_val = y_val_np.astype(int)
            y_prob_pos_val = y_prob_val
        else:
            y_true_pos_val = (1 - y_val_np).astype(int)
            y_prob_pos_val = 1.0 - np.asarray(y_prob_val)
        if strategy == "youden_j":
            threshold = choose_threshold_youden(y_true_pos_val, y_prob_pos_val)
        elif strategy in {"f1", "f1_max", "max_f1"}:
            threshold = choose_threshold_f1(y_true_pos_val, y_prob_pos_val)
        else:
            threshold = float(thr_cfg.get("value", 0.5))
    else:
        if strategy == "youden_j":
            threshold = choose_threshold_youden(y_true_pos_test, y_prob_pos_test)
        elif strategy in {"f1", "f1_max", "max_f1"}:
            threshold = choose_threshold_f1(y_true_pos_test, y_prob_pos_test)
        else:
            threshold = float(thr_cfg.get("value", 0.5))

    # Compute metrics and plots using chosen positive class and threshold
    metrics = compute_metrics_binary(y_true_pos_test, y_prob_pos_test, threshold=threshold)

    # Save common artifacts (latest)
    save_metrics(metrics, reports_dir / "metrics.json")
    plot_learning_curves(history_obj, figures_dir / "learning_curves.png")
    cm = confusion_metrics_at_threshold(y_true_pos_test, y_prob_pos_test, threshold)
    plot_roc_curve(y_true_pos_test, y_prob_pos_test, figures_dir / "roc_curve.png", point=(cm["fpr"], cm["tpr"]))
    plot_pr_curve(y_true_pos_test, y_prob_pos_test, figures_dir / "pr_curve.png", point=(cm["precision"], cm["recall"]))

    # Save confusion metrics
    cm_path = reports_dir / "confusion.json"
    with open(cm_path, "w", encoding="utf-8") as f:
        _json.dump(cm, f, indent=2)
    # W&B: log an interactive confusion matrix visualization
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            import numpy as _np  # local alias to avoid shadowing
            y_pred_pos = (_np.asarray(y_prob_pos_test) >= float(threshold)).astype(int)
            # Class names reflecting the configured positive class
            if int(pos_label_cfg) == 0:
                # 0 => Charged Off is the positive class in our mapping above
                class_names = ["fully_paid", "charged_off"]  # index 0,1 align to y_true_pos
            else:
                class_names = ["charged_off", "fully_paid"]
            cm_plot = wandb.plot.confusion_matrix(
                y_true=_np.asarray(y_true_pos_test).astype(int),
                preds=y_pred_pos.astype(int),
                class_names=class_names,
            )
            wandb.log({"confusion_matrix": cm_plot})
        except Exception:
            pass

    # Compute and save ROC/PR point sweeps as CSV in run folder later

    # Per-run summary folder
    if single_run_dir_mode:
        # Already created above
        run_fig_dir = figures_dir
    else:
        run_dir = (reports_dir / "runs" / run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_fig_dir = run_dir / "figures"
        run_fig_dir.mkdir(parents=True, exist_ok=True)

    # Duplicate from latest folders only when not using single-run directory layout
    if not single_run_dir_mode:
        try:
            # Figures
            for fname in ["learning_curves.png", "roc_curve.png", "pr_curve.png"]:
                src = figures_dir / fname
                if src.exists():
                    shutil.copy2(src, run_fig_dir / fname)
            # Metrics + confusion
            shutil.copy2(reports_dir / "metrics.json", run_dir / "metrics.json")
            shutil.copy2(cm_path, run_dir / "confusion.json")
            # Model
            if model_path.exists():
                shutil.copy2(model_path, run_dir / model_path.name)
        except Exception:
            pass

    # Save per-threshold sweeps for ROC and PR (CSV)
    try:
        # Use the test-set positives per configured pos_label
        fpr, tpr, thr_roc = roc_curve(y_true_pos_test, y_prob_pos_test)
        with open(run_dir / "roc_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,fpr,tpr\n")
            # First ROC point corresponds to no threshold (0,0); leave threshold blank
            for i in range(len(fpr)):
                th = "" if i == 0 else float(thr_roc[i - 1])
                f.write(f"{th},{float(fpr[i])},{float(tpr[i])}\n")
        prec, rec, thr_pr = precision_recall_curve(y_true_pos_test, y_prob_pos_test)
        with open(run_dir / "pr_points.csv", "w", encoding="utf-8") as f:
            f.write("threshold,precision,recall\n")
            # Precision-Recall pairs are length N; thresholds length N-1; align accordingly
            # Write the baseline point first with empty threshold
            if len(prec) > 0:
                f.write(f",{float(prec[0])},{float(rec[0])}\n")
            for i in range(1, len(prec)):
                th = "" if i - 1 >= len(thr_pr) else float(thr_pr[i - 1])
                f.write(f"{th},{float(prec[i])},{float(rec[i])}\n")
        # Also emit a richer threshold metrics sweep for convenience
        try:
            import numpy as _np
            from src.eval.metrics import confusion_metrics_at_threshold as _cm_thr
            thr_grid = _np.linspace(0.0, 1.0, 101)
            with open(run_dir / "threshold_metrics.csv", "w", encoding="utf-8") as f:
                f.write("threshold,precision,recall,tpr,fpr,specificity,f1\n")
                for th in thr_grid:
                    m = _cm_thr(_np.asarray(y_true_pos_test).astype(int), _np.asarray(y_prob_pos_test), float(th))
                    prec_v = float(m.get("precision", 0.0))
                    rec_v = float(m.get("recall", 0.0))
                    tpr_v = float(m.get("tpr", 0.0))
                    fpr_v = float(m.get("fpr", 0.0))
                    spec_v = 1.0 - fpr_v
                    f1_v = (2 * prec_v * rec_v) / (prec_v + rec_v + 1e-12)
                    f.write(f"{th:.4f},{prec_v:.6f},{rec_v:.6f},{tpr_v:.6f},{fpr_v:.6f},{spec_v:.6f},{f1_v:.6f}\n")
        except Exception:
            pass
    except Exception:
        pass

    # W&B: also log ROC/PR sweeps and threshold metrics as interactive panels
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            import numpy as _np
            # Built-in ROC/PR plots (interactive)
            try:
                wandb.log({
                    "roc_curve": wandb.plot.roc_curve(_np.asarray(y_true_pos_test).astype(int), _np.asarray(y_prob_pos_test)),
                    "pr_curve": wandb.plot.pr_curve(_np.asarray(y_true_pos_test).astype(int), _np.asarray(y_prob_pos_test)),
                })
            except Exception:
                pass
            # Tables for ROC/PR points
            fpr, tpr, thr_roc = roc_curve(y_true_pos_test, y_prob_pos_test)
            roc_tbl = wandb.Table(
                data=[[float(_np.nan if i == 0 else thr_roc[i - 1]), float(fpr[i]), float(tpr[i])] for i in range(len(fpr))],
                columns=["threshold", "fpr", "tpr"],
            )
            prec, rec, thr_pr = precision_recall_curve(y_true_pos_test, y_prob_pos_test)
            pr_tbl = wandb.Table(
                data=(([[float(_np.nan), float(prec[0]), float(rec[0])]] if len(prec) > 0 else []) +
                      [[float(thr_pr[i - 1]), float(prec[i]), float(rec[i])] for i in range(1, len(prec))]),
                columns=["threshold", "precision", "recall"],
            )
            # Threshold metrics sweep (precision/recall/specificity/f1 vs threshold)
            try:
                from src.eval.metrics import confusion_metrics_at_threshold as _cm_thr
                thr_grid = _np.linspace(0.0, 1.0, 101)
                rows = []
                for th in thr_grid:
                    m = _cm_thr(_np.asarray(y_true_pos_test).astype(int), _np.asarray(y_prob_pos_test), float(th))
                    prec_v = float(m.get("precision", 0.0))
                    rec_v = float(m.get("recall", 0.0))
                    tpr_v = float(m.get("tpr", 0.0))
                    fpr_v = float(m.get("fpr", 0.0))
                    spec_v = 1.0 - fpr_v
                    f1_v = (2 * prec_v * rec_v) / (prec_v + rec_v + 1e-12)
                    rows.append([float(th), prec_v, rec_v, tpr_v, fpr_v, spec_v, f1_v])
                thr_tbl = wandb.Table(data=rows, columns=["threshold", "precision", "recall", "tpr", "fpr", "specificity", "f1"])
                # Line charts for threshold sweeps
                try:
                    wandb.log({
                        "threshold_precision": wandb.plot.line_series(xs=[r[0] for r in rows], ys=[[r[1] for r in rows]], keys=["precision"], title="Precision vs Threshold", xname="threshold"),
                        "threshold_recall": wandb.plot.line_series(xs=[r[0] for r in rows], ys=[[r[2] for r in rows]], keys=["recall"], title="Recall vs Threshold", xname="threshold"),
                        "threshold_f1": wandb.plot.line_series(xs=[r[0] for r in rows], ys=[[r[6] for r in rows]], keys=["f1"], title="F1 vs Threshold", xname="threshold"),
                    })
                except Exception:
                    pass
            except Exception:
                thr_tbl = None
            log_payload = {"roc_table": roc_tbl, "pr_table": pr_tbl}
            if thr_tbl is not None:
                log_payload["threshold_metrics_table"] = thr_tbl
            wandb.log(log_payload)
        except Exception:
            pass

    # Save resolved config used for this run
    resolved_cfg_path = run_dir / "config_resolved.yaml"
    try:
        with open(resolved_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception:
        pass

    # Snapshot current Python environment (pip freeze)
    try:
        import subprocess
        freeze_txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL).decode()
        (run_dir / "requirements.freeze.txt").write_text(freeze_txt, encoding="utf-8")
    except Exception:
        pass

    # Save data manifest with provenance and date ranges
    dataset_info: Dict[str, Any] | None = None
    try:
        csv_path_raw = str(data_cfg.get("csv_path", ""))
        csv_path = Path(csv_path_raw)
        csv_abs = csv_path.resolve()
        manifest: Dict[str, Any] = {
            "csv_path": csv_path_raw,
            "csv_path_abs": csv_abs.as_posix(),
        }
        try:
            st = csv_abs.stat()
            manifest.update(
                {
                    "filesize_bytes": int(st.st_size),
                    "mtime": int(st.st_mtime),
                    "sha256": _file_sha256(csv_abs),
                }
            )
        except Exception:
            pass

        # Dataset-level stats
        try:
            y_series = df[data_cfg["target_col"]]
            counts = y_series.value_counts().to_dict()
            manifest["n_rows"] = int(df.shape[0])
            manifest["n_cols"] = int(df.shape[1])
            manifest["class_counts"] = {int(k): int(v) for k, v in counts.items()}
        except Exception:
            pass

        # Date ranges
        try:
            time_col = split_cfg.get("time_col", "issue_d")
            if time_col in df.columns:
                def _fmt_range(s):
                    s = s.dropna()
                    if s.empty:
                        return {"min": None, "max": None}
                    return {"min": str(s.min().date()), "max": str(s.max().date())}

                manifest["date_ranges"] = {"dataset": _fmt_range(df[time_col])}
                if train_df is not None and time_col in train_df.columns:
                    manifest["date_ranges"]["train"] = _fmt_range(train_df[time_col])
                if test_df is not None and time_col in test_df.columns:
                    manifest["date_ranges"]["test"] = _fmt_range(test_df[time_col])
        except Exception:
            pass

        # Train/test class counts
        try:
            import numpy as _np
            def _counts(arr):
                unique, cnts = _np.unique(arr.astype(int), return_counts=True)
                return {int(k): int(v) for k, v in zip(unique, cnts)}
            manifest["train_class_counts"] = _counts(y_train_np)
            manifest["test_class_counts"] = _counts(y_test_np)
        except Exception:
            pass

        with open(run_dir / "data_manifest.json", "w", encoding="utf-8") as f:
            _json.dump(manifest, f, indent=2)
        dataset_info = manifest
    except Exception:
        pass

    # Save feature lists used by preprocessor
    try:
        features_manifest = {
            "numerical_features": list(num_cols),
            "categorical_features": list(cat_cols),
            "feature_inputs": list(feature_inputs),
        }
        with open(run_dir / "features.json", "w", encoding="utf-8") as f:
            _json.dump(features_manifest, f, indent=2)
    except Exception:
        pass

    # Save history CSV
    try:
        import csv as _csv
        hist_csv = run_dir / "history.csv"
        with open(hist_csv, "w", newline="", encoding="utf-8") as hf:
            writer = _csv.writer(hf)
            writer.writerow(["epoch", "loss", "val_loss"])
            tr = history_obj.history.get("loss", [])
            va = history_obj.history.get("val_loss", [])
            for i in range(max(len(tr), len(va))):
                l = tr[i] if i < len(tr) else ""
                vl = va[i] if i < len(va) else ""
                writer.writerow([i + 1, l, vl])
    except Exception:
        pass
    # Save verbose training log (per-epoch)
    try:
        ep_stats = result.get("epoch_stats") if isinstance(result, dict) else None
        if ep_stats:
            log_path = run_dir / "training.log"
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("epoch,loss,val_loss,val_auc,lr,time_sec\n")
                for s in ep_stats:
                    lf.write(
                        f"{s.get('epoch')},{s.get('loss')},{s.get('val_loss')},{s.get('val_auc')},{s.get('lr')},{s.get('time_sec')}\n"
                    )
    except Exception:
        pass
    # Compute model size for summary table
    try:
        _model_size = int((run_dir / model_path.name).stat().st_size)
    except Exception:
        _model_size = None

    # Human-readable start/end timestamps (UTC)
    t_eval_end = time.time()
    _start_iso = datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(timespec="seconds")
    _end_iso = datetime.fromtimestamp(t_eval_end, tz=timezone.utc).isoformat(timespec="seconds")

    # Compose README content (rich template)
    notes_text = (notes or "").strip()
    summary_lines = [
        f"# Training Summary — {run_id}",
        "",
        f"Config: `{cfg_path}`",
        f"Backend: {model_backend}",
        f"Positive class: {pos_label_name}",
        f"Threshold strategy: {strategy}",
        f"Chosen threshold: {threshold:.6f}",
        "",
        "## Run Summary",
        "",
        "| Key | Value |",
        "| --- | --- |",
        f"| Device | {device_used or 'n/a'} |",
        f"| Epochs (ran) | {int(epochs_ran) if epochs_ran is not None else 'n/a'} |",
        f"| Param count | {int(param_count) if param_count is not None else 'n/a'} |",
        f"| Model size | {(_model_size/1024):.1f} KB |" if _model_size is not None else "| Model size | n/a |",
        f"| Start (UTC) | {_start_iso} |",
        f"| End (UTC) | {_end_iso} |",
        f"| Total time | {(t_eval_end - t0):.2f} s |",
        f"| Load | {(t_load_end - t_load_start):.2f} s |",
        f"| Split | {(t_split_end - t_split_start):.2f} s |",
        f"| Preprocess | {(t_preproc_end - t_split_end):.2f} s |",
        f"| Train | {(t_train_end - t_preproc_end):.2f} s |",
        f"| Eval | {(t_eval_end - t_train_end):.2f} s |",
        "",
        "## What Changed",
        notes_text if notes_text else "(no notes provided)",
        "",
        "## Metrics",
        f"- ROC AUC: {metrics.get('roc_auc'):.3f}",
        f"- Average Precision: {metrics.get('average_precision'):.3f}",
        f"- Precision (at threshold): {cm['precision']:.3f}",
        f"- Recall (TPR): {cm['recall']:.3f}",
        f"- Specificity (TNR): {1.0 - cm['fpr']:.3f}",
        f"- Confusion: TP={int(cm['tp'])}, FP={int(cm['fp'])}, TN={int(cm['tn'])}, FN={int(cm['fn'])}",
        f"- n_train: {int(len(y_train_np))}",
        f"- n_test: {int(len(y_test_np))}",
        f"- n_features: {int(X_train_np.shape[1])}",
        "",
        "## Classification Report (at threshold)",
        "```json",
        _json.dumps(metrics.get("classification_report", {}), indent=2),
        "```",
        "",
        "## Artifacts",
        f"- Model: `{model_path.name}`",
        f"- Metrics: `metrics.json`",
        f"- Confusion: `confusion.json`",
        f"- History CSV: `history.csv`",
        f"- ROC points CSV: `roc_points.csv`",
        f"- PR points CSV: `pr_points.csv`",
        f"- Learning curves: `figures/learning_curves.png`",
        f"- ROC curve: `figures/roc_curve.png`",
        f"- PR curve: `figures/pr_curve.png`",
        f"- Resolved config: `config_resolved.yaml`",
        f"- Features manifest: `features.json`",
        "",
        "## Notes",
        ("- Evaluated defaults as the positive class." if int(pos_label_cfg) == 0 else "- Evaluated fully paid as the positive class."),
        "- Threshold selected according to configured strategy and annotated on curves.",
    ]
    # Add a simple threshold sanity note for extreme operating points
    try:
        if cm.get("precision", 1.0) < 1e-3 or cm.get("recall", 1.0) < 1e-3:
            summary_lines.append("")
            summary_lines.append(
                "> Note: precision or recall is near 0 at the chosen threshold. Consider revising the threshold strategy or dataset balance."
            )
    except Exception:
        pass

    # Instead of saving README locally, upload as a W&B artifact file
    readme_content = "\n".join(summary_lines)
    t_eval_end = time.time()

    # W&B: log summary metrics and selected artifacts
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            # Set a friendly run name if not provided
            wb_cfg = tracking_cfg.get("wandb", {})
            if wb_cfg.get("run_name"):
                wandb.run.name = str(wb_cfg.get("run_name"))
            else:
                wandb.run.name = run_id
            # Tags for fast filtering
            try:
                tag_list = [
                    f"backend:{model_backend}",
                    f"split:{split_cfg.get('method', 'time')}",
                    f"threshold:{strategy}",
                    f"pos_label:{int(pos_label_cfg)}",
                ]
                csv_base = Path(str(data_cfg.get("csv_path", ""))).name
                if csv_base:
                    tag_list.append(f"data:{csv_base}")
                wandb.run.tags = list({*list(wandb.run.tags or []), *tag_list})  # type: ignore[attr-defined]
            except Exception:
                pass
            # Log scalar metrics
            log_payload = {
                "roc_auc": float(metrics.get("roc_auc", float("nan")) or float("nan")),
                "average_precision": float(metrics.get("average_precision", float("nan")) or float("nan")),
                "threshold": float(threshold),
                "precision_at_thr": float(cm.get("precision", float("nan")) or float("nan")),
                "recall_at_thr": float(cm.get("recall", float("nan")) or float("nan")),
                "specificity_at_thr": float(1.0 - cm.get("fpr", 0.0)),
                "n_train": int(len(y_train_np)),
                "n_test": int(len(y_test_np)),
                "n_features": int(X_train_np.shape[1]),
            }
            wandb.log(log_payload)

            # Enrich summary and config with metadata
            try:
                env_meta = _collect_env_metadata()
                sys_meta = _collect_system_info()
                # model size on disk
                model_size = None
                try:
                    st = (run_dir / model_path.name).stat()
                    model_size = int(st.st_size)
                except Exception:
                    pass
                wandb.summary.update({
                    "run_id": run_id,
                    "param_count": int(param_count) if param_count is not None else None,
                    "model_size_bytes": model_size,
                    "model_filename": model_path.name,
                    "threshold_strategy": strategy,
                    "pos_label_name": pos_label_name,
                    "device.used": device_used,
                    "epochs_ran": int(epochs_ran) if epochs_ran is not None else None,
                    # Timing breakdown
                    "timing.total_sec": float(t_eval_end - t0),
                    "timing.load_sec": float(t_load_end - t_load_start),
                    "timing.split_sec": float(t_split_end - t_split_start),
                    "timing.preprocess_sec": float(t_preproc_end - t_split_end),
                    "timing.train_sec": float(t_train_end - t_preproc_end),
                    "timing.eval_sec": float(t_eval_end - t_train_end),
                    # Start/end timestamps
                    "time.start_epoch": int(t0),
                    "time.end_epoch": int(t_eval_end),
                    "time.start_iso": datetime.fromtimestamp(t0, tz=timezone.utc).isoformat(timespec="seconds"),
                    "time.end_iso": datetime.fromtimestamp(t_eval_end, tz=timezone.utc).isoformat(timespec="seconds"),
                })
                # env versions and git
                for k, v in (env_meta.get("env", {}) or {}).items():
                    wandb.summary.update({f"env.{k}": v})
                for k, v in (env_meta.get("git", {}) or {}).items():
                    wandb.summary.update({f"git.{k}": v})
                # system info
                for k, v in (sys_meta or {}).items():
                    wandb.summary.update({f"system.{k}": v})
                # add commit as tag, if present
                try:
                    sha = env_meta.get("git", {}).get("commit")
                    if sha:
                        wandb.run.tags = list({*list(wandb.run.tags or []), f"commit:{sha}"})  # type: ignore[attr-defined]
                except Exception:
                    pass
                if dataset_info:
                    # add a compact subset of dataset manifest
                    ds = dataset_info
                    wandb.summary.update({
                        "data.csv_path": ds.get("csv_path"),
                        "data.sha256": ds.get("sha256"),
                        "data.n_rows": ds.get("n_rows"),
                        "data.n_cols": ds.get("n_cols"),
                        "data.class_counts": ds.get("class_counts"),
                        "data.date_ranges": ds.get("date_ranges"),
                    })
            except Exception:
                pass

            # Log W&B-native plots (no PNG uploads)
            try:
                # Learning curves as a multi-series line chart
                try:
                    tr = history_obj.history.get("loss", [])
                    va = history_obj.history.get("val_loss", [])
                    npts = max(len(tr), len(va))
                    xs = list(range(1, npts + 1))
                    ys = []
                    keys = []
                    if tr:
                        ys.append([float(x) for x in tr])
                        keys.append("loss")
                    if va:
                        ys.append([float(x) for x in va])
                        keys.append("val_loss")
                    if ys:
                        plot = wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title="Learning Curves", xname="epoch")
                        wandb.log({"learning_curves_plot": plot})
                except Exception:
                    pass
                # ROC and PR curves from raw predictions
                try:
                    from sklearn.metrics import roc_curve as _roc_curve, precision_recall_curve as _pr_curve
                    _fpr, _tpr, _ = _roc_curve(y_true_pos_test, y_prob_pos_test)
                    roc_table = wandb.Table(columns=["fpr", "tpr"])  # type: ignore[attr-defined]
                    for i in range(len(_fpr)):
                        roc_table.add_data(float(_fpr[i]), float(_tpr[i]))
                    roc_plot = wandb.plot.line(roc_table, "fpr", "tpr", title="ROC Curve")
                    wandb.log({"roc_curve_plot": roc_plot})
                except Exception:
                    pass
                try:
                    from sklearn.metrics import precision_recall_curve as _pr_curve
                    _prec, _rec, _ = _pr_curve(y_true_pos_test, y_prob_pos_test)
                    pr_table = wandb.Table(columns=["recall", "precision"])  # type: ignore[attr-defined]
                    for i in range(len(_prec)):
                        pr_table.add_data(float(_rec[i]), float(_prec[i]))
                    pr_plot = wandb.plot.line(pr_table, "recall", "precision", title="Precision-Recall Curve")
                    wandb.log({"pr_curve_plot": pr_plot})
                except Exception:
                    pass
            except Exception:
                pass

            # Refine run name using template or default descriptive pattern
            try:
                csv_base = Path(str(data_cfg.get("csv_path", ""))).stem
                split_method = split_cfg.get("method", "time")
                pos_tok = "co" if int(pos_label_cfg) == 0 else "fp"
                layers = model_cfg.get("layers")
                layers_str = "-".join(str(x) for x in (layers or [])) if isinstance(layers, (list, tuple)) else str(layers)
                nf = int(X_train_np.shape[1])
                auc = float(metrics.get("roc_auc", float("nan")))
                # optional commit for template
                sha = None
                try:
                    sha = _collect_env_metadata().get("git", {}).get("commit")
                except Exception:
                    sha = None
                template = wb_cfg.get("run_name_template")
                if template:
                    ctx = {
                        "dataset": csv_base,
                        "split": split_method,
                        "pos": pos_tok,
                        "layers": layers_str,
                        "nf": nf,
                        "auc": auc,
                        "sha": sha or "",
                        "run_id": run_id,
                    }
                    name = str(template).format(**ctx)
                else:
                    name = f"{csv_base}|{split_method}|{pos_tok}|mlp[{layers_str}]|nf{nf}|auc{auc:.3f}"
                if len(name) > 120:
                    name = name[:120]
                wandb.run.name = name
                # Also render tag templates and add static tags
                try:
                    tags_cfg = wb_cfg.get("tags", []) or []
                    tag_tmps = wb_cfg.get("tag_templates", []) or []
                    # Build the same context used above
                    ctx = {
                        "dataset": csv_base,
                        "split": split_method,
                        "pos": pos_tok,
                        "layers": layers_str,
                        "nf": nf,
                        "auc": auc,
                        "sha": sha or "",
                        "run_id": run_id,
                    }
                    rendered = []
                    for t in tag_tmps:
                        try:
                            rendered.append(str(t).format(**ctx))
                        except Exception:
                            continue
                    current = list(wandb.run.tags or [])  # type: ignore[attr-defined]
                    wandb.run.tags = list({*current, *tags_cfg, *rendered})  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                pass

            # Log a lightweight artifact with key files
            if bool(wb_cfg.get("log_artifacts", True)):
                art = wandb.Artifact(name=run_id, type="run")
                if notes_text:
                    art.description = f"Run notes: {notes_text}"
                for rel in [
                    "metrics.json",
                    "confusion.json",
                    "config_resolved.yaml",
                    "requirements.freeze.txt",
                    "features.json",
                    "history.csv",
                    "training.log",
                    # include the model weights saved for this run
                    f"{model_path.name}",
                ]:
                    p = run_dir / rel
                    if p.exists():
                        art.add_file(p.as_posix(), name=rel)
                # Add README.md dynamically from in-memory content via temp file
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as _td:
                        _rp = Path(_td) / "README.md"
                        _rp.write_text(readme_content, encoding="utf-8")
                        art.add_file(_rp.as_posix(), name="README.md")
                except Exception:
                    pass
                wandb.log_artifact(art)

            # Also log a versioned "model" artifact with metadata and aliases
            try:
                model_file = run_dir / model_path.name
                if model_file.exists():
                    model_art = wandb.Artifact(
                        name="loan-default",  # stable name; W&B versions it
                        type="model",
                        metadata={
                            "param_count": int(param_count) if param_count is not None else None,
                            "n_features": int(X_train_np.shape[1]),
                            "layers": model_cfg.get("layers"),
                            "dropout": model_cfg.get("dropout"),
                            "batchnorm": bool(model_cfg.get("batchnorm", True)),
                            "pos_label": int(pos_label_cfg),
                            "threshold_strategy": strategy,
                            "run_id": run_id,
                            # attach git and env pointers as lightweight metadata
                            **({f"env.{k}": v for k, v in (_collect_env_metadata().get("env", {}) or {}).items()}),
                            **({f"git.{k}": v for k, v in (_collect_env_metadata().get("git", {}) or {}).items()}),
                            "notes": notes_text if notes_text else None,
                        },
                    )
                    model_art.add_file(model_file.as_posix(), name=model_path.name)
                    wandb.log_artifact(model_art, aliases=[run_id, "latest"])
            except Exception:
                pass
        except Exception:
            pass

    # Persist basic W&B identifiers for downstream automation
    try:
        if wandb_enabled:
            wb_info = {"id": wb_id if 'wb_id' in locals() else None, "path": wb_path if 'wb_path' in locals() else None, "url": wb_url if 'wb_url' in locals() else None}
            with open((run_dir / "wandb.json"), "w", encoding="utf-8") as _wf:
                _json.dump(wb_info, _wf, indent=2)
    except Exception:
        pass

    # Clean up W&B run if open
    if wandb_enabled and wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass

    elapsed = time.time() - t0
    # Add a simple threshold sanity note for extreme operating points
    try:
        if cm.get("precision", 1.0) < 1e-3 or cm.get("recall", 1.0) < 1e-3:
            summary_lines.append("\n> Note: Precision/Recall is near 0 at the chosen threshold. Consider revising the threshold strategy or dataset balance.")
    except Exception:
        pass

    return {
        "model_path": (run_dir / model_path.name).as_posix() if not single_run_dir_mode else model_path.as_posix(),
        "metrics_path": (run_dir / "metrics.json").as_posix() if not single_run_dir_mode else (reports_dir / "metrics.json").as_posix(),
        "figures_path": (run_fig_dir / "learning_curves.png").as_posix(),
        "roc_curve_path": (run_fig_dir / "roc_curve.png").as_posix(),
        "pr_curve_path": (run_fig_dir / "pr_curve.png").as_posix(),
        "roc_auc": metrics.get("roc_auc"),
        "average_precision": metrics.get("average_precision"),
        "threshold": metrics.get("threshold"),
        "pos_label": pos_label_name,
        "elapsed_sec": elapsed,
        "n_train": int(len(y_train_np)),
        "n_test": int(len(y_test_np)),
        "n_features": int(X_train_np.shape[1]),
        "run_summary_path": (run_dir / "README.md").as_posix(),
        "run_dir": run_dir.as_posix(),
        "wandb_run_path": (wb_path if wandb_enabled else None),
        "wandb_run_url": (wb_url if wandb_enabled else None),
    }
