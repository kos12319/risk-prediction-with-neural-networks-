from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.models.torch_factory import build_torch_model
from src.training.history import SimpleHistory
from src.utils.seed import make_torch_generator, make_worker_init_fn

if TYPE_CHECKING:  # pragma: no cover
    import torch


logger = logging.getLogger(__name__)


def resolve_torch_device(force_cpu: bool) -> Tuple["torch.device", Dict[str, Any]]:
    """Best-effort accelerator selection with CPU fallback."""
    import torch

    meta: Dict[str, Any] = {"forced_cpu": bool(force_cpu)}

    if force_cpu:
        meta["selected"] = "cpu"
        meta["reason"] = "FORCE_CPU env toggle"
        return torch.device("cpu"), meta

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        meta.update({"selected": "cuda", "cuda_index": int(idx)})
        try:
            meta["cuda_name"] = torch.cuda.get_device_name(idx)
        except Exception:
            pass
        return torch.device("cuda"), meta

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        try:
            if mps_backend.is_available():
                meta["selected"] = "mps"
                try:
                    meta["mps_is_built"] = bool(mps_backend.is_built())
                except Exception:
                    pass
                ane_fn = getattr(mps_backend, "is_neural_engine_available", None)
                if callable(ane_fn):
                    try:
                        meta["mps_neural_engine"] = bool(ane_fn())
                    except Exception:
                        pass
                return torch.device("mps"), meta
        except Exception:
            pass

    xpu_backend = getattr(torch, "xpu", None)
    if xpu_backend is not None:
        try:
            if callable(getattr(xpu_backend, "is_available", None)) and xpu_backend.is_available():
                meta["selected"] = "xpu"
                return torch.device("xpu"), meta
        except Exception:
            pass

    meta["selected"] = "cpu"
    meta.setdefault("reason", "no accelerator detected")
    return torch.device("cpu"), meta


def attach_wandb_hooks(model_cfg: Dict[str, Any], wandb_run: Any) -> None:
    """Attach lightweight W&B logging callbacks to ``train_pytorch``."""
    try:
        import wandb  # type: ignore
    except Exception:
        return

    epochs = int(model_cfg.get("epochs", 30))

    def _wb_on_epoch(epoch: int, loss: float, val_loss: Optional[float] = None, **kwargs: Any) -> None:
        data: Dict[str, Any] = {"epoch": int(epoch), "loss": float(loss)}
        if val_loss is not None and not (isinstance(val_loss, float) and np.isnan(val_loss)):
            data["val_loss"] = float(val_loss)
        for key, value in (kwargs or {}).items():
            if value is None:
                continue
            try:
                data[key] = float(value) if isinstance(value, (int, float)) else value
            except Exception:
                pass
        wandb.log(data, step=int(epoch))

    def _wb_watch_model(model: Any) -> None:
        try:
            wandb.watch(model, log="gradients", log_freq=max(1, epochs // 10))
        except Exception:
            pass

    train_pytorch.on_epoch = _wb_on_epoch  # type: ignore[attr-defined]
    train_pytorch.on_watch_model = _wb_watch_model  # type: ignore[attr-defined]


def clear_wandb_hooks() -> None:
    """Remove any previously attached W&B callbacks."""
    for attr in ("on_epoch", "on_watch_model"):
        if hasattr(train_pytorch, attr):
            delattr(train_pytorch, attr)


def train_pytorch(
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
) -> Tuple[Dict[str, Any], SimpleHistory]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split as torch_random_split

    from src.models.torch_nn import focal_binary_loss as torch_focal_loss

    # Device selection with optional CPU override (FORCE_CPU=1)
    force_cpu = str(os.environ.get("FORCE_CPU", "")).lower() in {"1", "true", "yes"}
    device, device_meta = resolve_torch_device(force_cpu)
    device_meta.setdefault("repr", str(device))
    logger.info(
        "Starting PyTorch training on device=%s with train=%s, val=%s, test=%s",
        device,
        X_train_np.shape,
        None if X_val_np is None else X_val_np.shape,
        X_test_np.shape,
    )
    if device.type == "cpu" and force_cpu:
        try:
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception:
            pass

    model = build_torch_model(input_dim=X_train_np.shape[1], model_cfg=model_cfg).to(device)

    on_watch = getattr(train_pytorch, "on_watch_model", None)
    if callable(on_watch):
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

    if X_val_np is not None and y_val_np is not None:
        val_tensor_x = torch.tensor(X_val_np, dtype=torch.float32)
        val_tensor_y = torch.tensor(y_val_np, dtype=torch.float32)
        val_ds = TensorDataset(val_tensor_x, val_tensor_y)
        train_ds = ds
    else:
        n_total = len(ds)
        n_val = int(max(1, round(n_total * val_split)))
        n_train = n_total - n_val
        generator = make_torch_generator(random_state)
        train_ds, val_ds = torch_random_split(ds, [n_train, n_val], generator=generator) if n_val > 0 else (ds, None)

    generator = make_torch_generator(random_state)
    worker_fn = make_worker_init_fn(random_state)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, worker_init_fn=worker_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, worker_init_fn=worker_fn) if val_ds is not None else None

    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    wait = 0
    tr_losses: List[float] = []
    va_losses: List[float] = []
    epoch_stats: List[Dict[str, Any]] = []

    class_weight_cfg = model_cfg.get("_class_weight")
    use_weighted_bce = class_weight_cfg is not None and loss_name != "focal"
    if use_weighted_bce:
        w0 = float(class_weight_cfg.get(0, 1.0))
        w1 = float(class_weight_cfg.get(1, 1.0))

    on_epoch_cb = getattr(train_pytorch, "on_epoch", None)

    for epoch in range(epochs):
        ep_start = time.time()
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yb = yb.view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if use_weighted_bce:
                loss_per = criterion(logits, yb)
                sample_weight = yb * w1 + (1.0 - yb) * w0
                loss = (loss_per * sample_weight).mean()
            else:
                loss = criterion(logits, yb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        tr_losses.append(epoch_loss)

        val_loss = None
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
                        sample_weight = yb * w1 + (1.0 - yb) * w0
                        loss = (loss_per * sample_weight).mean()
                    else:
                        loss = criterion(logits, yb).mean()
                    val_loss += loss.item() * xb.size(0)
                    val_targets.append(yb.detach().cpu())
                    val_logits_all.append(logits.detach().cpu())
            val_loss /= len(val_loader.dataset)
            va_losses.append(val_loss)
            try:
                from sklearn.metrics import roc_auc_score

                logits_np = torch.cat(val_logits_all, dim=0).numpy().reshape(-1)
                targets_np = torch.cat(val_targets, dim=0).numpy().reshape(-1)
                probs = 1.0 / (1.0 + np.exp(-logits_np))
                if int(pos_label_for_auc) == 0:
                    y_true_auc = (1 - targets_np).astype(int)
                    y_prob_auc = 1.0 - probs
                else:
                    y_true_auc = targets_np.astype(int)
                    y_prob_auc = probs
                val_auc = float(roc_auc_score(y_true_auc, y_prob_auc))
            except Exception:
                val_auc = None

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if patience and wait >= patience:
                    break

        else:
            va_losses.append(np.nan)

        epoch_stats.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "lr": optimizer.param_groups[0].get("lr"),
                "time_sec": float(time.time() - ep_start),
            }
        )
        logger.debug(
            "Epoch %d complete | loss=%.4f | val_loss=%s | val_auc=%s | time=%.2fs",
            epoch + 1,
            epoch_loss,
            "{:.4f}".format(val_loss) if val_loss is not None else "n/a",
            "{:.4f}".format(val_auc) if val_auc is not None else "n/a",
            epoch_stats[-1]["time_sec"],
        )

        if callable(on_epoch_cb):
            try:
                on_epoch_cb(epoch + 1, epoch_loss, val_loss, val_auc=val_auc, lr=optimizer.param_groups[0].get("lr"), epoch_time_sec=epoch_stats[-1]["time_sec"])
            except Exception:
                pass

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_np, dtype=torch.float32, device=device))
        logits = logits.cpu().numpy().reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits)) if use_logits else logits

    y_prob_val = None
    if X_val_np is not None:
        with torch.no_grad():
            logits_val = model(torch.tensor(X_val_np, dtype=torch.float32, device=device))
            logits_val = logits_val.cpu().numpy().reshape(-1)
            y_prob_val = 1.0 / (1.0 + np.exp(-logits_val)) if use_logits else logits_val

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_model_path)

    history = SimpleHistory(tr_losses, va_losses)
    result: Dict[str, Any] = {
        "y_prob": probs,
        "y_prob_val": y_prob_val,
        "param_count": n_params,
        "device": str(device),
        "device_info": device_meta,
        "epochs_ran": len(tr_losses),
        "epoch_stats": epoch_stats,
        "model_path": out_model_path.as_posix(),
        "y_prob_label": 1,
    }
    try:
        best_val_loss = float(np.nanmin(va_losses)) if va_losses else float("nan")
    except ValueError:
        best_val_loss = float("nan")
    logger.info(
        "Finished PyTorch training | epochs=%d | best_val=%s | model_path=%s",
        len(tr_losses),
        "{:.4f}".format(best_val_loss) if not np.isnan(best_val_loss) else "n/a",
        out_model_path,
    )
    return result, history
