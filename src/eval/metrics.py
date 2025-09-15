from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


def compute_metrics(y_true, y_prob) -> Dict:
    y_pred = (np.asarray(y_prob) > 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    auc = float(roc_auc_score(y_true, y_prob))
    return {"classification_report": report, "roc_auc": auc}


def save_metrics(metrics: Dict, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_learning_curves(history, out_path: str | Path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    if "loss" in history.history:
        plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

