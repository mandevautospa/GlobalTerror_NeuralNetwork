"""reporter.py – core implementation of save_run_artifacts."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for CI / Colab
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import seaborn as sns
    _SNS_AVAILABLE = True
except ImportError:
    _SNS_AVAILABLE = False


# ── Public API ────────────────────────────────────────────────────────────────

def save_run_artifacts(
    config: dict,
    history: dict,
    confusion_matrices: dict | None = None,
    run_tag: str | None = None,
) -> Path:
    """Persist all artefacts for a single training run.

    Parameters
    ----------
    config:
        Full CONFIG dict used for the run.
    history:
        Dict of metric lists, one entry per epoch.  E.g.::

            {
                "train_loss": [...],
                "val_loss":   [...],
                "val_att_acc": [...],
                ...
            }
    confusion_matrices:
        Optional dict mapping head names to (cm_array, label_names) tuples::

            {
                "attacktype1": (np.ndarray, ["Bombing", "Shooting", ...]),
                "gname":       (np.ndarray, ["Taliban", ...]),
            }
    run_tag:
        Optional human-readable tag appended to the run-folder name.
        Defaults to an ISO timestamp.

    Returns
    -------
    ``pathlib.Path`` pointing to the run folder.
    """
    # Determine run directory
    output_root = Path(config.get("reporting", {}).get("output_dir", "outputs"))
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    run_dir = output_root / f"run_{ts}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save config
    _save_json(config, run_dir / "config.json")

    # 2. Save git info
    git_info = _get_git_info()
    _save_json(git_info, run_dir / "git_info.json")

    # 3. Save metrics CSV
    _save_metrics_csv(history, run_dir / "metrics.csv")

    # 4. Save learning-curve plots
    if _MPL_AVAILABLE and history:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        _save_learning_curves(history, plots_dir)

    # 5. Save confusion matrices
    if confusion_matrices and _MPL_AVAILABLE:
        cm_dir = run_dir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        for head_name, (cm_array, label_names) in confusion_matrices.items():
            _save_confusion_matrix(cm_array, label_names, head_name, cm_dir)

    print(f"[reporting] Artefacts saved to: {run_dir}")
    return run_dir


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_json(obj: dict, path: Path) -> None:
    """Serialise *obj* to *path*, converting non-serialisable values to strings."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def _get_git_info() -> dict:
    """Return the current git commit hash and dirty status."""
    info = {"commit": "unknown", "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["commit"] = commit

        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["dirty"] = len(status) > 0
    except Exception:
        pass
    return info


def _save_metrics_csv(history: dict, path: Path) -> None:
    """Write the training history to a CSV file."""
    if not history:
        return
    try:
        df = pd.DataFrame(history)
        df.index.name = "epoch"
        df.to_csv(path)
    except Exception as exc:
        print(f"[reporting] Could not save metrics CSV: {exc}")


def _save_learning_curves(history: dict, plots_dir: Path) -> None:
    """Save one PNG per pair of train/val metrics."""
    if not history:
        return
    first_series = next(
        (v for v in history.values() if v), None
    )
    if first_series is None:
        return
    epochs = range(1, len(first_series) + 1)

    # Helper: plot a single curve pair
    def _plot_pair(train_key, val_key, title, ylabel, filename):
        if train_key not in history and val_key not in history:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        if train_key in history:
            ax.plot(epochs, history[train_key], label="Train")
        if val_key in history:
            ax.plot(epochs, history[val_key], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=100)
        plt.close(fig)

    _plot_pair("train_loss",     "val_loss",     "Total Loss",       "Loss",     "total_loss.png")
    _plot_pair("train_att_loss", "val_att_loss", "Attack-Type Loss", "Loss",     "att_loss.png")
    _plot_pair("train_suc_loss", "val_suc_loss", "Success Loss",     "Loss",     "suc_loss.png")
    _plot_pair("train_grp_loss", "val_grp_loss", "Group Loss",       "Loss",     "grp_loss.png")
    _plot_pair("train_cas_loss", "val_cas_loss", "Casualties Loss",  "Loss",     "cas_loss.png")

    # Accuracy curves (val only)
    acc_keys = [k for k in history if "acc" in k and k.startswith("val_")]
    if acc_keys:
        fig, ax = plt.subplots(figsize=(7, 4))
        for key in acc_keys:
            ax.plot(epochs, history[key], label=key.replace("val_", ""))
        ax.set_title("Validation Accuracy (Classification Heads)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(plots_dir / "val_accuracy.png", dpi=100)
        plt.close(fig)

    # Regression metrics
    for key in ("val_cas_mse", "val_cas_mae", "val_cas_rmse_orig",
                "val_cas_mae_orig"):
        if key in history:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(epochs, history[key], label=key)
            ax.set_title(f"Regression Metric: {key}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Error")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{key}.png", dpi=100)
            plt.close(fig)


def _save_confusion_matrix(
    cm: np.ndarray,
    label_names: list,
    head_name: str,
    cm_dir: Path,
) -> None:
    """Save a confusion matrix as a PNG heatmap."""
    n = len(label_names)
    fig_size = max(6, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    if _SNS_AVAILABLE:
        sns.heatmap(
            cm, annot=(n <= 20), fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names, ax=ax,
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticklabels(label_names)

    ax.set_title(f"Confusion Matrix – {head_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    safe_name = head_name.replace(" ", "_").replace("/", "_")
    fig.savefig(cm_dir / f"cm_{safe_name}.png", dpi=100)
    plt.close(fig)
