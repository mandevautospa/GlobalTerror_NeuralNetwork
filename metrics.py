"""metrics.py – evaluation metrics for the GTD multi-task model.

Provides macro / weighted F1, top-k accuracy, and per-head report printing.
All functions operate on plain NumPy arrays or PyTorch tensors so they can be
called during or after training without a Keras dependency.

Deliverables
------------
* ``macro_f1``
* ``weighted_f1``
* ``top_k_accuracy``
* ``print_head_report``
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    """Convert a PyTorch tensor or array-like to a NumPy array."""
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ── F1 metrics ────────────────────────────────────────────────────────────────

def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """Return per-class precision, recall, and F1 as arrays of length *num_classes*."""
    precision = np.zeros(num_classes, dtype=float)
    recall    = np.zeros(num_classes, dtype=float)
    f1        = np.zeros(num_classes, dtype=float)

    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[c] = p
        recall[c]    = r
        f1[c]        = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return precision, recall, f1


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    """Macro-averaged F1: unweighted mean of per-class F1 scores.

    Parameters
    ----------
    y_true: array-like (N,) – ground-truth class indices.
    y_pred: array-like (N,) – predicted class indices.
    num_classes: int – total number of classes.

    Returns
    -------
    float in [0, 1].
    """
    y_true = _to_numpy(y_true).ravel()
    y_pred = _to_numpy(y_pred).ravel()
    _, _, f1 = _per_class_f1(y_true, y_pred, num_classes)
    return float(np.mean(f1))


def weighted_f1(y_true, y_pred, num_classes: int) -> float:
    """Weighted-averaged F1: F1 scores weighted by support (true class counts).

    Parameters
    ----------
    y_true: array-like (N,) – ground-truth class indices.
    y_pred: array-like (N,) – predicted class indices.
    num_classes: int – total number of classes.

    Returns
    -------
    float in [0, 1].
    """
    y_true = _to_numpy(y_true).ravel()
    y_pred = _to_numpy(y_pred).ravel()
    _, _, f1 = _per_class_f1(y_true, y_pred, num_classes)
    support = np.array([np.sum(y_true == c) for c in range(num_classes)], dtype=float)
    total   = support.sum()
    return float(np.dot(f1, support) / total) if total > 0 else 0.0


# ── Top-k accuracy ────────────────────────────────────────────────────────────

def top_k_accuracy(y_true, y_logits, k: int) -> float:
    """Compute top-k accuracy for a single classification head.

    Parameters
    ----------
    y_true:   array-like (N,) – ground-truth class indices.
    y_logits: array-like (N, C) – raw logits or class probabilities.
    k:        int – number of top predictions to consider.

    Returns
    -------
    float in [0, 1].
    """
    y_true   = _to_numpy(y_true).ravel().astype(int)
    y_logits = _to_numpy(y_logits)

    if y_logits.ndim != 2:
        raise ValueError(
            f"y_logits must be 2-D (N, C), got shape {y_logits.shape}"
        )

    n_classes = y_logits.shape[1]
    k = min(k, n_classes)

    # Indices of the top-k predicted classes (unsorted within top-k is fine).
    # argpartition is O(N * C) rather than O(N * C * log C) for large class counts.
    top_k_preds = np.argpartition(-y_logits, k, axis=1)[:, :k]  # (N, k)
    correct = np.any(top_k_preds == y_true[:, None], axis=1)
    return float(correct.mean())


# ── Per-head report ───────────────────────────────────────────────────────────

def print_head_report(
    head_name: str,
    y_true,
    y_logits,
    label_names: list | None = None,
    top_k_list: list[int] | None = None,
) -> dict:
    """Print a clean evaluation report for one classification head.

    Reports
    -------
    * Accuracy
    * Macro F1
    * Weighted F1
    * Top-k accuracy for each k in *top_k_list*

    Parameters
    ----------
    head_name:   Human-readable head name, e.g. ``"attacktype1"``.
    y_true:      Ground-truth class indices (N,).
    y_logits:    Raw logits (N, C).
    label_names: Optional list of class name strings (length C).
    top_k_list:  List of k values to evaluate, e.g. ``[3, 5]``.

    Returns
    -------
    dict with all computed metric values.
    """
    if top_k_list is None:
        top_k_list = [3, 5]

    y_true_np   = _to_numpy(y_true).ravel().astype(int)
    y_logits_np = _to_numpy(y_logits)
    y_pred_np   = y_logits_np.argmax(axis=1)
    num_classes = y_logits_np.shape[1]

    acc       = float(np.mean(y_pred_np == y_true_np))
    m_f1      = macro_f1(y_true_np, y_pred_np, num_classes)
    w_f1      = weighted_f1(y_true_np, y_pred_np, num_classes)
    topk_vals = {k: top_k_accuracy(y_true_np, y_logits_np, k) for k in top_k_list}

    width = 50
    print("=" * width)
    print(f"  Head: {head_name}")
    print("=" * width)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {m_f1:.4f}")
    print(f"  Weighted F1 : {w_f1:.4f}")
    for k, v in topk_vals.items():
        print(f"  Top-{k} Acc   : {v:.4f}")
    print("-" * width)

    results = {
        "accuracy":     acc,
        "macro_f1":     m_f1,
        "weighted_f1":  w_f1,
    }
    for k, v in topk_vals.items():
        results[f"top_{k}_acc"] = v
    return results
