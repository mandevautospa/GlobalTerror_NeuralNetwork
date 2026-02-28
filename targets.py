"""targets.py – casualty target transforms for the GTD regression head.

Transforms the raw ``nkill`` / ``nwound`` values to log-scale before
training and inverts them for evaluation on the original scale.

Deliverables
------------
* ``to_log1p``            – apply log1p to a NumPy array or Tensor.
* ``from_log1p``          – inverse (expm1).
* ``casualty_sample_weights`` – per-sample weights that upweight high-casualty
                                 examples.

Usage
-----
    from targets import to_log1p, from_log1p, casualty_sample_weights

    y_cas_log = to_log1p(df_train[["nkill", "nwound"]].values)
    y_cas_raw = from_log1p(predicted_log_values)
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Transform helpers ─────────────────────────────────────────────────────────

def to_log1p(x):
    """Apply ``log1p`` to *x* element-wise.

    Works with NumPy arrays, plain Python scalars, and (if torch is
    installed) PyTorch tensors.

    Parameters
    ----------
    x : array-like | torch.Tensor
        Raw casualty values (nkill, nwound, or their sum).  Non-negative.

    Returns
    -------
    Same type as input, transformed via log1p.
    """
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.log1p(x.float())
    arr = np.asarray(x, dtype=float)
    return np.log1p(arr)


def from_log1p(x):
    """Invert ``to_log1p`` via ``expm1``.

    Parameters
    ----------
    x : array-like | torch.Tensor
        Log-scale predictions.

    Returns
    -------
    Same type as input, back-transformed to the original (counts) scale.
    """
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return torch.expm1(x.float())
    arr = np.asarray(x, dtype=float)
    return np.expm1(arr)


# ── Sample weights ────────────────────────────────────────────────────────────

def casualty_sample_weights(
    y_raw: np.ndarray,
    n_bins: int = 5,
    max_weight: float = 10.0,
) -> np.ndarray:
    """Compute per-sample weights that upweight high-casualty events.

    Bins the total casualty count (nkill + nwound or scalar) into *n_bins*
    quantile bins and assigns inverse-frequency weights so that rare
    high-casualty events contribute more to the loss.

    Parameters
    ----------
    y_raw:
        2-D array of shape (N, 2) with columns [nkill, nwound], **or**
        1-D array of shape (N,) with total casualties.  Non-negative values.
    n_bins:
        Number of quantile bins (default 5 → quintiles).
    max_weight:
        Cap on the maximum per-sample weight to avoid extreme values.

    Returns
    -------
    weights : np.ndarray of shape (N,), dtype float32.
    """
    y = np.asarray(y_raw, dtype=float)
    if y.ndim == 2:
        total = y.sum(axis=1)
    else:
        total = y.ravel()

    # Assign each sample to a quantile bin
    bin_edges    = np.quantile(total, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9  # ensure max value falls in the last bin
    bin_indices  = np.digitize(total, bin_edges[1:])  # 0 … n_bins-1

    counts  = np.bincount(bin_indices, minlength=n_bins).astype(float)
    counts  = np.where(counts > 0, counts, 1.0)
    inv_freq = total.shape[0] / (n_bins * counts)
    weights  = inv_freq[bin_indices]
    weights  = np.clip(weights, 1.0, max_weight)
    return weights.astype(np.float32)
