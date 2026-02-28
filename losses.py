"""losses.py – custom loss functions for the GTD multi-task model.

Deliverables
------------
* ``FocalLoss``        – focal cross-entropy (recommended for gname head).
* ``class_weighted_ce`` – CrossEntropyLoss with inverse-frequency weights.
* ``build_gname_loss`` – factory that returns the right loss per config.
* ``build_casualty_loss`` – factory for Huber vs. MSE casualty loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Focal loss down-weights well-classified examples so the model focuses on
    hard, misclassified examples.  Especially effective for severely imbalanced
    class distributions such as the ``gname`` head.

    Parameters
    ----------
    gamma:
        Focusing parameter (γ ≥ 0).  γ = 0 reduces to cross-entropy.
        γ = 2 is the default recommended by the original paper.
    alpha:
        Optional per-class weight tensor of shape ``(num_classes,)``.
        ``None`` → no per-class weighting.
    reduction:
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits:  (N, C) raw (un-softmaxed) class scores.
        targets: (N,)  ground-truth class indices.
        """
        log_prob = F.log_softmax(logits, dim=1)           # (N, C)
        prob     = log_prob.exp()                          # (N, C)

        # Gather the log-prob and prob for the true class
        log_p_t = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        p_t     = prob.gather(1, targets.unsqueeze(1)).squeeze(1)      # (N,)

        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * log_p_t                    # (N,)

        # Optional per-class α weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, reduction={self.reduction!r}"


# ── Class-weighted cross-entropy ──────────────────────────────────────────────

def class_weighted_ce(
    class_counts: np.ndarray | list,
    device: torch.device | str = "cpu",
) -> nn.CrossEntropyLoss:
    """Return a CrossEntropyLoss with inverse-frequency class weights.

    Weight for class *c* = total_samples / (num_classes * count_c).

    Parameters
    ----------
    class_counts:
        1-D array with the number of training samples per class.
    device:
        Device on which the weight tensor is placed.

    Returns
    -------
    nn.CrossEntropyLoss with ``weight`` set.
    """
    counts  = np.asarray(class_counts, dtype=float)
    total   = counts.sum()
    n_cls   = len(counts)
    # Replace zero counts with 1 to avoid division by zero; those classes have
    # no training examples so their weight doesn't affect the loss in practice.
    _MIN_COUNT = 1.0
    weights = total / (n_cls * np.where(counts > 0, counts, _MIN_COUNT))
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


# ── Loss factories ────────────────────────────────────────────────────────────

def build_gname_loss(
    config: dict,
    train_gname_labels: np.ndarray | None = None,
    num_groups: int | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Return the loss module for the gname head based on *config*.

    Parameters
    ----------
    config:
        Full CONFIG dict.  Only ``config["losses"]`` is used.
    train_gname_labels:
        Encoded group labels from the training set (required for
        ``weighted_ce`` to compute class counts).
    num_groups:
        Total number of group classes (required for ``weighted_ce`` when
        *train_gname_labels* may not cover all classes).
    device:
        Compute device.
    """
    loss_type = config["losses"].get("gname_loss", "focal")

    if loss_type == "focal":
        gamma = config["losses"].get("focal_gamma", 2.0)
        alpha_cfg = config["losses"].get("focal_alpha", None)
        alpha = None
        if alpha_cfg is not None:
            alpha = torch.tensor(alpha_cfg, dtype=torch.float32, device=device)
        return FocalLoss(gamma=gamma, alpha=alpha)

    if loss_type == "weighted_ce":
        if train_gname_labels is None or num_groups is None:
            raise ValueError(
                "train_gname_labels and num_groups are required for 'weighted_ce'."
            )
        labels = np.asarray(train_gname_labels)
        counts = np.bincount(labels.astype(int), minlength=num_groups)
        return class_weighted_ce(counts, device=device)

    raise ValueError(
        f"Unknown gname_loss type '{loss_type}'. Choose 'focal' or 'weighted_ce'."
    )


def build_casualty_loss(config: dict) -> nn.Module:
    """Return the loss module for the casualties regression head.

    Parameters
    ----------
    config: Full CONFIG dict.  Only ``config["losses"]`` is used.
    """
    loss_type = config["losses"].get("casualties_loss", "huber")

    if loss_type == "huber":
        delta = config["losses"].get("huber_delta", 1.0)
        return nn.HuberLoss(delta=delta)

    if loss_type == "mse":
        return nn.MSELoss()

    raise ValueError(
        f"Unknown casualties_loss type '{loss_type}'. Choose 'huber' or 'mse'."
    )
