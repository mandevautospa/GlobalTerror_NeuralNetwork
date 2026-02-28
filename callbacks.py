"""callbacks.py – training callbacks for the GTD multi-task model.

Provides pure-Python / PyTorch callbacks that mirror the Keras callback API
but work with a standard PyTorch training loop.

Deliverables
------------
* ``EarlyStopping``     – stop training when the monitored metric stops improving.
* ``ReduceLROnPlateau`` – thin wrapper around ``torch.optim.lr_scheduler.ReduceLROnPlateau``.
* ``ModelCheckpoint``   – save the best model state-dict to disk.
* ``CallbackList``      – convenience wrapper to call multiple callbacks.
* ``build_callbacks``   – factory that builds the standard callback set from *config*.

Usage
-----
    from callbacks import build_callbacks

    cbs = build_callbacks(config, model, optimizer)

    # inside training loop:
    cbs.on_epoch_end(epoch, val_loss, model)
    if cbs.should_stop:
        break
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import torch
import torch.nn as nn


# ── EarlyStopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
    min_delta:
        Minimum change to qualify as an improvement.
    mode:
        ``"min"`` (lower is better) or ``"max"`` (higher is better).
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode

        self.best_score: float = math.inf if mode == "min" else -math.inf
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, metric: float) -> bool:
        """Update state.  Returns ``True`` when training should stop."""
        improved = (
            metric < self.best_score - self.min_delta
            if self.mode == "min"
            else metric > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = metric
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── ModelCheckpoint ───────────────────────────────────────────────────────────

class ModelCheckpoint:
    """Save the best model state-dict to disk.

    Parameters
    ----------
    filepath:
        Path where the ``.pth`` file is saved.
    mode:
        ``"min"`` or ``"max"``.
    """

    def __init__(self, filepath: str | Path, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.filepath   = Path(filepath)
        self.mode       = mode
        self.best_score = math.inf if mode == "min" else -math.inf

    def step(self, metric: float, model: nn.Module) -> bool:
        """Save model if *metric* improved.  Returns ``True`` when saved."""
        improved = (
            metric < self.best_score
            if self.mode == "min"
            else metric > self.best_score
        )
        if improved:
            self.best_score = metric
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.filepath)
            return True
        return False


# ── ReduceLROnPlateau wrapper ─────────────────────────────────────────────────

class ReduceLROnPlateau:
    """Thin wrapper around ``torch.optim.lr_scheduler.ReduceLROnPlateau``.

    Parameters
    ----------
    optimizer:
        The optimizer whose LR is adjusted.
    patience:
        Number of epochs with no improvement before reducing LR.
    factor:
        Factor by which the LR is reduced.
    mode:
        ``"min"`` or ``"max"``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 3,
        factor: float = 0.5,
        mode: str = "min",
    ) -> None:
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor,
        )

    def step(self, metric: float) -> None:
        """Pass *metric* to the underlying scheduler."""
        self._scheduler.step(metric)

    @property
    def last_lr(self) -> float:
        """Current learning rate of the first param group."""
        return self._scheduler.optimizer.param_groups[0]["lr"]


# ── CallbackList ──────────────────────────────────────────────────────────────

class CallbackList:
    """Aggregate multiple callbacks into a single object.

    Attributes
    ----------
    should_stop : bool
        Set to ``True`` by ``EarlyStopping`` when training should halt.
    """

    def __init__(
        self,
        early_stopping: EarlyStopping | None = None,
        reduce_lr: ReduceLROnPlateau | None = None,
        checkpoint: ModelCheckpoint | None = None,
    ) -> None:
        self.early_stopping = early_stopping
        self.reduce_lr      = reduce_lr
        self.checkpoint     = checkpoint
        self.should_stop    = False

    def on_epoch_end(self, epoch: int, metric: float, model: nn.Module) -> None:
        """Call all callbacks at the end of an epoch.

        Parameters
        ----------
        epoch:  Current epoch number (1-based, for logging).
        metric: The monitored scalar (e.g. val_loss).
        model:  The model being trained (needed for checkpoint).
        """
        if self.reduce_lr is not None:
            self.reduce_lr.step(metric)
            lr = self.reduce_lr.last_lr
            print(f"  [ReduceLR]  current lr = {lr:.2e}")

        if self.checkpoint is not None:
            saved = self.checkpoint.step(metric, model)
            if saved:
                print(f"  [Checkpoint] best model saved (score={metric:.6f})")

        if self.early_stopping is not None:
            stop = self.early_stopping.step(metric)
            if stop:
                self.should_stop = True
                print(
                    f"  [EarlyStopping] triggered after epoch {epoch} "
                    f"(patience={self.early_stopping.patience}, "
                    f"best={self.early_stopping.best_score:.6f})"
                )


# ── Factory ───────────────────────────────────────────────────────────────────

def build_callbacks(
    config: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> CallbackList:
    """Build a standard ``CallbackList`` from *config*.

    Parameters
    ----------
    config:     Full CONFIG dict.
    model:      The model (needed to wire up ModelCheckpoint).
    optimizer:  The optimizer (needed to wire up ReduceLR).

    Returns
    -------
    ``CallbackList`` ready to use in the training loop.
    """
    cb_cfg = config.get("callbacks", {})

    es = EarlyStopping(
        patience=cb_cfg.get("early_stopping_patience", 5),
        mode="min",
    )
    rlr = ReduceLROnPlateau(
        optimizer,
        patience=cb_cfg.get("reduce_lr_patience", 3),
        factor=cb_cfg.get("reduce_lr_factor", 0.5),
        mode="min",
    )
    ckpt_path = cb_cfg.get("model_checkpoint_path", "outputs/best_model.pth")
    ckpt = ModelCheckpoint(filepath=ckpt_path, mode="min")

    return CallbackList(early_stopping=es, reduce_lr=rlr, checkpoint=ckpt)
