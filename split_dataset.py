"""split_dataset.py – time-based (and optionally geo-based) train/val/test split.

Deliverables
------------
* ``make_splits(df, config)`` – returns (df_train, df_val, df_test).
* Overlap assertion using a composite incident key.
* ``make_incident_key(df, cols)`` – helper for building the key Series.

Usage
-----
    from split_dataset import make_splits
    from config import CONFIG

    df_train, df_val, df_test = make_splits(df_raw, CONFIG)
"""

import warnings
from typing import Tuple

import pandas as pd


# ── Public helpers ────────────────────────────────────────────────────────────

def make_incident_key(df: pd.DataFrame, cols: list) -> pd.Series:
    """Build a composite incident key from *cols* that are present in *df*.

    Only columns that actually exist in *df* are used so the helper is robust
    when applied to subsets that lack some raw columns.
    """
    available = [c for c in cols if c in df.columns]
    return df[available].astype(str).agg("||".join, axis=1)


def make_splits(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val / test according to *config["split"]*.

    Strategy
    --------
    temporal (default)
        Train  : iyear <= split_year_train_val
        Val    : split_year_train_val < iyear <= split_year_val_test
        Test   : iyear > split_year_val_test

    Optional geo split
        When ``config["split"]["geo_split"]`` is True the rows whose
        ``region`` code appears in ``held_out_regions`` are removed from the
        train fold and appended to the test fold.

    After splitting, a duplicate-event check is run; a warning is emitted
    when the collision rate exceeds ``collision_warn_threshold``.

    Parameters
    ----------
    df:
        Pre-processed DataFrame (label-encoded columns still acceptable).
    config:
        Full CONFIG dict (only ``config["split"]`` is used).

    Returns
    -------
    df_train, df_val, df_test
    """
    cfg = config["split"]
    strategy = cfg.get("strategy", "temporal")

    if strategy != "temporal":
        raise NotImplementedError(
            f"Split strategy '{strategy}' is not implemented. "
            "Only 'temporal' is currently supported."
        )

    if "iyear" not in df.columns:
        raise ValueError("DataFrame must contain an 'iyear' column for temporal split.")

    train_val_year = cfg["split_year_train_val"]
    val_test_year  = cfg["split_year_val_test"]

    df_train = df[df["iyear"] <= train_val_year].copy()
    df_val   = df[(df["iyear"] > train_val_year) & (df["iyear"] <= val_test_year)].copy()
    df_test  = df[df["iyear"] > val_test_year].copy()

    # Optional geo split: hold-out regions go to test
    if cfg.get("geo_split", False) and cfg.get("held_out_regions"):
        held_out = set(cfg["held_out_regions"])
        if "region" in df.columns:
            mask_held = df_train["region"].isin(held_out)
            if mask_held.any():
                df_test  = pd.concat([df_test, df_train[mask_held]], ignore_index=True)
                df_train = df_train[~mask_held].copy()

    # ── Overlap / leakage check ───────────────────────────────────────────────
    key_cols  = cfg.get("incident_key_cols", ["iyear", "imonth", "iday"])
    threshold = cfg.get("collision_warn_threshold", 0.001)

    _check_no_overlap(df_train, df_val,  key_cols, "train", "val",  threshold)
    _check_no_overlap(df_train, df_test, key_cols, "train", "test", threshold)
    _check_no_overlap(df_val,   df_test, key_cols, "val",   "test", threshold)

    print(
        f"Split complete – "
        f"train={len(df_train):,}  val={len(df_val):,}  test={len(df_test):,}"
    )
    return df_train, df_val, df_test


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_no_overlap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_cols: list,
    name_a: str,
    name_b: str,
    threshold: float,
) -> None:
    """Emit a warning if the two splits share incident keys above *threshold*.

    For a purely temporal split the overlap should be exactly zero.  Any
    overlap indicates a data-leakage bug.
    """
    if df_a.empty or df_b.empty:
        return

    keys_a = set(make_incident_key(df_a, key_cols))
    keys_b = set(make_incident_key(df_b, key_cols))
    overlap = keys_a & keys_b

    if not overlap:
        return

    # Collision rate is relative to the smaller split because even a single
    # duplicate key in a tiny split fully leaks that event's label signal.
    # The intent is to detect any cross-split overlap; the threshold guards
    # against noisy key collisions from near-duplicate raw events.
    rate = len(overlap) / min(len(keys_a), len(keys_b))
    msg  = (
        f"Data-leakage warning: {len(overlap)} duplicate incident keys found "
        f"across '{name_a}' and '{name_b}' splits "
        f"(collision rate = {rate:.4%})."
    )
    if rate > threshold:
        warnings.warn(msg, UserWarning, stacklevel=3)
    else:
        print(msg)
