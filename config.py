"""Central configuration for the GTD multi-task neural network.

All dataset paths, split strategies, model hyperparameters, loss weights,
and enabled metrics live here.  Import this dict in every module instead of
hard-coding values.
"""

import os

CONFIG: dict = {
    # ── Dataset ──────────────────────────────────────────────────────────────
    "dataset": {
        # Set to an explicit CSV path to skip kagglehub download.
        "csv_path": os.environ.get("GTD_CSV_PATH", None),
        "encoding": "latin1",
    },

    # ── Split strategy ───────────────────────────────────────────────────────
    "split": {
        # "temporal" splits by iyear; "random" uses sklearn train_test_split.
        "strategy": "temporal",
        # Train  ≤ split_year_train_val, Val > split_year_train_val and ≤ split_year_val_test
        # Test   > split_year_val_test
        "split_year_train_val": 2013,
        "split_year_val_test": 2016,
        # Optional: hold out specific region codes from the test fold.
        "geo_split": False,
        "held_out_regions": [],
        # Incident key fields used to detect cross-split duplicates.
        "incident_key_cols": ["iyear", "imonth", "iday",
                              "country_txt", "city",
                              "attacktype1", "nkill", "nwound"],
        # Warn when duplicate-event collision rate exceeds this fraction.
        "collision_warn_threshold": 0.001,
    },

    # ── Model architecture ───────────────────────────────────────────────────
    "model": {
        "hidden_dim": 256,
        "dropout": 0.3,
        # Per-head tower: number of extra dense layers before each output head.
        "tower_layers": 2,
        "tower_hidden": 128,
    },

    # ── Training ─────────────────────────────────────────────────────────────
    "training": {
        "batch_size": 64,
        "lr": 1e-3,
        "num_epochs": 30,
        # Random seed applied to Python random, NumPy, and PyTorch.
        "seed": 42,
        # L2 regularisation (weight_decay in Adam).
        "l2_weight_decay": 1e-4,
    },

    # ── Loss weights ─────────────────────────────────────────────────────────
    # loss_total = w_attack*L_attack + w_success*L_success + ...
    # Starting point from ablation studies; tune as needed.
    "loss_weights": {
        "attack":     1.0,
        "success":    1.0,
        "group":      2.0,   # upweighted to combat gname gravity well
        "casualties": 0.5,
    },

    # ── Loss functions ───────────────────────────────────────────────────────
    "losses": {
        # "focal" or "weighted_ce" for the gname head
        "gname_loss": "focal",
        "focal_gamma": 2.0,
        # None  → uniform α; list → per-class α weights (length = num_groups)
        "focal_alpha": None,
        # "huber" or "mse" for the casualties regression head
        "casualties_loss": "huber",
        "huber_delta": 1.0,
    },

    # ── Metrics ──────────────────────────────────────────────────────────────
    "metrics": {
        # Top-k values computed for attack-type and group classification heads.
        "top_k_attack": [3, 5],
        "top_k_group":  [3, 5],
        "confusion_matrix": True,
        # Top-N groups shown in the confusion matrix (configurable)
        "top_n_cm_groups": 10,
    },

    # ── Callbacks ────────────────────────────────────────────────────────────
    "callbacks": {
        "early_stopping_patience": 5,
        # Metric key from the history dict to monitor (lower is better).
        "early_stopping_monitor": "val_loss",
        "reduce_lr_patience": 3,
        "reduce_lr_factor": 0.5,
        "model_checkpoint_path": "outputs/best_model.pth",
    },

    # ── Reporting ────────────────────────────────────────────────────────────
    "reporting": {
        "output_dir": "outputs",
    },
}
