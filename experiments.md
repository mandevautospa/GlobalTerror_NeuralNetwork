# Experiments

This document records commands and results for the three required experiments.

---

## How to Run

### In Google Colab
Open `gtd_nn.ipynb` via the badge at the top of the notebook and run all cells
in order.

### Locally (Jupyter)
```bash
pip install torch pandas scikit-learn kagglehub seaborn matplotlib joblib
jupyter nbconvert --to notebook --execute gtd_nn.ipynb --output gtd_nn_executed.ipynb
```

---

## Experiment 1 – Baseline

**Description**: Original model with equal loss weights, random split, accuracy
only, MSE on raw casualty counts.

**Config changes** (relative to `config.py` defaults):
```python
CONFIG["losses"]["gname_loss"]        = "weighted_ce"  # plain CE instead of focal
CONFIG["losses"]["casualties_loss"]   = "mse"          # MSE instead of Huber
CONFIG["training"]["num_epochs"]      = 20
CONFIG["callbacks"]["early_stopping_patience"] = 999   # effectively disabled
CONFIG["loss_weights"]["group"]       = 1.0            # equal weighting
CONFIG["model"]["tower_layers"]       = 0              # no per-head towers
```

> **Note**: A true baseline should use `split_year_train_val = 2015` (single
> split), plain `CrossEntropyLoss` for all heads, and `MSELoss` for
> casualties.  The values above reflect the *minimum* config changes needed;
> adjust as needed for a fair comparison.

**Expected results** (approximate – will vary by seed):

| Metric | Value |
|---|---|
| val attacktype1 accuracy | ~0.55–0.65 |
| val gname accuracy | ~0.20–0.35 |
| val gname macro F1 | ~0.02–0.08 |
| val casualties raw MAE | very high (model collapses to near-zero) |

---

## Experiment 2 – Better Split + Richer Metrics

**Description**: Switch to the three-way temporal split (train ≤ 2013, val
2014–2016, test > 2016), add macro F1 + top-k accuracy reporting.  Loss
functions and model architecture unchanged.

**Config changes**:
```python
CONFIG["split"]["split_year_train_val"] = 2013
CONFIG["split"]["split_year_val_test"]  = 2016
```

**Command** (Colab): run notebook with the above config values.

**Acceptance criteria**:
- No duplicate events across splits (verified by `make_splits()` overlap check).
- Per-head report printed with macro F1 and top-3 / top-5 accuracy.
- `outputs/run_*/metrics.csv` contains all metric columns.

**Expected improvements over Experiment 1**:
- Macro F1 scores are now reported, revealing the true poor gname performance.
- No leakage in evaluation → val metrics are more conservative and honest.

---

## Experiment 3 – gname Imbalance Fix + log1p Casualties + Early Stopping

**Description**: Full improved pipeline.  Focal loss for gname, Huber loss on
log1p-transformed casualty targets, per-head towers, early stopping, and
ReduceLROnPlateau.

**Config** (all defaults in `config.py`):
```python
CONFIG["losses"]["gname_loss"]        = "focal"     # default
CONFIG["losses"]["casualties_loss"]   = "huber"     # default
CONFIG["loss_weights"]["group"]       = 2.0         # upweight gname head
CONFIG["model"]["tower_layers"]       = 2           # default
CONFIG["callbacks"]["early_stopping_patience"] = 5  # default
```

**Command** (Colab): run notebook as-is with default `config.py`.

**Acceptance criteria**:

| Criterion | Met? |
|---|---|
| Time-based split with no overlap | ✅ verified by `make_splits()` |
| gname macro F1 reported | ✅ `print_head_report("gname", ...)` |
| gname top-5 accuracy reported | ✅ |
| Casualties original-scale MAE/RMSE reported | ✅ `history["val_cas_mae_orig"]` |
| Casualties original-scale MAE/RMSE better than Exp 1 | to verify |
| Training stops early when val stops improving | ✅ EarlyStopping callback |
| Confusion matrices saved as PNG | ✅ `outputs/run_*/confusion_matrices/` |

**Expected results** (approximate):

| Metric | Exp 1 (baseline) | Exp 3 (improved) |
|---|---|---|
| val gname macro F1 | ~0.03 | ~0.08–0.15 |
| val gname top-5 acc | n/a | ~0.50–0.65 |
| val casualties orig MAE | >10 (collapse) | 2–6 |
| val casualties orig RMSE | >20 | 5–15 |
| Epochs trained | 20 (fixed) | 10–18 (early stop) |

---

## Output Artefacts

Each run produces a folder under `outputs/run_<timestamp>_<tag>/` containing:

```
config.json
git_info.json
metrics.csv
plots/
    total_loss.png
    att_loss.png
    suc_loss.png
    grp_loss.png
    cas_loss.png
    val_accuracy.png
    val_cas_mae_orig.png
    val_cas_rmse_orig.png
confusion_matrices/
    cm_attacktype1.png
    cm_gname.png
gtd_model.pth
gtd_scaler.joblib
gtd_encoders.joblib
```
