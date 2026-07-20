# GluMindUni — Univariate Glucose Transformer

Univariate variant of the GluMind model. Removes all covariate cross-attention
branches (Heart Rate, Step Count) and focuses purely on the glucose time-series.

## Architecture differences vs. GluMind

| Component | GluMind (multimodal) | GluMindUni (univariate) |
|---|---|---|
| Input channels | 3 — glucose, HR, steps | 1 — glucose only |
| Cross-attention block | ✅ glucose queries HR & steps | ❌ removed |
| Multi-scale self-attention | ✅ DS=1/2/4 | ✅ DS=1/2/4 (unchanged) |
| Parallel block fusion | ✅ cross + multi-scale | ❌ not needed |
| Parameters (default) | ~3× more per block | leaner, faster |

The multi-scale self-attention (resolutions DS=1, DS=2, DS=4) is preserved because
it captures short-, medium-, and long-range temporal patterns which are just as
important in univariate glucose forecasting.

## Files

| File | Description |
|---|---|
| `glumind_uni_model.py` | Model-only module (`GluMindUniModel`). Safe to import without training deps. |
| `train_uniglumind.py` | Full training pipeline with CLI (Typer). |

## Quick start

```bash
# Global model, 1 epoch smoke test
uv run scripts/glumind_uni/train_uniglumind.py train \
    --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
    --epochs 1 \
    --batch-size 128 \
    --out-dir data/output/runs/glumind_uni
```

## Training modes

| Mode | Description |
|---|---|
| `global` | Single model trained on all cohorts |
| `per_group` | One model per study group |
| `cohort_wise` | Sequential training, fresh model per cohort |
| `continual` | Sequential training with LwF distillation |

## Key CLI options

```
--csv PATH              Processed dataset CSV (required)
--mode TEXT             global | per_group | cohort_wise | continual  [global]
--horizon INT           Prediction steps (12 = 60 min)  [12]
--input-steps INT       Input window steps (80 = 400 min)  [80]
--d-model INT           Embedding dimension  [32]
--n-heads INT           Attention heads  [4]
--n-blocks INT          Transformer blocks  [3]
--ff-units INT          FFN hidden units  [128]
--dropout FLOAT         Dropout  [0.1]
--epochs INT            Training epochs  [200]
--batch-size INT        Batch size  [64]
--lr FLOAT              Learning rate  [1e-3]
--patience INT          Early stopping patience (0=off)  [20]
--device TEXT           cpu | cuda | mps  [cuda]
--out-dir PATH          Output directory  [data/output/runs/glumind_uni]
--resume-from TEXT      Path to checkpoint.pt to resume
--lwf-lambda FLOAT      LwF distillation weight (continual mode)  [0.5]
--precision TEXT        fp32 | bf16 | fp16  [bf16]
```

## Output structure

Each run writes to `<out-dir>/<run-name>/`:

```
best_model.pt               # weights at best validation loss
last_model.pt               # weights at final epoch
last_checkpoint.pt          # full checkpoint (model + optimizer + scheduler)
best_info.json              # epoch and val_loss for best_model.pt
config.json                 # hyperparameter snapshot
tuning_meta.json            # dataset sizes and start time
val_metrics_overall.csv     # MAE / RMSE / MARD on validation set
val_metrics_by_study_group.csv
test_metrics_overall.csv    # MAE / RMSE / MARD on test set
test_metrics_by_study_group.csv
checkpoints/epoch_NNNN/     # periodic checkpoints (--ckpt-every-n-epochs)
```

## Resuming training

```bash
uv run scripts/glumind_uni/train_uniglumind.py train \
    --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
    --resume-from data/output/runs/glumind_uni/my_run/last_checkpoint.pt \
    --epochs 100
```
