# CLI Reference

Every script supports built-in help when run with `uv`:

| How to run | Help flag |
|------------|-----------|
| Installed console commands (see `pyproject.toml` `[project.scripts]`) | `uv run <command> --help` or `-h` |
| Python entry files | `uv run python scripts/.../script.py --help` or `-h` |

Argparse-based CLIs (`train_glumind.py`, `eval_gluformer_val_test_masked.py`) print defaults via `ArgumentDefaultsHelpFormatter`. Typer apps list each option with `--help`.

---

## `train-glumind` — `scripts/glumind/train_glumind.py`

`uv run train-glumind --help`

| Option | Meaning |
|--------|---------|
| `--csv` | Path to processed dataset CSV (required). |
| `--unique_id` | `sequence_id` or `user_id`: which column defines a series. |
| `--chunk_size` | Reserved for streaming chunk size (default 1_000_000). |
| `--max_train_series` | Cap number of training series; `0` = all. |
| `--max_eval_series` | Cap val/test series; `0` = all. |
| `--drop_interpolated` | Drop rows with `Event Type == Interpolated`. |
| `--mask_interpolated_targets` | Defined in CLI; not wired in the current training loop (no effect). |
| `--study_groups` | Comma-separated `Study Group` filter; empty = all. |
| `--split_scheme` | `classic` or `trainval_test_as_val` (train+val merged, test→val; no held-out test). |
| `--mode` | `global`, `per_group`, `cohort_wise`, or `continual`. |
| `--horizon` | Forecast length in steps (e.g. 12 = 60 min at 5 min). |
| `--input_steps` | History length in steps (e.g. 80). |
| `--d_model`, `--n_heads`, `--n_blocks`, `--ff_units`, `--dropout` | Architecture hyperparameters. |
| `--epochs`, `--batch_size` | Training length and batch size. |
| `--precision` | `fp32`, `bf16`, or `fp16` (mixed precision on CUDA when not fp32). |
| `--compile_mode` | `none`, `default`, `reduce-overhead`, `max-autotune` (`torch.compile`). |
| `--disable_tf32` | Turn off TF32 on CUDA. |
| `--num_workers` | DataLoader workers; `-1` auto (GPU: up to 8). |
| `--prefetch_factor` | Prefetch when `num_workers > 0`. |
| `--lr`, `--weight_decay` | AdamW optimizer. |
| `--patience` | Early stopping on val loss; `0` disables. |
| `--log_every` | Print progress every N epochs. |
| `--ckpt_every_n_epochs` | Save full checkpoint + val/test metrics under `checkpoints/epoch_NNNN/`; `0` off. |
| `--val_every_n_epochs` | Run validation every N epochs. |
| `--resume_from` | Path to `checkpoint.pt` (full state) to resume. |
| `--lwf_lambda` | Learning-without-forgetting weight in `continual` mode. |
| `--continual_order` | `default` or `reverse` study-group order. |
| `--continual_val_scope` | `current_group` or `all_groups` for val in continual mode. |
| `--device` | `cpu`, `mps`, or `cuda`. |
| `--seed` | RNG seed. |
| `--out_dir` | Base output directory for runs. |
| `--save_predictions` | Defined in CLI; not wired in the current training script (no effect). |

---

## `evaluate-glumind` — `scripts/glumind/evaluate_glumind.py`

`uv run evaluate-glumind --help`

| Option | Meaning |
|--------|---------|
| `--registry-dir` | Folder with `_analysis_registry.csv`; picks lowest `val_mae` run. |
| `--run-dir` | Explicit run directory with `tuning_meta.json` / `config.json` and weights. Overrides registry. |
| `--checkpoint` | Specific `.pt` weights; still need `--run-dir` for architecture metadata. |
| `--test-csv` | CSV to score (required). |
| `--train-csv` | CSV to fit MinMax scalers; default from metadata. |
| `--test-split` | If the CSV has `Recommended Split`, keep only this value (e.g. `test`). |
| `--glucose-only` | Ablation: replace HR/steps with zeros or a fixed scaled value. |
| `--default-value` | With `--glucose-only`: `zero`, `mean`, or `median` for HR/steps replacement. |
| `--batch-size` | Override DataLoader batch size (default from metadata). |
| `--device` | Torch device (string, e.g. `cuda` or `cpu`). |

You must pass either `--registry-dir` or `--run-dir`.

For cross-model evaluation (GluMind or SugarOne on any compatible CSV), prefer **`evaluate-model`** below.

---

## `evaluate-model` — `scripts/sugar_one/evaluate_model.py`

`uv run evaluate-model --help`

Unified evaluation for **GluMind** (HR + steps) and **SugarOne** (basal + bolus + carbs). Loads architecture metadata from the run folder, fits MinMax scalers on training rows, and reports **MAE, RMSE, MARD**.

| Option | Meaning |
|--------|---------|
| `--test-csv` | CSV to score (required). |
| `--run-dir` | Run directory with `tuning_meta.json` / `config.json` and `best_model.pt`. |
| `--registry-dir` | Folder with `_analysis_registry.csv`; picks lowest `val_mae` run. |
| `--checkpoint` | Explicit `.pt` weights; still need `--run-dir` for architecture metadata. |
| `--train-csv` | CSV for scaler fitting (default: `csv` from metadata). Override when the training file from metadata is not on disk. |
| `--model-type` | `auto` (detect from checkpoint), `glumind`, or `sugar_one`. |
| `--test-split` | Keep rows where `Recommended Split` equals this value (default `test`). Use `--test-split=''` to score all rows. |
| `--batch-size` | DataLoader batch size (default from metadata). |
| `--device` | Torch device (default `cuda` when available). |
| `--output-json` | Write metrics JSON for batch comparisons. |
| `--log-interval` | Seconds between inference progress logs (default `10`; `0` = first and last only). |
| `--zero-cov` | Zero all non-glucose covariates after imputation (glucose-only inference). Mutually exclusive with `--include-cov` / `--exclude-cov`. |
| `--include-cov` | Comma-separated covariates to keep; zero all other non-glucose channels (e.g. `basal,bolus`). |
| `--exclude-cov` | Comma-separated covariates to zero; keep the rest (e.g. `carbs`). |
| `--covariates` | Print covariate columns and fill stats for `--test-csv`; no checkpoint required. |

Full usage, ablation examples, and alias list: `scripts/sugar_one/README.md`.

You must pass either `--registry-dir` or `--run-dir` (unless using `--covariates` only).

---

## `inference-glumind` — `scripts/glumind/inference_glumind.py`

`uv run inference-glumind --help`

| Option | Meaning |
|--------|---------|
| `--run-dir` | Run directory with metadata and `best_model.pt` / `last_model.pt` (required). |
| `--mode` | `auto` (from `split_scheme` in metadata), `test`, or `val_as_test`. |
| `--glucose-only` | Ablation: zero or constant HR/steps in scaled space. |
| `--default-value` | `zero`, `mean`, or `median` (non-glucose channels). |
| `--device` | Torch device. |

Re-runs inference on the **training CSV** from metadata and compares to saved metrics when present.

---

## `download-glumind-hf` — `scripts/glumind/download_from_huggingface.py`

`uv run download-glumind-hf --help`

| Option | Meaning |
|--------|---------|
| `--repo-id` | Hugging Face model repo id, e.g. `OrgName/model-name`. |
| `--output-dir` | Local directory for downloaded files. |
| `--token` | Access token (private repos); empty for public. |
| `--revision` | Branch, tag, or commit (default `main`). |

Skips `checkpoints/` and `README.md` in the remote repo.

---

## `upload_to_huggingface.py` (not a console script)

`uv run python scripts/glumind/upload_to_huggingface.py --help`

| Option | Meaning |
|--------|---------|
| `--model-dir` | Local run directory with weights and JSON metadata. |
| `--repo-name` | Repo name under the org, e.g. `glumind-global-h12`. |
| `--org` | Hugging Face organization name. |
| `--token` | Write token. |
| `--private` / `--public` | Create private repo (default public). |

---

## `tune-sugar-one` — `scripts/sugar_one/tune_sugar_one.py`

`uv run tune-sugar-one --help`

Random hyperparameter search for SugarOne (global mode only).

| Option | Meaning |
|--------|---------|
| `--config`, `-c` | TOML config path (default: `scripts/sugar_one/tune_sugar_one_full.toml`). |
| `--device` | `cuda`, `cpu`, or `mps` (default `cuda`). |
| `--seed` | Override `.random_seed` from the config. |

Shipped configs: `tune_sugar_one_full.toml` (production) and `tune_sugar_one_dev.toml` (laptop).

---

## `scripts/glumind_uni/train_uniglumind.py` (GluMindUni)

`uv run python scripts/glumind_uni/train_uniglumind.py train --help`

Typer subcommand `train` (glucose-only model). Options match `train-glumind` except: glucose-only inputs; default `--out-dir` is `data/output/runs/glumind_uni`.

---

## `scripts/sugar_one/train_sugar_one.py` (SugarOne)

`uv run python scripts/sugar_one/train_sugar_one.py --help`

Root Typer command (no subcommand name). Same shape as GluMindUni: insulin/carb covariates, default `--out-dir data/output/runs/sugar_one`, `--csv` should be the joined Loop + AI-READI CSV.

---

## `eval_gluformer_val_test_masked.py`

`uv run python scripts/eval_gluformer_val_test_masked.py -h`

| Option | Meaning |
|--------|---------|
| `--csv` | Dataset CSV (required). |
| `--unique_id` | `sequence_id` or `user_id`. |
| `--model_id` | Hugging Face model id (default `njeffrie/Gluformer`). |
| `--device` | `cpu`, `mps`, or `cuda`. |
| `--splits` | `val`, `test`, or `both`. |
| `--chunk_size` | Streaming read chunk size. |
| `--max_eval_series` | Subsample series; `0` = all. |
| `--max_points_per_series` | Truncate each series. |
| `--drop_interpolated` | Remove interpolated rows before eval. |
| `--mask_interpolated_targets` | Exclude interpolated targets from metrics. |
| `--out_dir` | Base run output directory. |
| `--save_predictions` | Save per-row predictions. |

---

## Expected dataset columns

Produce these CSVs with [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing), then place them under `data/input/` (see [docs/DATA.md](docs/DATA.md)).

**GluMind / AI-READI-style:**
`sequence_id`, `User ID`, `Timestamp (YYYY-MM-DDThh:mm:ss)`, `Recommended Split` (`train`/`val`/`test`), `Study Group`, `Event Type`, `Glucose Value (mg/dL)`, `Heart Rate`, `Step Count`

**Loop / SugarOne:**
`Glucose (mg/dL)` or `Glucose Value (mg/dL)`, `Basal Rate (U/h)`, `Bolus Insulin (U)`, `Carbohydrates (g)`

---

## Checkpoints and model reuse

GluMind / SugarOne checkpoints are saved as:
- `best_model.pt` / `last_model.pt` — plain `state_dict`, loaded with `weights_only=True`
- `checkpoint.pt` / `last_checkpoint.pt` — full training state (optimizer, scheduler, epoch)

Load a checkpoint without the training script:

```python
import torch
from glucose_forecasting.models.sugar_one import SugarOneModel

model = SugarOneModel(
    n_time_steps=128, n_features=4, d_model=32, n_heads=8,
    ff_units=128, n_blocks=5, prediction_horizon=12, dropout=0.1
)
state = torch.load("path/to/best_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
```

---

## Run artifacts

Typical outputs under `data/output/runs/<run_name>/`:

```
val_metrics_overall.csv
val_metrics_by_study_group.csv
test_metrics_overall.csv
test_metrics_by_study_group.csv
tuning_meta.json / config.json
best_model.pt / last_model.pt
checkpoints/
```

---

## Notes

- In `trainval_test_as_val` mode, held-out test metrics are intentionally disabled.
- `--mask_interpolated_targets` / `--save_predictions` on `train-glumind` are defined in the CLI but not wired in the current training loop.
