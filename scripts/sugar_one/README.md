# evaluate-model — unified GluMind / SugarOne evaluation CLI

`scripts/sugar_one/evaluate_model.py` is registered as **`uv run evaluate-model`**.

It loads a trained checkpoint, fits MinMax scalers on training rows, runs sliding-window inference on a CSV, and reports **MAE, RMSE, and MARD**. The same entry point works for:

| Model | Covariate channels | Typical CSV schema |
|-------|-------------------|-------------------|
| **GluMind** | glucose, heart rate, steps | `ai_ready` wearable export |
| **SugarOne** | glucose, basal, bolus, carbs | loop / pump export (`loop_ai_ready_joined2.csv`) |

SugarOne was formerly named **GluMindIC**. ML-ready CSVs come from [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing); local layout: [docs/DATA.md](../../docs/DATA.md).

---

## Quick start

From the repository root (after `uv sync`). Paths below use `data/input/`; symlink or adjust if your files live under `data/loop_and_ai_ready/`.

```bash
# Inspect covariates in a CSV (no model required)
uv run evaluate-model \
  --test-csv data/input/ablation_test.csv \
  --covariates \
  --model-type sugar_one

# Evaluate SugarOne on the loop benchmark test split
uv run evaluate-model \
  --run-dir data/output/runs/sugar_one_tune/production/trial_0000_bcd3813f \
  --model-type sugar_one \
  --test-csv data/input/loop_ai_ready_joined2.csv \
  --train-csv data/input/loop_ai_ready_joined2.csv \
  --batch-size 256 \
  --output-json data/output/runs/comparison_loop/sugar_one_trial0.json
```

Bundled reviewer checkpoints (demo CSV; add `--test-split ''`):

```bash
uv run evaluate-model \
  --run-dir test_model_sugar_one \
  --model-type sugar_one \
  --test-csv test_data/livia_sugar_one_ready.csv \
  --train-csv test_data/livia_sugar_one_ready.csv \
  --test-split ''
```

---

## Required arguments

| Flag | Required when | Meaning |
|------|---------------|---------|
| `--test-csv` | always | CSV to score or inspect. |
| `--run-dir` or `--registry-dir` | evaluation (not `--covariates`) | Checkpoint + metadata location. |

---

## Model resolution

| Flag | Default | Meaning |
|------|---------|---------|
| `--run-dir` | — | Folder with `tuning_meta.json` / `config.json` and `best_model.pt`. |
| `--registry-dir` | — | Folder with `_analysis_registry.csv`; picks lowest `val_mae` run. |
| `--checkpoint` | `best_model.pt` | Explicit `.pt` weights; still needs `--run-dir` for architecture metadata. |
| `--model-type` | `auto` | `glumind`, `sugar_one`, or auto-detect from checkpoint keys. |
| `--train-csv` | metadata `csv` field | CSV used to fit MinMax scalers (override when training file is not on disk). |

---

## Data loading

| Flag | Default | Meaning |
|------|---------|---------|
| `--test-split` | `test` | Keep rows where `Recommended Split` equals this value. Use `--test-split=''` to score all rows. |

Column names are resolved automatically (see [Covariate mapping](#covariate-mapping)). Missing covariate columns are loaded as **0.0** before imputation.

Imputation matches training:

- **SugarOne:** basal forward/back-fill; bolus and carbs event-only (null → 0).
- **GluMind:** glucose forward/back-fill; HR and steps forward/back-fill.

---

## Covariate inspection (`--covariates`)

Print which covariate columns exist in `--test-csv`, how many rows have non-empty values, and which names are accepted by `--include-cov` / `--exclude-cov`.

```bash
uv run evaluate-model \
  --test-csv data/input/ablation_test.csv \
  --covariates \
  --model-type sugar_one
```

With `--model-type auto` (default), both GluMind and SugarOne mappings are shown.

No checkpoint or GPU is needed. Respects `--test-split` when counting filled rows.

---

## Covariate ablation at inference

Covariates are zeroed **after imputation** so forward-filled basal rates on loop rows do not leak back when ablated. Scalers are still fit on the full training covariates from `--train-csv`.

| Flag | Effect |
|------|--------|
| *(none)* | Use all model covariates present in the CSV. |
| `--zero-cov` | Zero **all** non-glucose covariates. Same as excluding every auxiliary channel. |
| `--include-cov NAME[,NAME...]` | Keep **only** the listed covariates; zero the rest. |
| `--exclude-cov NAME[,NAME...]` | Zero the listed covariates; keep the rest. |

`--zero-cov` is mutually exclusive with `--include-cov` and `--exclude-cov`.

### SugarOne canonical names

`basal`, `bolus`, `carbs`

### GluMind canonical names

`hr`, `steps`

### Accepted aliases (case-insensitive)

| Canonical | Aliases |
|-----------|---------|
| `basal` | `basal_rate`, `basal rate`, `basalrate` |
| `bolus` | `bolus_insulin`, `bolus insulin`, `insulin` |
| `carbs` | `carb`, `carbohydrates`, `carbohydrate` |
| `hr` | `heart_rate`, `heart rate`, `heartrate` |
| `steps` | `step`, `step_count`, `step count`, `stepcount` |

Glucose is always required and cannot be included/excluded.

### Ablation examples (SugarOne)

```bash
# Glucose only (same as --zero-cov)
uv run evaluate-model ... --zero-cov

# Basal + bolus only (zero carbs)
uv run evaluate-model ... --include-cov basal,bolus

# All except carbs
uv run evaluate-model ... --exclude-cov carbs

# Basal only
uv run evaluate-model ... --include-cov basal
```

Typical T1DM ablation workflow on `data/input/ablation_test.csv`:

```bash
uv run evaluate-model \
  --run-dir data/output/runs/sugar_one_tune/production/trial_0000_bcd3813f \
  --model-type sugar_one \
  --test-csv data/input/ablation_test.csv \
  --train-csv data/input/loop_ai_ready_joined2.csv \
  --test-split '' \
  --include-cov basal \
  --output-json data/output/runs/ablation/basal_only.json
```

Repeat with `--include-cov bolus`, `--include-cov carbs`, `--include-cov basal,bolus`, etc., and compare JSON outputs.

---

## Performance and output

| Flag | Default | Meaning |
|------|---------|---------|
| `--batch-size` | from metadata | DataLoader batch size. |
| `--device` | `cuda` if available | Torch device. |
| `--log-interval` | `10` | Seconds between inference progress logs (`0` = first and last only). |
| `--output-json` | — | Write metrics (+ covariate selection) as JSON. |

Console output includes MAE, RMSE, MARD, active/zeroed covariates, and window count.

JSON payload fields include `active_covariates`, `zeroed_covariates`, `include_cov`, `exclude_cov`, and `zero_cov`.

---

## Covariate mapping

### GluMind

| Channel | CSV column aliases |
|---------|-------------------|
| glucose | `Glucose Value (mg/dL)`, `Glucose (mg/dL)` |
| hr | `Heart Rate` |
| steps | `Step Count` |

### SugarOne

| Channel | CSV column aliases |
|---------|-------------------|
| glucose | `Glucose Value (mg/dL)`, `Glucose (mg/dL)` |
| basal | `Basal Rate (U/h)` |
| bolus | `Bolus Insulin (U)` |
| carbs | `Carbohydrates (g)` |

Unique series ID: `sequence_id` or `User ID` (from run metadata / `tuning_meta.json`).

Optional columns: `Recommended Split`, `Study Group`, `Event Type` (use `--drop_interpolated` from metadata when set).

---

## Cross-domain notes

- **GluMind on loop CSV:** no HR/steps columns → channels are 0-filled; use `--zero-cov` for an explicit glucose-only baseline.
- **SugarOne on wearable CSV:** no insulin columns → basal/bolus/carbs are 0-filled; `--zero-cov` ablates any residual signal after imputation.
- Scalers should be fit on the **same domain** as training when possible (`--train-csv` override).

See also `docs/GLUMIND_VS_SUGARONE_COMPARISON.md` for benchmark numbers and interpretation.

---

## Related commands

| Command | Scope |
|---------|-------|
| `uv run evaluate-glumind` | GluMind only; re-runs on training CSV from metadata. |
| `uv run inference-glumind` | GluMind only; compares to saved metrics files. |

For GluMind-specific Hugging Face download and Livia sanity checks, see `scripts/glumind/README.md`.
