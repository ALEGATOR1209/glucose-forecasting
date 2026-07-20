# GluMind Inference & Evaluation

Prefer the unified **`evaluate-model`** CLI for new work (GluMind and SugarOne). This page documents the GluMind-only `inference-glumind` helper.

For reviewer smoke tests, see [How_to_run_checkpoint.md](../How_to_run_checkpoint.md).

## Core Features

- **Auto-Detection**: Derives the evaluation mode (`test` or `val_as_test`) from `tuning_meta.json` using the `split_scheme`.
- **Reproducibility Check**: Compares newly computed metrics against saved metrics in the run directory.
- **Feature Sensitivity**: `--glucose-only` replaces non-glucose features (HR, Steps) with a constant.

## Usage

```bash
uv run inference-glumind --run-dir <path_to_run> [OPTIONS]
```

| Option | Description |
| :--- | :--- |
| `--run-dir` | **Required**. Path to the run directory. |
| `--mode` | `auto` (default), `test`, or `val_as_test`. |
| `--glucose-only` | Replace HR/Steps with a default value. |
| `--default-value` | `zero` (default), `mean`, or `median`. |
| `--device` | Torch device (`cpu` or `cuda`). |

## Examples

```bash
uv run inference-glumind \
  --run-dir marked_runs/glumind/ai_ready/glumind_global_h12_20260222_194108

uv run inference-glumind \
  --run-dir marked_runs/glumind/type1_only/glumind_global_h12_20260225_120905 \
  --glucose-only \
  --default-value mean

uv run inference-glumind \
  --run-dir marked_runs/glumind/ai_ready/glumind_global_h12_20260222_194108 \
  --mode val_as_test
```

## Internal Logic

1. Load `tuning_meta.json` / `config.json` for architecture and data params.
2. Stream splits via `load_splits_streaming`.
3. `classic` → evaluate `test_df`; `trainval_test_as_val` → evaluate `val_df`.
4. Diff against saved `*_metrics_overall.csv` when present.

Training CSVs referenced in run metadata must be available locally (see [DATA.md](DATA.md)).
