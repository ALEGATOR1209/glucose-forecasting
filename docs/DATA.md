# Data pipeline and local layout

This repository trains and evaluates forecasting models. It does **not** produce the ML-ready CSVs from raw CGM exports. That work lives in the companion preprocessing repo:

**[GlucoseDAO/glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing)**

Use that repo to download/convert public datasets, resample to 5-minute grids, interpolate small gaps, and emit ML-ready CSVs. Then copy (or symlink) those outputs into this repo under `data/` before training.

## End-to-end flow

```text
raw CGM / pump exports
        │
        ▼
glucose_data_processing  (glucose-process / configs)
        │
        ▼
ML-ready CSVs  (e.g. ai_ready_processed_dataset.csv, loop ML-ready)
        │
        ├─ optional joins in this repo (scripts/loop_ai_ready/)
        │     → loop_ai_ready_joined2.csv
        │
        ▼
glucose-forecasting  (train / tune / evaluate-model)
```

## Local data directory

Licensed datasets (AI-READI, Loop, joined benchmarks) are **not** redistributed with this git repo.

| Path | Role |
|------|------|
| `data/input/` | Preferred place for local ML-ready CSVs (gitignored except `.gitignore`). |
| `data/output/runs/` | Default location for generated training and evaluation artifacts (gitignored except `.gitignore`). |
| `test_data/` | Small demo CSVs shipped in-repo for smoke tests (Livia). |

Historical docs and some TOML/scripts still mention older folders such as:

- `data/loop_and_ai_ready/` — joined Loop + AI-READI benchmarks
- `data/actual/with_complex_steps_processing/` — AI-READI / AI-READI+Type1 wearable CSVs

Those are equivalent *logical* locations. Place files under `data/input/` and pass `--csv data/input/<file>.csv` (or `--test-csv` / `--train-csv`).

`evaluate-model` / training helpers also **resolve by basename**: if checkpoint metadata still points at an absolute Windows path or `data/loop_and_ai_ready/foo.csv`, a file named `foo.csv` under `data/input/` is used automatically.

## Important datasets used by this project

| Logical name | Typical filename | Schema / covariates | Produced by |
|--------------|------------------|---------------------|-------------|
| AI-READI processed | `ai_ready_processed_dataset.csv` | glucose, HR, steps (+ split/group columns) | `glucose_data_processing` with `glucose_config_ai_ready.yaml` |
| AI-READI + Type1 | `ai_ready_plus_type1_*.csv` | wearable schema + small T1DM supplement | preprocessing + project-specific combine |
| Loop ML-ready | loop export CSV | glucose, basal, bolus, carbs | `glucose_data_processing` with `glucose_config_loop.yaml` |
| Joined benchmark | `loop_ai_ready_joined2.csv` (~12M rows) | loop-style columns; ~50% loop T1DM / ~50% ai_ready | `scripts/loop_ai_ready/build_loop_ai_ready_joined2.py` |
| Joined dev subset | `loop_ai_ready_joined2_dev.csv` | same schema, ~1/N rows | same builder (`--dev-output-csv`) |

Column expectations for training/eval are summarized in the root [README.md](../README.md#expected-dataset-columns).

## Building the joined Loop + AI-READI benchmark

After you have ML-ready Loop and AI-READI CSVs from the preprocessing repo:

```bash
uv run python scripts/loop_ai_ready/build_loop_ai_ready_joined2.py --help
```

Related helpers in `scripts/loop_ai_ready/`:

- `join_loop_ai_ready.py` — earlier join variant
- `export_joined_to_loop_csv.py` — export joined parquet → loop-style CSV
- `sample_joined_csv_dev.py` — smaller dev sample

## Demo data (no private exports needed)

| File | Use |
|------|-----|
| `test_data/livia_glumind_ready.csv` | GluMind-shaped personal CGM (~140k rows); no `Recommended Split` |
| `test_data/livia_sugar_one_ready.csv` | Same subject with pump covariates for SugarOne |

See [How_to_run_checkpoint.md](../How_to_run_checkpoint.md) for reviewer smoke-test commands.

## Licensing note

AI-READI and some Loop/JAEB exports require registration and cannot be redistributed. Only share derived artifacts that your data agreements allow. Public demo CSVs under `test_data/` are intentionally small personal/demo traces for CI and reviewers.
