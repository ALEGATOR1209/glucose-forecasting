# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Training, tuning, and evaluation pipelines for blood-glucose forecasting from CGM (continuous glucose monitor) data, on AI-READI-style and Loop-pump-style datasets.

**Data preprocessing is a separate repo:** [GlucoseDAO/glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing). This forecasting repo consumes ML-ready CSVs (prefer `data/input/`, gitignored). See `docs/DATA.md`.

**Naming:** BGI text called the wearable multimodal model **“GluMind (Ours)”** and the insulin/carb adaptation **GluMindIC**. GluMindIC was renamed to **SugarOne**. SugarOne is the current primary model for pump/loop data; GluMind remains the HR/steps baseline. Details: `docs/MILESTONES.md`.

Model variants (shared training/eval scaffolding, different covariates):

- **SugarOne** (`scripts/sugar_one/`) — glucose + basal rate + bolus insulin + carbohydrates (Loop pump data), 3-way cross-attention with learnable softmax mixing weights. Formerly GluMindIC.
- **GluMind** (`scripts/glumind/`) — glucose + heart rate + step count. Independent reimplementation of Farahmand et al., 2025b (arXiv:2509.18457): parallel cross-attention multimodal fusion + multi-scale self-attention, optional LwF for continual cross-cohort training.
- **GluMind-Uni** (`scripts/glumind_uni/`) — glucose-only variant of the same architecture.
- **NeuralForecast baselines** (`src/glucose_forecasting/backends/neuralforecast/`) — package-native fixed-split holdout and rolling cross-validation evaluation, selected through `glucose train --backend neuralforecast --eval ...`.
- **GluFormer** (`scripts/eval_gluformer_val_test_masked.py`) — evaluation only, against a pretrained Hugging Face model (`njeffrie/Gluformer`).

Forecast horizon defaults to 12 steps = 60 minutes at 5-minute sampling frequency.

## Commands

Environment (Python >=3.12, managed with `uv`):
```bash
uv sync
```

Run tests:
```bash
uv run pytest -q                                      # full suite
uv run pytest tests/test_train_cli_smoke.py -q        # parametrized train smokes (all model families)
uv run pytest tests/test_evaluate_model_covariates.py -q
```
No lint/format command is configured in `pyproject.toml`.

**Device selection:** never hardcode `--device cpu` or `device="cpu"` in commands, scripts, or tests. Always use `device="auto"` (the default), which detects CUDA → MPS → CPU. If the machine has a GPU, use it.

**Testing style (also in `AGENTS.md`):** prefer behavioral smokes over thin helper unit files; parametrize GluMind/SugarOne/Uni (do not copy a whole test per model); construct shared strings/flags/dims from constants or tables instead of pasting argv/artifact literals. Patterns: `tests/test_train_cli_smoke.py`, `tests/test_models_forward.py`, `tests/test_datasets.py`.

Installed console commands (defined in `pyproject.toml` `[project.scripts]`, all runnable as `uv run <name> --help`):
- `train-glumind` → `scripts/glumind/train_glumind.py:main` (argparse CLI)
- `evaluate-glumind` → `scripts/glumind/evaluate_glumind.py:app` (Typer; GluMind-only)
- `evaluate-model` → `scripts/sugar_one/evaluate_model.py:app` (Typer; **unified** GluMind + SugarOne eval — preferred for new work)
- `inference-glumind` → `scripts/glumind/inference_glumind.py:app`
- `tune-sugar-one` → `scripts/sugar_one/tune_sugar_one.py:app` (TOML-driven random hyperparameter search)
- `download-glumind-hf` → `scripts/glumind/download_from_huggingface.py:app`

Scripts without a console entry point are run directly, e.g.:
```bash
uv run python scripts/sugar_one/train_sugar_one.py --help          # Typer, no subcommand name
uv run python scripts/glumind_uni/train_uniglumind.py train --help # Typer, `train` subcommand
uv run glucose train --backend neuralforecast --data DATA.csv --help
uv run python scripts/eval_gluformer_val_test_masked.py -h          # argparse
uv run python scripts/glumind/upload_to_huggingface.py --help
```

Full flag reference and worked examples for every script live in the root `README.md` — read that before guessing at a flag name; it documents which flags exist per-CLI (argparse vs. Typer scripts have different flags even for equivalent features, e.g. `--csv` vs `--csv`, snake_case vs kebab-case).

Fast smoke test after code changes (no GPU, no full dataset needed):
```bash
uv run evaluate-model --run-dir test_model_sugar_one --model-type sugar_one \
 --test-csv test_data/livia_sugar_one_ready.csv --train-csv test_data/livia_sugar_one_ready.csv \
 --test-split '' --batch-size 256
```
Bundled checkpoints: `test_model_sugar_one/`, `test_model_glumind/`. Demo CSVs under `test_data/` have no usable `Recommended Split` — always pass `--test-split ''` and `--train-csv` pointing at the demo file. Full commands: `How_to_run_checkpoint.md`.

## Architecture

### Shared utilities: `scripts/common/`

As of the last refactor, model-agnostic logic that used to be duplicated across the training scripts (and re-imported cross-model, e.g. `evaluate_model.py` importing from `train_glumind.py`) now lives here. **New model variants or eval tools should use these instead of reimplementing.**

- `data_loading.py` — `load_splits_streaming`, `apply_split_scheme` (`classic` vs `trainval_test_as_val`), `impute_and_sort` (per-series forward/backward-fill for continuous signals like glucose/HR/basal, zero-fill for discrete signals like bolus/carbs), `limit_series`, `normalize_study_group_label`/`normalize_study_groups_column`, `resolve_num_workers`. Column names are passed in by the caller (`value_columns: dict[str, str]` mapping canonical → source CSV column) since each model variant reads a different CSV column set.
- `metrics.py` — `mae_rmse_mard` (MAE/RMSE/MARD — the metric triple used everywhere in this repo), `per_study_group_breakdown`, `overall_metrics_to_csv`.
- `checkpoint.py` — `save_full_checkpoint`/`load_full_checkpoint` (generalized across the three slightly different checkpoint shapes used by GluMind/SugarOne/GluMind-Uni via a `config_key` param — `"args"`, `"config"`, or `"cfg"`), `read_checkpoint_meta`, `update_latest_symlink`, `strip_compile_prefix` (strips the `_orig_mod.` prefix `torch.compile` adds to state_dict keys).
- `registry.py` — `find_best_run_dir` (reads `_analysis_registry.csv`, picks lowest `val_mae`), `load_run_meta` (reads `tuning_meta.json` or `config.json`), `resolve_checkpoint` (finds `best_model.pt`/`last_model.pt`), `resolve_csv_path`.
- `evaluation.py` — covariate alias/ablation machinery (`GLUMIND_COVARIATES`, `SUGAR_ONE_COVARIATES`, `COVARIATE_NAME_ALIASES`, `_alias_to_canonical`, `_parse_covariate_names`, `_resolve_covariate_zeroing`, `_load_csv_flexible`) and the shared inference loop (`_run_evaluate`), extracted from `evaluate_model.py`.

`train_glumind.py`, `train_sugar_one.py`, `train_uniglumind.py`, `evaluate_glumind.py`, and `evaluate_model.py` all import from `scripts/common/*` internally but **re-export the same names under their original locations** — e.g. `from scripts.glumind.train_glumind import load_splits_streaming` still works. Do not break this: `evaluate_model.py` still cross-imports several names from `train_glumind.py`/`train_sugar_one.py` by their original names, and `tests/test_evaluate_model_covariates.py` imports covariate helpers from `scripts.sugar_one.evaluate_model` directly.

### Model files — checkpoint-friendly, kept separate from training logic

- `scripts/glumind/glumind_model.py` — `GluMindModel`. 2-auxiliary parallel cross-attention (HR, steps) with fixed averaging + multi-scale self-attention.
- `scripts/sugar_one/sugar_one_model.py` — `SugarOneModel`. 3-auxiliary parallel cross-attention (basal, bolus, carbs) with **learnable softmax mixing weights** (the main architectural difference from GluMind).
- `scripts/glumind_uni/glumind_uni_model.py` — glucose-only variant, same block structure.

These are intentionally isolated from the training scripts so a checkpoint can be loaded with just the model file + `torch.load(..., weights_only=True)` — no training-script imports needed (see README.md "Checkpoints and Model Reuse" for a minimal load example). `PositionalEncoding` and much of `MultiScaleAttentionBlock` are near-duplicated between `glumind_model.py` and `sugar_one_model.py`; this has been left as-is (not deduplicated) to avoid touching model internals and risking checkpoint/behavior drift — treat these files as intentionally frozen unless a change is specifically requested.

### Checkpoints

Two kinds are saved per run, both plain `torch.save` of dicts (no pickled class references, so moving code around is safe as long as model `state_dict` key names don't change):
- `best_model.pt` / `last_model.pt` — plain `state_dict` only, loaded with `weights_only=True`.
- `checkpoint.pt` / `last_checkpoint.pt` — full training state (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, epoch, best_val_loss, config), loaded with `weights_only=False`, used to `--resume_from` a training run.

Architecture hyperparameters live separately in `tuning_meta.json` / `config.json` inside the run directory, not in the checkpoint — evaluation/inference scripts read that JSON to reconstruct the right model shape before loading weights.

### Training modes

All training scripts (GluMind, SugarOne, GluMind-Uni) support the same four modes: `global` (one model, all study groups), `per_group` (independent model per study group), `cohort_wise`, and `continual` (sequential per-group training with optional LwF regularization via `--lwf_lambda` to reduce catastrophic forgetting across cohorts).

### CLI framework split

`train_glumind.py` and `eval_gluformer_val_test_masked.py` use argparse; `train_sugar_one.py`, `train_uniglumind.py`, `evaluate_glumind.py`, `evaluate_model.py`, `tune_sugar_one.py`, and `glucose train` use Typer. This is a known inconsistency (not enforced) — check which framework a script uses before assuming flag syntax (argparse: `--snake_case`; Typer: `--kebab-case`).

### CLI naming convention

For new CLIs, avoid long hyphenated executable names. Prefer one concise root Typer command with action-first subcommands, such as `glucose forecast`, `glucose train`, and `glucose models`. Select a model or backend with options, for example `glucose train --model sugarone` or `glucose train --backend neuralforecast --eval holdout`. Use `--eval holdout` or `--eval cross-val` when selecting an ML evaluation protocol; reserve “workflow” for multi-step orchestration. Use kebab-case only for multi-word option names when necessary. Do not rename legacy commands solely for this convention; preserve their compatibility.

### Data expectations

ML-ready CSVs come from **glucose_data_processing**, not this repo. Put them in `data/input/` (or symlink historical paths — see `docs/DATA.md`).

Core AI-READI CSV columns: `sequence_id`, `User ID`, `Timestamp (YYYY-MM-DDThh:mm:ss)`, `Recommended Split` (`train`/`val`/`test`), `Study Group`, `Event Type`, `Glucose Value (mg/dL)`, `Heart Rate`, `Step Count`.

Loop/SugarOne CSVs additionally/instead have: `Glucose (mg/dL)` (or `Glucose Value (mg/dL)`), `Basal Rate (U/h)`, `Bolus Insulin (U)`, `Carbohydrates (g)`.

`evaluate-model` resolves column aliases automatically and can zero out or ablate individual covariates at inference time (`--zero-cov`, `--include-cov`, `--exclude-cov`) for cross-model/cross-covariate comparison — this is the main tool for comparing GluMind vs. SugarOne on the same data.

`scripts/loop_ai_ready/` joins Loop + AI-READI ML-ready CSVs into `loop_ai_ready_joined2.csv` after preprocessing.

### Known dead/unwired flags

`--mask_interpolated_targets` and `--save_predictions` are defined in some CLIs' argument parsers but have no effect in the current training loop for `train_glumind.py` (per README.md) — don't assume they work without checking the specific script.

### Reports and run artifacts

`data/output/runs/` holds generated training outputs (checkpoints, per-split metrics CSVs, `tuning_meta.json`). `marked_runs/` is a separate curated/annotated subset with `RUNS_ANALYSIS.md` writeups per model/dataset combo. `reports/` and `docs/reports/` hold longer-form comparison and milestone writeups (e.g. `docs/GLUMIND_VS_SUGARONE_COMPARISON.md`, `docs/T1DM_COVARIATE_ABLATION_REPORT.md`). Root `CROSS_MODEL_COMPARISON.md` is the cross-model summary.

## Things I don't know / should ask about

- Whether `docs/MILESTONE_6_MODEL_ARCHITECTURE_STATUS.md` and `docs/MILESTONE_6_SELECTION_EVALUATION_REPORT.md` (untracked, not yet committed) represent decisions that should change how new work is scoped — worth reading before starting architecture work. (Confirmed: these are temporary working files left over from preparing the milestone 6 report, not necessarily final decisions.)
