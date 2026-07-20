# GluMind vs SugarOne — Joined Benchmark Comparison Report

**Date:** 2026-06-06 (updated with `test_data` demo evaluation)  
**Evaluation script:** `scripts/sugar_one/evaluate_model.py` (`uv run evaluate-model`)  
**Primary benchmark:** `loop_ai_ready_joined2.csv` (test split) — place under `data/input/` (or the historical `data/loop_and_ai_ready/` path); see [DATA.md](DATA.md).

> **Naming:** SugarOne was called **GluMindIC** in early Milestone 07 drafts. BGI “GluMind (Ours)” refers to wearable **GluMind**, not SugarOne.  
**Demo / sanity check:** `test_data/livia_sugar_one_ready.csv` (all models; bundled reviewer checkpoints)

## Datasets and terminology

This report compares models on a **joined benchmark** built from two distinct source datasets. In project naming, **`ai_ready`** and **`loop`** are peer dataset names — not “loop” as a dataset and “ai_ready” as a modifier.

| Source dataset | Role | Typical covariates | Study groups | Users (joined2) |
|----------------|------|--------------------|--------------|-----------------|
| **`ai_ready`** | Wearable CGM cohort export (`ai_ready_full4.csv`) | glucose, heart rate, step count | Healthy, pre-diabetes, oral-T2DM, insulin-dependent T2DM | 2,232 |
| **`loop`** | Pump/CGM closed-loop cohort export (`loop.csv`) | glucose, basal rate, bolus insulin, carbohydrates | T1DM only | 60 |

**`loop_ai_ready_joined2.csv`** vertically stacks both sources into one loop-style schema (glucose + basal/bolus/carbs columns). Rows from **`ai_ready`** keep wearable glucose but have empty insulin/carb fields; rows from **`loop`** carry pump insulin signals. Sequence IDs are prefixed `A-` (**ai_ready**) or `L-` (**loop**) to avoid collisions. Built by `scripts/loop_ai_ready/build_loop_ai_ready_joined2.py` with T1DM (**loop**) row mass balanced against combined non-T1DM (**ai_ready**) row mass.

### Training data per model

| Model | Training CSV | Composition |
|-------|--------------|-------------|
| **GluMind** | `ai_ready_plus_type1_v1_val_in_val_and_test.csv` | Predominantly **`ai_ready`** (wearable HR/steps) plus a **small type-1 supplement** (~686k rows, ~10% of training mass) — **not** the joined benchmark |
| **SugarOne** | `loop_ai_ready_joined2.csv` | Full join of **`ai_ready`** + **`loop`** (~50% / ~50% row mass) |

### `loop_ai_ready_joined2.csv` — distribution (12,090,991 rows)

**By source**

| Source | Rows | Share | Users | Sequences |
|--------|------|-------|-------|-----------|
| **ai_ready** | 6,027,765 | 49.9% | 2,232 | 3,201 |
| **loop** | 6,063,226 | 50.1% | 60 | 5,773 |

**By split** (same train/val/test fractions applied user-wise within each source)

| Split | Rows | Share |
|-------|------|-------|
| train | 8,390,218 | 69.3% |
| val | 1,840,551 | 15.2% |
| test | 1,860,222 | 15.4% |

**By study group** (T1DM rows are entirely from **`loop`**; other groups from **`ai_ready`**)

| Study group | Rows | Share | Source |
|-------------|------|-------|--------|
| T1DM | 6,063,226 | 50.1% | **loop** |
| healthy | 2,072,688 | 17.1% | **ai_ready** |
| oral_medication_and_or_non_insulin_injectable_medication_controlled | 1,818,059 | 15.0% | **ai_ready** |
| pre_diabetes_lifestyle_controlled | 1,515,841 | 12.5% | **ai_ready** |
| insulin_dependent | 621,177 | 5.1% | **ai_ready** |

**Test split used in this report** (1,860,222 rows): **loop** 937,989 (50.4%, all T1DM) · **ai_ready** 922,233 (49.6%, non-T1DM groups).

## Summary

SugarOne (basal/bolus/carb covariates, trained on the joined **`ai_ready` + `loop`** benchmark) slightly outperforms the best GluMind checkpoint (HR/steps covariates, trained on **`ai_ready`** plus a small type-1 supplement) when both are evaluated on the **`loop_ai_ready_joined2` test split**. The gap is modest (~0.33 mg/dL MAE) and should be interpreted in light of the cross-domain setup for GluMind (no HR/steps columns in the joined CSV schema).

A **covariate ablation** on the same SugarOne checkpoint (`--zero-cov`) shows that insulin and carb channels contribute a measurable but modest gain (~0.23 mg/dL MAE). With covariates zeroed, SugarOne (12.63) remains slightly ahead of cross-domain GluMind (12.73), suggesting part of the edge comes from in-domain training and architecture rather than covariates alone.

| Model | Training data | Covariates at inference | Windows | MAE ↓ | RMSE ↓ | MARD ↓ |
|-------|---------------|-------------------------|---------|-------|--------|--------|
| **SugarOne** (trial_0000_bcd3813f) | joined2 (ai_ready + loop) | basal, bolus, carbs | 1,667,437 | **12.40** | **19.03** | **9.91%** |
| **SugarOne** (trial_0000_bcd3813f, `--zero-cov`) | joined2 (ai_ready + loop) | glucose only (basal/bolus/carbs zeroed) | 1,667,437 | 12.63 | 19.47 | 9.98% |
| **GluMind** (best global run) | ai_ready + type-1 supplement | glucose only (HR/steps zero-filled) | 1,733,576 | 12.73 | 19.66 | 10.28% |

SugarOne with full covariates wins on all three metrics on this benchmark. Zeroing covariates degrades SugarOne but it still beats cross-domain GluMind on MAE.

## Model selection

### GluMind (best checkpoint)

Scanned **53** runs under `marked_runs/glumind/ai_ready_plus_type1` and selected the run with the lowest **validation MAE** from each run’s `val_metrics_overall.csv`.

**Selected run:** `marked_runs/glumind/ai_ready_plus_type1/glumind_global_h12_20260226_032703`

| Split | MAE | RMSE | MARD |
|-------|-----|------|------|
| Validation (in-domain, **ai_ready** + type-1 supplement) | **11.43** | 17.87 | 8.46% |
| Test (in-domain, **ai_ready** + type-1 supplement) | 11.70 | 18.46 | 8.52% |

**Architecture / training:** global mode, `input_steps=80`, `horizon=12`, `d_model=32`, `n_heads=4`, `n_blocks=3`, trained on `data/actual/with_complex_steps_processing/ai_ready_plus_type1_v1_val_in_val_and_test.csv` (6.84M rows: ~90% **`ai_ready`** cohorts, ~10% type-1 **`T1DM`** supplement — wearable schema with HR/steps; **not** `loop_ai_ready_joined2`).

The next-best runs were continual-learning steps (val MAE 11.69–11.82); the global run was clearly best on validation.

### SugarOne

**Selected run:** `runs/sugar_one_tune/production/trial_0000_bcd3813f`  
Best trial from production hyperparameter search (`leaderboard.csv`, combo hash `bcd3813f`).

| Split | MAE | RMSE | MARD |
|-------|-----|------|------|
| Validation (in-domain, joined2) | 12.69 | 19.73 | 9.64% |
| Test (in-domain, joined2, from training run) | 12.41 | 19.05 | 9.90% |

**Architecture / training:** global mode, `input_steps=128`, `horizon=12`, `d_model=32`, `n_heads=8`, `n_blocks=5`, trained on `data/input/loop_ai_ready_joined2.csv` (full **`ai_ready`** + **`loop`** join).

Re-evaluated test metrics (12.40 / 19.03 / 9.91%) match the saved training-run test metrics within rounding, confirming reproducibility.

## Cross-dataset evaluation setup

Both models were evaluated on the **same `loop_ai_ready_joined2` test split** using `evaluate-model`:

```bash
# GluMind (scalers fit on ai_ready train split)
uv run evaluate-model \
  --run-dir marked_runs/glumind/ai_ready_plus_type1/glumind_global_h12_20260226_032703 \
  --model-type glumind \
  --test-csv data/input/loop_ai_ready_joined2.csv \
  --batch-size 4096 \
  --output-json runs/comparison_loop/glumind_global.json

# SugarOne (scalers fit on joined2 train split)
uv run evaluate-model \
  --run-dir runs/sugar_one_tune/production/trial_0000_bcd3813f \
  --model-type sugar_one \
  --test-csv data/input/loop_ai_ready_joined2.csv \
  --train-csv data/input/loop_ai_ready_joined2.csv \
  --batch-size 256 \
  --output-json runs/comparison_loop/sugar_one_trial0.json

# SugarOne covariate ablation (basal/bolus/carbs zeroed after imputation)
uv run evaluate-model \
  --run-dir runs/sugar_one_tune/production/trial_0000_bcd3813f \
  --model-type sugar_one \
  --test-csv data/input/loop_ai_ready_joined2.csv \
  --train-csv data/input/loop_ai_ready_joined2.csv \
  --batch-size 256 \
  --zero-cov \
  --output-json runs/comparison_loop/sugar_one_trial0_zero_cov.json
```

**GluMind on joined2 (cross-domain):** the joined CSV uses loop-style columns only — no HR or Step Count. Missing wearable covariates are filled with **0.0** before imputation, so GluMind runs in effective glucose-only mode even on **`ai_ready`**-origin rows where it was trained with HR/steps.

**SugarOne on joined2 (in-domain):** same file and covariate schema as training. Insulin channels are populated on **`loop`**-source rows (~50% of test mass); **`ai_ready`**-source rows have empty basal/bolus/carbs (0-filled after imputation), as in training.

**SugarOne `--zero-cov` ablation:** basal, bolus, and carbs are forced to **0.0 after imputation** (so forward-filled basal rates on **`loop`** rows do not leak back). Scalers are still fit on the full training covariates. This isolates how much insulin/carb channels help at inference on the joined benchmark.

## Results analysis

### Overall test metrics (joined2 test split)

| Metric | SugarOne (full cov) | SugarOne (`--zero-cov`) | GluMind | Δ cov vs zero-cov | Δ GluMind − full cov |
|--------|----------------------|--------------------------|---------|-------------------|----------------------|
| MAE | 12.40 | 12.63 | 12.73 | **−0.23** | +0.33 |
| RMSE | 19.03 | 19.47 | 19.66 | **−0.44** | +0.63 |
| MARD | 9.91% | 9.98% | 10.28% | **−0.07 pp** | +0.37 pp |

SugarOne with full covariates is ~2.6% better on MAE relative to cross-domain GluMind on this test set.

### Covariate influence (SugarOne ablation)

Same checkpoint (`trial_0000_bcd3813f`), same test split, same 1,667,437 windows — only the covariate inputs differ:

| Condition | MAE | RMSE | MARD | vs full cov |
|-----------|-----|------|------|-------------|
| Full covariates (basal, bolus, carbs) | **12.40** | **19.03** | **9.91%** | — |
| `--zero-cov` (glucose only) | 12.63 | 19.47 | 9.98% | +0.23 MAE (+1.8%) |

**Takeaways:**

1. **Covariates help, but modestly on the full joined test split.** Insulin and carb channels improve MAE by ~0.23 mg/dL (~1.8%) averaged over **`loop`** + **`ai_ready`** test rows. RMSE gains are larger (+0.44), suggesting covariates especially reduce large errors on **`loop`**-source (T1DM) windows.

2. **Most of SugarOne's edge over GluMind is not from covariates alone.** With covariates zeroed, SugarOne (12.63) is still ~0.10 mg/dL better than cross-domain GluMind (12.73). That residual gap likely reflects in-domain training on joined2, longer input window (128 vs 80 steps), and architecture differences — not insulin signals alone at inference.

3. **Combined covariate + domain effect.** Full-cov SugarOne beats zero-cov SugarOne by 0.23 and beats GluMind by 0.33. Rough decomposition: ~70% of the total MAE gap vs GluMind (0.23/0.33) comes from using covariates; ~30% (0.10/0.33) from other factors. This is approximate — not a controlled architecture ablation.

4. **Fair comparison baseline.** `--zero-cov` makes SugarOne comparable to models evaluated without auxiliary channels on a given dataset, while keeping the same trained weights and scaler fitting. Use it when benchmarking against glucose-only models on datasets where covariates are absent or should be ignored.

### Interpretation

1. **Domain mismatch hurts GluMind.** Trained on **`ai_ready`** wearable data (HR, steps) and evaluated on joined2 (loop-style schema, no HR/steps), GluMind cannot use its auxiliary covariates on any test row. Its joined2 performance (MAE 12.73) is ~1.0 mg/dL worse than its in-domain **`ai_ready`** test MAE (11.70).

2. **SugarOne uses insulin signals where present.** Basal/bolus/carbs are populated on **`loop`**-source test rows and 0-filled on **`ai_ready`**-source rows — matching training. The `--zero-cov` ablation confirms insulin/carb channels contribute ~0.23 mg/dL MAE improvement (~1.8%) on the full joined test split, but in-domain training and architecture account for an additional ~0.10 mg/dL over cross-domain GluMind.

3. **Not a controlled architecture ablation.** The models differ in input window length (128 vs 80), block count (5 vs 3), attention heads (8 vs 4), and training data (**`ai_ready`**-centric vs joined **`ai_ready` + `loop`**). A fair covariate-only comparison would require training both architectures on the same joined split with matched hyperparameters. `--zero-cov` isolates the inference-time covariate effect for a single SugarOne checkpoint only.

4. **Window counts differ slightly** (1,667,437 vs 1,733,576) because GluMind uses `input_steps=80` and SugarOne uses `input_steps=128`; shorter input windows yield more sliding windows per series.

### GluMind by study group (saved training-run test metrics)

In-domain **`ai_ready` + type-1 supplement** test split from `glumind_global_h12_20260226_032703` (984,256 windows total).

| Study group | Windows | MAE | RMSE | MARD |
|-------------|---------|-----|------|------|
| Healthy | 220,829 | 9.57 | 14.79 | 8.15% |
| Pre-T2DM | 227,730 | 9.89 | 15.37 | 7.95% |
| Oral-T2DM | 230,379 | 12.35 | 19.23 | 8.45% |
| Insulin-T2DM | 198,811 | 13.58 | 21.15 | 8.22% |
| T1DM | 106,507 | 15.06 | 23.56 | 11.21% |

T1DM has the highest per-group MAE on GluMind's training data as well, though it is a smaller share of that test set (~11%) than in joined2 (~50% of rows from the **`loop`** source).

### SugarOne by study group (saved training-run test metrics)

In-domain **joined2** test split from `trial_0000_bcd3813f` (1,667,437 windows total).

| Study group | Windows | MAE | RMSE | MARD |
|-------------|---------|-----|------|------|
| Healthy | 213,994 | 9.62 | 14.81 | 8.24% |
| Pre-T2DM | 221,588 | 10.07 | 15.65 | 8.02% |
| Oral-T2DM | 223,016 | 12.72 | 19.71 | 8.54% |
| T1DM | 819,013 | 13.09 | 19.74 | 11.49% |
| Insulin-T2DM | 189,826 | 14.98 | 22.75 | 8.71% |

T1DM dominates the joined benchmark test split (~49% of test windows; all **`loop`**-source rows) and has the highest errors, consistent with greater glycemic variability in insulin-dependent cohorts.

## Conclusions

- On `loop_ai_ready_joined2.csv` test data, **SugarOne (trial_0000_bcd3813f) is the best model**, with MAE 12.40 vs 12.73 for the best GluMind global checkpoint.
- On bundled reviewer checkpoints evaluated on the shared Livia demo file (`test_data/livia_sugar_one_ready.csv`), **SugarOne (`test_model_sugar_one`) wins** — MAE **17.57** with insulin covariates vs 20.11 for `test_model_glumind` (HR/steps absent, 0-filled). With `--zero-cov`, SugarOne MAE is 18.89 (+1.32 vs full cov). Treat this as a sanity check only — not comparable to in-domain joined2 or **ai_ready** benchmarks.
- **Covariates matter but are not the whole story:** `--zero-cov` raises SugarOne MAE from 12.40 to 12.63 (+0.23 mg/dL). Zero-cov SugarOne still beats cross-domain GluMind (12.63 vs 12.73), so in-domain joined2 training and model design also contribute.
- GluMind’s cross-domain deployment on joined2 (no HR/steps columns) is a significant handicap; for **`loop`** pump traces or joined benchmarks with insulin covariates, SugarOne is the appropriate architecture.
- For **`ai_ready`** wearable data with HR and steps, the best GluMind run (`glumind_global_h12_20260226_032703`, val MAE 11.43) remains the recommended checkpoint.
- Use **`--zero-cov`** when evaluating a covariate-trained SugarOne checkpoint on datasets without insulin/carb signals, for apples-to-apples comparison with glucose-only baselines.

## `test_data` evaluation (`test_model_glumind` vs `test_model_sugar_one`)

The repo ships reviewer checkpoint bundles (`test_model_glumind/`, `test_model_sugar_one/`) and a shared Livia demo CSV (`test_data/livia_sugar_one_ready.csv`, 139,613 rows) in loop/SugarOne schema: glucose, basal rate, bolus insulin (no carb entries). **All three evaluations use this same file** so comparisons are on identical glucose/insulin traces. (`test_data/livia_glumind_ready.csv` remains available as a GluMind-only wearable-shaped export with HR/steps.)

GluMind and SugarOne (with and without `--zero-cov`) were re-evaluated with `evaluate-model` on **2026-06-06**.

This is a **sanity check / proof-of-life** run, not a headline benchmark: Livia is personal type-1 CGM + pump data, scalers are fit on the demo file (not the original private training CSVs referenced in bundled metadata), and neither model was trained on this subject.

### Setup

| Setting | Value |
|---------|-------|
| Test CSV | `test_data/livia_sugar_one_ready.csv` (all models) |
| Split | All rows (`--test-split ''`; `Recommended Split` column empty) |
| Scaler fitting | `--train-csv test_data/livia_sugar_one_ready.csv` (demo file only) |
| Covariates in file | basal (444 non-zero rows, 2.0–30.0 U/h), bolus (1,475 rows, 1.0–28.0 U); carbs absent (0-filled) |
| GluMind at inference | glucose + HR/steps (both missing → 0-filled, same as joined2 cross-domain eval) |
| SugarOne at inference | glucose + basal/bolus (full cov), or glucose only (`--zero-cov`) |

```bash
# GluMind (bundled reviewer checkpoint; HR/steps absent → 0-filled)
uv run evaluate-model \
  --run-dir test_model_glumind \
  --model-type glumind \
  --test-csv test_data/livia_sugar_one_ready.csv \
  --train-csv test_data/livia_sugar_one_ready.csv \
  --test-split '' \
  --batch-size 4096 \
  --output-json runs/comparison_test_data/glumind_test_model.json

# SugarOne (bundled reviewer checkpoint, full covariate path — no --zero-cov)
uv run evaluate-model \
  --run-dir test_model_sugar_one \
  --model-type sugar_one \
  --test-csv test_data/livia_sugar_one_ready.csv \
  --train-csv test_data/livia_sugar_one_ready.csv \
  --test-split '' \
  --batch-size 256 \
  --output-json runs/comparison_test_data/sugar_one_test_model_full_cov.json

# SugarOne (bundled reviewer checkpoint, glucose-only ablation via --zero-cov)
uv run evaluate-model \
  --run-dir test_model_sugar_one \
  --model-type sugar_one \
  --test-csv test_data/livia_sugar_one_ready.csv \
  --train-csv test_data/livia_sugar_one_ready.csv \
  --zero-cov \
  --test-split '' \
  --batch-size 256 \
  --output-json runs/comparison_test_data/sugar_one_test_model.json
```

### Results (all rows, demo scalers)

| Model | Checkpoint | Covariates at inference | Windows | MAE ↓ | RMSE ↓ | MARD ↓ |
|-------|------------|---------------------------|---------|-------|--------|--------|
| **SugarOne** (`test_model_sugar_one`, full cov) | `test_model_sugar_one/best_model.pt` | glucose + basal/bolus (carbs 0-filled) | 127,659 | **17.57** | **26.27** | **14.24%** |
| **SugarOne** (`test_model_sugar_one`, `--zero-cov`) | `test_model_sugar_one/best_model.pt` | glucose only (basal/bolus/carbs zeroed) | 127,659 | 18.89 | 27.68 | 14.99% |
| **GluMind** (`test_model_glumind`) | `test_model_glumind/best_model.pt` | glucose + HR/steps (missing → 0-filled) | 131,787 | 20.11 | 28.87 | 17.64% |

All rows from `test_data/livia_sugar_one_ready.csv`.

SugarOne with insulin covariates leads on all three metrics. The MAE gap vs GluMind is **2.54 mg/dL** (~12.6% relative). Zeroing covariates (`--zero-cov`) raises SugarOne MAE by **1.32 mg/dL** (~7.0%), showing basal/bolus channels help on this personal pump trace even without carb data.

| Metric | SugarOne (full cov) | SugarOne (`--zero-cov`) | GluMind | Δ zero-cov − full cov | Δ GluMind − full cov |
|--------|----------------------|--------------------------|---------|----------------------|----------------------|
| MAE | 17.57 | 18.89 | 20.11 | **+1.32** | **+2.54** |
| RMSE | 26.27 | 27.68 | 28.87 | +1.41 | +2.60 |
| MARD | 14.24% | 14.99% | 17.64% | +0.75 pp | +3.40 pp |

### Covariate influence on Livia (SugarOne ablation)

Same checkpoint, same `livia_sugar_one_ready.csv`, same 127,659 windows — insulin channels present vs `--zero-cov`:

| Condition | MAE | RMSE | MARD | vs full cov |
|-----------|-----|------|------|-------------|
| Full covariates (basal + bolus; carbs 0-filled) | **17.57** | **26.27** | **14.24%** | — |
| `--zero-cov` (glucose only) | 18.89 | 27.68 | 14.99% | +1.32 MAE (+7.5%) |

The Livia covariate gain (+1.32 MAE) is larger than on the joined2 benchmark (+0.23 MAE), likely because this subject is insulin-dependent (T1) and pump bolus/basal events are informative for personal forecasting.

### Context vs in-domain bundled metrics

Saved training-run test metrics from the bundled folders (different datasets, official splits):

| Model | Training data | In-domain test MAE | Livia demo MAE (full cov) | Livia demo MAE (`--zero-cov`) |
|-------|---------------|--------------------|---------------------------|-------------------------------|
| GluMind (`test_model_glumind`) | ai_ready_plus_type1 | 11.70 | 20.11 | — |
| SugarOne (`test_model_sugar_one`) | loop_ai_ready_joined2 | 12.41 | **17.57** | 18.89 |

Both models degrade on out-of-distribution personal data, as expected. With insulin covariates available, SugarOne improves substantially over its glucose-only ablation (17.57 vs 18.89) and over GluMind (20.11).

**Caveats:**

1. Window counts differ (127,659 vs 131,787) because SugarOne uses `input_steps=128` vs GluMind `input_steps=80` on the same underlying rows.
2. Demo scalers are fit on all Livia rows; production metrics use train-split scalers from full training CSVs.
3. GluMind cannot use insulin columns from this CSV (architecture mismatch); it runs glucose-only in practice via 0-filled HR/steps — analogous to its joined2 cross-domain setup.
4. `livia_sugar_one_ready.csv` has no carbohydrate entries; carbs are 0-filled throughout (per data availability).

## Artifacts

| File | Description |
|------|-------------|
| `runs/comparison_loop/glumind_global.json` | GluMind on joined2 test split (cross-domain) |
| `runs/comparison_loop/sugar_one_trial0.json` | SugarOne on joined2 test split (full covariates) |
| `runs/comparison_loop/sugar_one_trial0_zero_cov.json` | SugarOne on joined2 test split (`--zero-cov` ablation) |
| `runs/comparison_test_data/glumind_test_model.json` | GluMind on `test_data/livia_sugar_one_ready.csv` |
| `runs/comparison_test_data/sugar_one_test_model_full_cov.json` | SugarOne on `test_data/livia_sugar_one_ready.csv` (no `--zero-cov`) |
| `runs/comparison_test_data/sugar_one_test_model.json` | SugarOne on `test_data/livia_sugar_one_ready.csv` (`--zero-cov`) |
| `docs/reports/milestone7_smoke_livia.json` | Earlier SugarOne smoke run (`--zero-cov` on old glucose-only file; superseded) |
| `scripts/sugar_one/evaluate_model.py` | Unified evaluation CLI with `--zero-cov` and progress logging |

## Tooling note

`evaluate-model` supports `--zero-cov` to zero all non-glucose covariates after imputation (for fair comparison with glucose-only baselines). It also logs inference progress roughly every 10 seconds (configurable via `--log-interval`), reporting batch progress, approximate windows processed, elapsed time, and ETA based on observed throughput.
