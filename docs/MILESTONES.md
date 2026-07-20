# BGI milestone notes (M06–M07)

Condensed from the April / June 2026 BGI submissions so the facts live in-repo. Full narrative reports:

- Architecture selection (M06): [CROSS_MODEL_COMPARISON_REPORT.md](CROSS_MODEL_COMPARISON_REPORT.md), [CROSS_MODEL_COMPARISON.md](../CROSS_MODEL_COMPARISON.md)
- Domain covariates (M07): [GLUMIND_VS_SUGARONE_COMPARISON.md](GLUMIND_VS_SUGARONE_COMPARISON.md), [T1DM_COVARIATE_ABLATION_REPORT.md](T1DM_COVARIATE_ABLATION_REPORT.md)

## Naming (read this first)

Older BGI text used **“GluMind (Ours)”** for *our* reimplementation of the multimodal architecture, and later **“GluMindIC”** for the insulin/carb adaptation.

| BGI / older name | Current name in this repo | Code |
|------------------|---------------------------|------|
| GluMind (Ours) | **GluMind** | `scripts/glumind/` — glucose + HR + steps |
| GluMind Uni | **GluMind-Uni** | `scripts/glumind_uni/` — glucose only |
| GluMindIC | **SugarOne** | `scripts/sugar_one/` — glucose + basal + bolus + carbs |

**SugarOne** is the current primary model for Loop / pump-style forecasting. GluMind remains the wearable (AI-READI HR/steps) baseline and the architectural parent of SugarOne.

## Milestone 06 — architecture selection (April 2026)

**Goal:** Evaluate candidate architectures and pick a primary baseline for further work.

**Candidates:** GluMind (Ours), GluMind-Uni, NHITS, GluFormer.

**Combined (AI-READI + Type1) overall metrics** (lower is better):

| Model | MAE (mg/dL) | RMSE | MARD (%) |
|-------|-------------|------|----------|
| GluMind (Ours) | **11.69** | **18.45** | **8.52** |
| GluFormer | 19.53 | 33.28 | 13.03 |
| NHITS | 20.20 | 33.73 | 13.11 |

**Per-group MAE (combined):** GluMind best in every group (Healthy 9.57, Pre-T2DM 9.89, Oral-T2DM 12.35, Insulin-T2DM 13.58, T1DM 15.06).

**Selection rationale (then):** best MAE across AI-READI / Type1 / Combined; beats NHITS and GluFormer; stable on high-variance T1DM (MAE 14.51, MARD 10.99% on Type1-only scope).

**Reviewer access:** open repo + bundled weights + [How_to_run_checkpoint.md](../How_to_run_checkpoint.md). Pretrained wearable checkpoint folder: `test_model_glumind/` (also mirrored as `test_model/` in some checkouts).

## Milestone 07 — domain variables → SugarOne (June 2026)

**Goal:** Adapt the architecture for basal insulin, bolus insulin, and carbohydrates; quantify impact.

**Dataset:** `loop_ai_ready_joined2.csv` (~12M rows) — AI-READI wearable cohort joined with Loop pump (T1DM) data. Built via preprocessing ([DATA.md](DATA.md)) + `scripts/loop_ai_ready/`.

**Model:** SugarOne (then called GluMindIC): longer window (`input_steps=128` vs 80), more blocks (`n_blocks=5` vs 3), learnable mixing over three auxiliary channels.

**Shared joined test split (~1.86M windows):**

| Model / condition | MAE | RMSE | MARD |
|-------------------|-----|------|------|
| GluMind (cross-domain on joined schema) | 12.73 | 19.66 | 10.28% |
| SugarOne (`--zero-cov`) | 12.63 | 19.47 | 9.98% |
| SugarOne (full covariates) | **12.40** | **19.03** | **9.91%** |

**T1DM loop-only ablation** (see [T1DM_COVARIATE_ABLATION_REPORT.md](T1DM_COVARIATE_ABLATION_REPORT.md)): full covariates beat glucose-only by ~0.47 MAE (~3.6%); bolus is the dominant channel.

**Livia personal demo:** insulin channels improve MAE by ~1.32 (~7%) vs `--zero-cov` — sanity check only, not a headline benchmark.

**Reproduce:** `uv run evaluate-model` with `--zero-cov` / `--include-cov` / `--exclude-cov`. Bundled SugarOne weights: `test_model_sugar_one/`.

## Success criteria checklist

| Criterion | Status |
|-----------|--------|
| ≥2 models implemented and compared (M06) | GluMind, Uni, NHITS, GluFormer |
| Quantitative MAE/RMSE/MARD (M06/M07) | Documented in comparison reports |
| Domain variables added (M07) | SugarOne basal/bolus/carbs |
| Ablation of variable impact (M07) | T1DM report + `--zero-cov` |
| Code accessible for review | This repo + checkpoint smoke tests |
