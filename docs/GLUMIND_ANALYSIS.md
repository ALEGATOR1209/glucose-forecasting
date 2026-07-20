# GluMind Model Performance & Logic Report

This report summarizes the GluMind model's architecture, training modes, and performance metrics across different study groups and datasets.

> **Naming:** This document is about wearable **GluMind** (BGI: “GluMind (Ours)”). The insulin/carb model is **SugarOne** (formerly GluMindIC). See [MILESTONES.md](MILESTONES.md). Training CSVs are produced by [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing) — [DATA.md](DATA.md).

## 1. Project Overview

The `glucose-forecasting` project predicts blood glucose from multimodal CGM data. This report covers **GluMind** (Glucose, Heart Rate, Step Count) on **AI-READI** and combined cohorts.

### Key Concepts:
- **Model**: `GluMind` - A Parallel-Attention Transformer architecture.
- **Horizon**: 12 steps (60 minutes at 5-min frequency).
- **Modes**:
    - `Global`: Standard training on the entire dataset at once.
    - `Continual`: Sequential training cohort-by-cohort using **Learning without Forgetting (LwF)** to prevent catastrophic forgetting.
- **Split Schemes**:
    - `classic`: Standard Train/Val/Test split.
    - `trainval_test_as_val`: Merges Train+Val for training and uses Test as Validation (primarily for hyperparameter tuning).

## 2. Best Performance Summary (GluMind)

The following tables highlight the top-performing GluMind runs as of February 2026.

| Scenario | Best Run Name | Split Scheme | MAE | RMSE | MARD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AI Ready Only** | `glumind_global_h12_20260223_195526` | `trainval_test_as_val` | **11.3336** | 17.7312 | 8.25% |
| **AI Ready + Type 1** | `glumind_global_h12_20260226_032703` | `classic` | **11.6987** | 18.4568 | 8.52% |
| **Type 1 Only** | `glumind_global_h12_20260225_120905` | `classic` | **14.5090** | 23.0004 | 10.99% |

*Note: Global mode consistently outperforms Continual mode, though Continual mode is competitive and preserves knowledge across cohorts.*

## 3. Metrics Distribution by Study Group

Errors increase as we move from healthy individuals to those with advanced diabetes or Type 1 diabetes. This "error gradient" is consistent across all models and modes.

### Best Global Run (AI Ready + Type 1) - `glumind_global_h12_20260226_032703`

| Study Group | MAE | RMSE | MARD |
| :--- | :--- | :--- | :--- |
| **Healthy** | 9.57 | 14.79 | 8.15% |
| **Pre-T2DM** | 9.89 | 15.37 | 7.95% |
| **Oral-T2DM** | 12.35 | 19.23 | 8.45% |
| **Insulin-T2DM** | 13.58 | 21.15 | 8.22% |
| **T1DM** | 15.06 | 23.56 | 11.21% |

## 4. Global vs. Continual Mode

Continual mode training follows the order: `Healthy` → `Pre-T2DM` → `Oral-T2DM` → `Insulin-T2DM`. 

| Metric | Best Global (Classic) | Best Continual (Classic) | Delta (Cont - Global) |
| :--- | :--- | :--- | :--- |
| **Overall MAE** | 11.3357 | 11.4803 | +0.1446 |
| **Overall RMSE** | 17.8744 | 17.9541 | +0.0797 |

While Continual mode has a slightly higher error, it demonstrates the feasibility of sequential learning in medical contexts where data may arrive in stages.

## 5. Logical System of Numbers

To navigate the metrics:
1.  **Baseline**: Start with the `Global` mode in `classic` split. This represents the upper bound of performance.
2.  **Cohort Sensitivity**: Always check the `by_group` breakdown. `Healthy` and `Pre-T2DM` are easier to predict; `Insulin-T2DM` and `T1DM` are the hardest due to higher variability.
3.  **Stability**: Compare `val_as_test` vs `test` metrics. If they are close, the model generalizes well.
4.  **LwF Lambda**: In Continual mode, `lwf_lambda=0.3` is generally the "sweet spot" for balancing new learning and old knowledge retention.

## 6. Where to find Checkpoints

Checkpoints are located in `marked_runs/glumind/<category>/<run_name>/checkpoints/`.
- `best_model.pt`: Weights for the best validation loss.
- `last_model.pt`: Weights at the end of training.
- `checkpoint.pt`: Full state (optimizer, scheduler) for resuming training.

---
*Report generated for onboarding - Mar 4, 2026*
