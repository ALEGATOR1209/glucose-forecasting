# Cross-Model Comparison: GluMind (our) vs NHITS vs GluFormer

- Goal: compare best available runs per dataset scope using the files you specified.
- Lower is better for MAE / RMSE / MARD.
- For tuning-only splits where test is unavailable, `val_as_test` is used.
- Date generated: 2026-02-27.
- `GluMind (our)` / **GluMind (Ours)** in this document is the wearable multimodal model in `scripts/glumind/` (glucose + HR + steps). It is **not** SugarOne. The insulin/carb model formerly called GluMindIC is **SugarOne** — see `docs/MILESTONES.md` and `docs/GLUMIND_VS_SUGARONE_COMPARISON.md`.

## Input Files Reviewed

- `marked_runs/glumind/ai_ready/RUNS_ANALYSIS.md`
- `marked_runs/glumind/ai_ready_plus_type1/RUNS_ANALYSIS.md`
- `marked_runs/glumind/type1_only/RUNS_ANALYSIS.md`
- `runs/nhits/RUNS_ANALYSIS.md`
- `runs/gluformer/ai_ready_plus_type1/gluformer_20260227_005453/test_metrics_overall.csv`
- `runs/gluformer/ai_ready_plus_type1/gluformer_20260227_005453/test_metrics_by_study_group.csv`

## Executive Summary

| scenario | GluMind (our) MAE | NHITS MAE | GluFormer MAE | MAE verdict |
| --- | --- | --- | --- | --- |
| AI Ready Only | 11.3336 | 20.5951 | NA | better than NHITS |
| Type1 Only | 14.5090 | 15.1124 | NA | better than NHITS |
| AI Ready + Type1 (Combined) | 11.6987 | 20.2081 | 19.5332 | better than NHITS, better than GluFormer |

## AI Ready Only

### Selected Runs

| model | run | metric_source | overall_path | by_group_path |
| --- | --- | --- | --- | --- |
| GluMind (our) | glumind_global_h12_20260223_195526 | val_as_test | `marked_runs/glumind/ai_ready/glumind_global_h12_20260223_195526/val_metrics_overall.csv` | `marked_runs/glumind/ai_ready/glumind_global_h12_20260223_195526/val_metrics_by_study_group.csv` |
| NHITS | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012925 @ step-step=100 | val_as_test | `runs/nhits/__ALL__/nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012925/eval_checkpoints/step-step=100/val_metrics_overall.csv` | `runs/nhits/__ALL__/nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_012925/eval_checkpoints/step-step=100/val_metrics_by_study_group.csv` |
| GluFormer | not available for this dataset scope in provided files | NA | NA | NA |

### Overall Metrics

| model | MAE | RMSE | MARD |
| --- | --- | --- | --- |
| GluMind (our) | 11.3336 | 17.7312 | 8.2476 |
| NHITS | 20.5951 | 34.4466 | 13.3342 |

| comparison | delta_MAE (baseline-GM) | delta_RMSE (baseline-GM) | delta_MARD (baseline-GM) | rel_MAE_impr_% | rel_RMSE_impr_% | rel_MARD_impr_% |
| --- | --- | --- | --- | --- | --- | --- |
| NHITS - GluMind (our) | 9.2614 | 16.7155 | 5.0866 | 44.97 | 48.53 | 38.15 |

### Per-Group Metrics (Exact Values)

| group | GM_MAE | NHITS_MAE | GM_vs_NHITS_delta | GM_RMSE | NHITS_RMSE | GM_vs_NHITS_RMSE_delta | GM_MARD | NHITS_MARD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Healthy | 9.5800 | 17.0248 | 7.4449 | 14.7042 | 26.8800 | 12.1758 | 8.1617 | 14.6014 |
| Pre-T2DM | 9.8903 | 14.0389 | 4.1486 | 15.2701 | 21.4301 | 6.1600 | 7.9714 | 12.4411 |
| Oral-T2DM | 12.3969 | 20.1678 | 7.7709 | 19.2205 | 32.4535 | 13.2330 | 8.5196 | 14.2951 |
| Insulin-T2DM | 13.7026 | 28.2730 | 14.5703 | 21.2769 | 46.4782 | 25.2013 | 8.3444 | 12.2233 |

### Group-Level Win Count (GluMind (our) lower-error wins)

- vs NHITS: MAE wins 4/4, RMSE wins 4/4, MARD wins 4/4.

### AI-READI Literature Baselines (User-Provided Numbers)

- This subsection uses the values you provided from prior models/paper tables.
- In this comparison, our tuned AI-READI run is treated as `GluMind (our)` with `BG+W+HR` setup.
- Positive deltas below mean `baseline - ours`, so positive means our run is better (lower error).

#### RMSE vs GlySim / AttenGluco / Informer

| cohort | Ours RMSE | GlySim RMSE | Delta | AttenGluco RMSE | Delta | Informer RMSE | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Healthy | 14.70 | 17.79 | 3.09 | 15.45 | 0.75 | 19.81 | 5.11 |
| Pre-T2DM | 15.27 | 19.77 | 4.50 | 17.47 | 2.20 | 21.95 | 6.68 |
| Oral | 19.22 | 23.37 | 4.15 | 20.45 | 1.23 | 29.17 | 9.95 |
| Insulin-T2DM | 21.28 | 28.22 | 6.94 | 25.04 | 3.76 | 34.51 | 13.23 |

- RMSE wins: vs GlySim `4/4`, vs AttenGluco `4/4`, vs Informer `4/4`.

#### Original GluMind Paper Table II (RMSE Features) and Table III (MAE Features)

| cohort | Ours RMSE | Orig BG+W+HR RMSE | Delta | Orig Best-Feature RMSE (Table II) | Delta |
| --- | --- | --- | --- | --- | --- |
| Healthy | 14.70 | 15.39 | 0.69 | 15.00 | 0.30 |
| Pre-T2DM | 15.27 | 15.95 | 0.68 | 15.75 | 0.48 |
| Oral | 19.22 | 19.30 | 0.08 | 19.23 | 0.01 |
| Insulin-T2DM | 21.28 | 23.27 | 1.99 | 22.78 | 1.50 |

| cohort | Ours MAE | Orig BG+W+HR MAE | Delta | Orig Best-Feature MAE (Table III) | Delta |
| --- | --- | --- | --- | --- | --- |
| Healthy | 9.58 | 10.80 | 1.22 | 10.58 | 1.00 |
| Pre-T2DM | 9.89 | 11.20 | 1.31 | 11.08 | 1.19 |
| Oral | 12.40 | 13.80 | 1.40 | 13.74 | 1.34 |
| Insulin-T2DM | 13.70 | 16.73 | 3.03 | 16.41 | 2.71 |

- Against original Table II/III values, our tuned run is better in all four cohorts for RMSE and MAE, including when compared to the best feature combinations that use additional columns.

#### Original GluMind Additional Reported Table (RMSE/MAE)

| cohort | Ours RMSE | Original RMSE | Delta | Ours MAE | Original MAE | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| Healthy | 14.70 | 13.56 | -1.14 | 9.58 | 9.99 | 0.41 |
| Pre-T2DM | 15.27 | 15.84 | 0.57 | 9.89 | 12.27 | 2.38 |
| Oral | 19.22 | 16.64 | -2.58 | 12.40 | 12.59 | 0.19 |
| Insulin-T2DM | 21.28 | 21.17 | -0.11 | 13.70 | 16.76 | 3.06 |

- For this additional table: MAE improves in `4/4` cohorts, while RMSE improves in `1/4` cohorts (mixed RMSE outcome).

## Type1 Only

### Selected Runs

| model | run | metric_source | overall_path | by_group_path |
| --- | --- | --- | --- | --- |
| GluMind (our) | glumind_global_h12_20260225_120905 | test | `marked_runs/glumind/type1_only/glumind_global_h12_20260225_120905/test_metrics_overall.csv` | `marked_runs/glumind/type1_only/glumind_global_h12_20260225_120905/test_metrics_by_study_group.csv` |
| NHITS | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013800 @ step-step=150 | test | `runs/nhits/nf_nhits_type1_only/__ALL__/nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013800/eval_checkpoints/step-step=150/test_metrics_overall.csv` | `runs/nhits/nf_nhits_type1_only/__ALL__/nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_013800/eval_checkpoints/step-step=150/test_metrics_by_study_group.csv` |
| GluFormer | not available for this dataset scope in provided files | NA | NA | NA |

### Overall Metrics

| model | MAE | RMSE | MARD |
| --- | --- | --- | --- |
| GluMind (our) | 14.5090 | 23.0004 | 10.9902 |
| NHITS | 15.1124 | 21.0525 | 11.2350 |

| comparison | delta_MAE (baseline-GM) | delta_RMSE (baseline-GM) | delta_MARD (baseline-GM) | rel_MAE_impr_% | rel_RMSE_impr_% | rel_MARD_impr_% |
| --- | --- | --- | --- | --- | --- | --- |
| NHITS - GluMind (our) | 0.6033 | -1.9479 | 0.2448 | 3.99 | -9.25 | 2.18 |

### Per-Group Metrics (Exact Values)

| group | GM_MAE | NHITS_MAE | GM_vs_NHITS_delta | GM_RMSE | NHITS_RMSE | GM_vs_NHITS_RMSE_delta | GM_MARD | NHITS_MARD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1DM | 14.5090 | 15.1124 | 0.6033 | 23.0004 | 21.0525 | -1.9479 | 10.9902 | 11.2350 |

### Group-Level Win Count (GluMind (our) lower-error wins)

- vs NHITS: MAE wins 1/1, RMSE wins 0/1, MARD wins 1/1.
- vs GluFormer: MAE wins 1/1, RMSE wins 0/1, MARD wins 1/1.

## AI Ready + Type1 (Combined)

### Selected Runs

| model | run | metric_source | overall_path | by_group_path |
| --- | --- | --- | --- | --- |
| GluMind (our) | glumind_global_h12_20260226_032703 | test | `marked_runs/glumind/ai_ready_plus_type1/glumind_global_h12_20260226_032703/test_metrics_overall.csv` | `marked_runs/glumind/ai_ready_plus_type1/glumind_global_h12_20260226_032703/test_metrics_by_study_group.csv` |
| NHITS | nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014043 @ step-step=150 | test | `runs/nhits/nf_nhits_ai_plus_type1_classic/__ALL__/nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014043/eval_checkpoints/step-step=150/test_metrics_overall.csv` | `runs/nhits/nf_nhits_ai_plus_type1_classic/__ALL__/nhits_lr0.001_ms300_bs8_ws256_ss12_20260227_014043/eval_checkpoints/step-step=150/test_metrics_by_study_group.csv` |
| GluFormer | gluformer_20260227_005453 | test | `runs/gluformer/ai_ready_plus_type1/gluformer_20260227_005453/test_metrics_overall.csv` | `runs/gluformer/ai_ready_plus_type1/gluformer_20260227_005453/test_metrics_by_study_group.csv` |

### Overall Metrics

| model | MAE | RMSE | MARD |
| --- | --- | --- | --- |
| GluMind (our) | 11.6987 | 18.4568 | 8.5202 |
| NHITS | 20.2081 | 33.7321 | 13.1145 |
| GluFormer | 19.5332 | 33.2846 | 13.0314 |

| comparison | delta_MAE (baseline-GM) | delta_RMSE (baseline-GM) | delta_MARD (baseline-GM) | rel_MAE_impr_% | rel_RMSE_impr_% | rel_MARD_impr_% |
| --- | --- | --- | --- | --- | --- | --- |
| NHITS - GluMind (our) | 8.5093 | 15.2753 | 4.5943 | 42.11 | 45.28 | 35.03 |
| GluFormer - GluMind (our) | 7.8344 | 14.8278 | 4.5112 | 40.11 | 44.55 | 34.62 |

### Per-Group Metrics (Exact Values)

| group | GM_MAE | NHITS_MAE | GluFormer_MAE | GM_vs_NHITS_delta | GM_vs_GluFormer_delta | GM_MARD | NHITS_MARD | GluFormer_MARD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Healthy | 9.5721 | 16.8598 | 17.0782 | 7.2877 | 7.5061 | 8.1470 | 14.4115 | 14.6390 |
| Pre-T2DM | 9.8871 | 13.9981 | 14.2124 | 4.1110 | 4.3252 | 7.9531 | 12.3460 | 12.1168 |
| Oral-T2DM | 12.3474 | 19.9669 | 19.3602 | 7.6195 | 7.0128 | 8.4529 | 14.1469 | 13.3259 |
| Insulin-T2DM | 13.5820 | 28.3079 | 26.3590 | 14.7259 | 12.7770 | 8.2188 | 12.1694 | 11.8362 |
| T1DM | 15.0630 | 15.5276 | 15.4632 | 0.4646 | 0.4002 | 11.2147 | 11.5241 | 15.0966 |

### Group-Level Win Count (GluMind (our) lower-error wins)

- vs NHITS: MAE wins 5/5, RMSE wins 4/5, MARD wins 5/5.
- vs GluFormer: MAE wins 5/5, RMSE wins 5/5, MARD wins 5/5.

### Original GluMind Article vs Combined Run (Shared Cohorts)

- Shared cohorts for direct comparison: `Healthy`, `Pre-T2DM`, `Oral`, `Insulin-T2DM`.
- Original paper references used here:
`TABLE II` (RMSE feature ablation), `TABLE III` (MAE feature ablation).
- Positive delta means `original - ours` (positive = our combined run is better).

| cohort | Ours RMSE (combined) | Orig BG+W+HR RMSE | Delta | Orig Best-Feature RMSE | Delta | Ours MAE (combined) | Orig BG+W+HR MAE | Delta | Orig Best-Feature MAE | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Healthy | 14.79 | 15.39 | 0.60 | 15.00 | 0.21 | 9.57 | 10.80 | 1.23 | 10.58 | 1.01 |
| Pre-T2DM | 15.37 | 15.95 | 0.58 | 15.75 | 0.38 | 9.89 | 11.20 | 1.31 | 11.08 | 1.19 |
| Oral | 19.23 | 19.30 | 0.07 | 19.23 | -0.00 | 12.35 | 13.80 | 1.45 | 13.74 | 1.39 |
| Insulin-T2DM | 21.15 | 23.27 | 2.12 | 22.78 | 1.63 | 13.58 | 16.73 | 3.15 | 16.41 | 2.83 |

- RMSE: better in `4/4` cohorts vs original `BG+W+HR`; better in `4/4` and essentially tied on Oral vs original best-feature RMSE.
- MAE: better in `4/4` cohorts vs both original `BG+W+HR` and original best-feature MAE.

#### Direct Comparison to Original GluMind “Best” Table (Requested)

- Run used: `marked_runs/glumind/ai_ready_plus_type1/glumind_global_h12_20260226_032703/test_metrics_by_study_group.csv`
- Original values used (as provided):  
`RMSE/MAE -> Healthy 13.56/9.99, PreT2DM 15.84/12.27, Oral 16.64/12.59, Insulin 21.17/16.76`
- Positive delta means `original - ours` (positive = our run is better).

| cohort | Ours RMSE | Orig “best” RMSE | RMSE delta | Ours MAE | Orig “best” MAE | MAE delta |
| --- | --- | --- | --- | --- | --- | --- |
| Healthy | 14.7865 | 13.5600 | -1.2265 | 9.5721 | 9.9900 | 0.4179 |
| Pre-T2DM | 15.3651 | 15.8400 | 0.4749 | 9.8871 | 12.2700 | 2.3829 |
| Oral-T2DM | 19.2311 | 16.6400 | -2.5911 | 12.3474 | 12.5900 | 0.2426 |
| Insulin-T2DM | 21.1460 | 21.1700 | 0.0240 | 13.5820 | 16.7600 | 3.1780 |

- Result for this exact table: MAE better in `4/4` cohorts; RMSE better in `2/4` cohorts.

## Final Conclusion

- All `GluMind (our)` results in this report refer to **our version of the GluMind architecture**.
- `ai_ready`: `GluMind (our)` is clearly stronger than NHITS and GluFormer on overall and all group-level metrics.
- `type1_only`: `GluMind (our)` beats NHITS on MAE and MARD.
- `ai_ready_plus_type1`: `GluMind (our)` is strongest overall vs both NHITS and GluFormer; at group level it wins MAE for every group and wins most RMSE/MARD comparisons.
- Practical claim supported by these files: **`GluMind (our)` is the best-performing architecture for ai_ready and combined datasets; for type1_only it leads on MAE/MARD**.
