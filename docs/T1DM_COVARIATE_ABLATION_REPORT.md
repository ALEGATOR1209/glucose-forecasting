# T1DM Covariate Ablation Report — SugarOne on Loop Test Data

**Date:** 2026-06-13  
**Evaluation tool:** `uv run evaluate-model` (`scripts/sugar_one/evaluate_model.py`)  
**Model checkpoint:** `runs/sugar_one_tune/production/trial_0000_bcd3813f` (`best_model.pt`)  
**Test dataset:** `ablation_test.csv` (historically under `data/loop_and_ai_ready/`; see [DATA.md](DATA.md) for `data/input/` layout)  
**Scaler fitting:** `loop_ai_ready_joined2.csv` (train split)

---

## Executive summary

Eight inference configurations were run on a **T1DM-only, covariate-complete loop test subset** (819,013 sliding windows). Using **all three insulin/carb covariates** yields the best MAE (**13.08 mg/dL**). Removing all covariates (`--zero-cov`) degrades MAE by **+0.47 mg/dL (+3.6%)**.

**Bolus insulin** is the strongest single channel: `--include-cov bolus` alone reaches MAE **13.22** (within **0.14 mg/dL** of full covariates), and `--exclude-cov bolus` causes the largest single-channel drop (**+0.32 mg/dL**). **Basal rate** contributes the least marginally (**+0.03 mg/dL** when excluded). **Carbohydrates** sit in between for both single-channel and exclusion tests.

---

## Dataset statistics

`ablation_test.csv` was extracted from `loop_ai_ready_joined2.csv` with:

- `Recommended Split == test`
- `Study Group == T1DM`
- Quality filter: every sequence for each user has basal, bolus, and carb values present on at least one row

| Metric | Value |
|--------|------:|
| Rows | 937,989 |
| Users | 9 |
| Sequences | 861 |
| Evaluation windows (128 input + 12 horizon) | 819,013 |
| Study groups | T1DM only |
| Time span | 2017-11-26 → 2020-01-11 |

### Glucose distribution

| Statistic | mg/dL |
|-----------|------:|
| Min | 39.0 |
| Median | 114.0 |
| Mean | 124.7 |
| Max | 401.0 |

### Event types

| Event Type | Rows | Share |
|------------|-----:|------:|
| EGV | 763,481 | 81.4% |
| Interpolated | 174,436 | 18.6% |
| BGM | 72 | <0.1% |

### Covariate fill rates (non-empty raw values)

Covariates are sparse at the row level (pump events vs continuous EGV), but imputation (basal forward-fill; bolus/carbs event-only) propagates signals into sliding windows.

| Covariate | Filled rows | Share of rows |
|-----------|------------:|--------------:|
| Basal rate | 345,084 | 36.8% |
| Bolus insulin | 37,660 | 4.0% |
| Carbohydrates | 21,559 | 2.3% |
| All three on same row | 2,705 | 0.3% |

### Benchmark alignment

The full-covariate run on this file (MAE **13.08**, 819,013 windows) matches the saved **T1DM study-group** metrics from the same checkpoint on the full joined2 test split (MAE **13.09**, 819,013 windows in `docs/GLUMIND_VS_SUGARONE_COMPARISON.md`). This confirms `ablation_test.csv` is the loop/T1DM portion of the joined benchmark test set.

---

## Experimental setup

| Setting | Value |
|---------|-------|
| Architecture | SugarOne — `input_steps=128`, `horizon=12`, `d_model=32`, `n_heads=8`, `n_blocks=5` |
| Unique ID | `sequence_id` |
| Batch size | 256 |
| Device | CUDA (when available) |
| Covariate ablation | Applied **after imputation** (zeroed channels do not leak forward-filled basal) |
| Scalers | Fit on joined2 **train** rows only (unchanged across all 8 runs) |
| Split filter | `--test-split ''` (CSV is pre-filtered to test) |

### Configurations (8 runs)

| # | Label | CLI flags | Active covariates |
|---|-------|-----------|-------------------|
| 1 | **All** | *(default)* | basal, bolus, carbs |
| 2 | **None** | `--zero-cov` | — |
| 3 | Include basal | `--include-cov basal` | basal |
| 4 | Include bolus | `--include-cov bolus` | bolus |
| 5 | Include carbs | `--include-cov carbs` | carbs |
| 6 | Exclude basal | `--exclude-cov basal` | bolus, carbs |
| 7 | Exclude bolus | `--exclude-cov bolus` | basal, carbs |
| 8 | Exclude carbs | `--exclude-cov carbs` | basal, bolus |

Raw JSON outputs: `runs/ablation_t1dm/*.json`

---

## Results

### Main comparison table

Sorted by MAE (best first). Δ columns are relative to the **All** configuration.

| # | Configuration | Active covariates | Windows | MAE ↓ | RMSE ↓ | MARD ↓ | Δ MAE | Δ RMSE |
|---|---------------|-------------------|--------:|------:|-------:|-------:|------:|-------:|
| 1 | **All** | basal + bolus + carbs | 819,013 | **13.078** | **19.722** | **11.50%** | — | — |
| 6 | Exclude basal | bolus + carbs | 819,013 | 13.109 | 19.761 | 11.50% | +0.030 | +0.039 |
| 4 | Include bolus | bolus | 819,013 | 13.216 | 20.104 | 11.38% | +0.138 | +0.382 |
| 8 | Exclude carbs | basal + bolus | 819,013 | 13.218 | 20.086 | 11.44% | +0.140 | +0.364 |
| 7 | Exclude bolus | basal + carbs | 819,013 | 13.400 | 20.242 | 11.97% | +0.321 | +0.521 |
| 3 | Include basal | basal | 819,013 | 13.402 | 20.306 | 11.74% | +0.323 | +0.585 |
| 5 | Include carbs | carbs | 819,013 | 13.501 | 20.412 | 12.02% | +0.423 | +0.690 |
| 2 | **None** | — | 819,013 | 13.546 | 20.571 | 11.66% | **+0.467** | **+0.849** |

### Marginal contribution (exclude-one analysis)

Effect of removing a single covariate while keeping the other two:

| Removed covariate | MAE after removal | Δ MAE vs All | Relative importance |
|-------------------|------------------:|-------------:|--------------------:|
| Basal | 13.109 | **+0.030** | Low |
| Carbs | 13.218 | +0.140 | Medium |
| Bolus | 13.400 | **+0.321** | **High** |

### Single-covariate sufficiency (include-one analysis)

Effect of keeping only one covariate (others zeroed):

| Only covariate kept | MAE | Δ MAE vs None | Δ MAE vs All |
|---------------------|----:|--------------:|-------------:|
| Bolus | **13.216** | **−0.329** | +0.138 |
| Basal | 13.402 | −0.144 | +0.323 |
| Carbs | 13.501 | −0.045 | +0.423 |

Bolus alone recovers **~70%** of the gain from using all covariates (0.33 mg/dL of the 0.47 mg/dL gap between None and All).

### RMSE vs MAE

RMSE spreads are larger than MAE spreads (e.g. All → None: **+0.85 RMSE** vs **+0.47 MAE**). Covariates reduce **large errors** disproportionately — consistent with bolus/carbs capturing meal and correction spikes that glucose history alone misses.

---

## Conclusions

1. **Covariates help on T1DM loop data.** Full covariates beat glucose-only by **0.47 mg/dL MAE (~3.6%)** and **0.85 mg/dL RMSE (~4.3%)** on this test set. The effect is larger than the ~0.23 mg/dL average gain on the **mixed** joined2 test split (loop + ai_ready), because every row here comes from pump-enriched loop traces.

2. **Bolus is the dominant channel.** It has the highest marginal value when excluded (+0.32 MAE) and performs best as a single channel (13.22 MAE). This aligns with bolus events marking intentional insulin delivery tied to near-future glucose dynamics, despite appearing on only **4%** of raw rows.

3. **Basal contributes least at the margin.** Excluding basal barely changes MAE (+0.03 mg/dL). Basal is more continuous (37% row fill, forward-filled across windows) and may be partially redundant with recent glucose trend after imputation — or its effect is spread thin compared to discrete bolus pulses.

4. **Carbs are useful but sparse.** Excluding carbs costs +0.14 MAE; carbs-only is the weakest single channel (+0.42 MAE vs All). Carb entries are rare (2.3% of rows) but still add information beyond bolus alone (Exclude carbs vs Exclude basal: +0.11 MAE difference).

5. **Include-one ≠ exclude-one symmetry.** `--include-cov bolus` (13.22) is much better than `--exclude-cov bolus` (13.40) relative to All (13.08). Single-channel modes discard complementary information; the model was trained with all channels present, so asymmetric degradation is expected.

6. **Practical implication.** For deployment on pump/CGM data where carb logging is unreliable, **basal + bolus** (Exclude carbs, MAE 13.22) is nearly as good as the full model (13.08). Prioritize bolus signal quality over carbs when sensor bandwidth or documentation is limited.

---

## Limitations and caveats

- **Same checkpoint, perturbed inputs only.** We ablate covariates at inference time; the model was trained with all channels. Retraining with matched covariate subsets could change relative rankings.
- **Not additive.** Marginal Δ MAE values from exclude-one tests do not sum to the total All vs None gap because channels interact non-linearly in the transformer.
- **Scaler fitting unchanged.** MinMax scalers always use full training covariates; ablated channels are zeroed after imputation, not re-scaled.
- **T1DM loop users only.** Results do not generalize to ai_ready wearable cohorts where insulin columns are empty.
- **Interpolated rows included.** ~18.6% of rows are `Interpolated` events; metadata `drop_interpolated` is false for this checkpoint.

---

## Reproduction

```bash
# Example: all covariates (best)
uv run evaluate-model \
  --run-dir runs/sugar_one_tune/production/trial_0000_bcd3813f \
  --model-type sugar_one \
  --test-csv data/loop_and_ai_ready/ablation_test.csv \
  --train-csv data/loop_and_ai_ready/loop_ai_ready_joined2.csv \
  --test-split "" \
  --batch-size 256 \
  --output-json runs/ablation_t1dm/all_cov.json

# Example: bolus only
uv run evaluate-model ... --include-cov bolus --output-json runs/ablation_t1dm/include_bolus.json

# Example: glucose only
uv run evaluate-model ... --zero-cov --output-json runs/ablation_t1dm/none_cov.json
```

Full flag reference: `scripts/sugar_one/README.md`

---

## Related artifacts

| Path | Description |
|------|-------------|
| `data/loop_and_ai_ready/ablation_test.csv` | T1DM test subset used here |
| `runs/ablation_t1dm/*.json` | Machine-readable metrics for all 8 runs |
| `docs/GLUMIND_VS_SUGARONE_COMPARISON.md` | Cross-model benchmark on full joined2 test split |
| `scripts/sugar_one/README.md` | `evaluate-model` CLI documentation |
