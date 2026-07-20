# RoBioinfo 2026 Presentation — Restructuring Guide

## Audience profile
- **Primary:** Bioinformaticians, wet-lab biologists, clinical researchers
- **Secondary:** A small number of ML/AI people
- **Implication:** Lead with biology and clinical relevance, minimize ML jargon, explain every metric

## What changed vs. the Timisoara version

### Slides REMOVED or heavily condensed
| Old slide | Why removed/condensed |
|---|---|
| Slide 4: "Multimodal transformers… what ARE they?" | Nearly empty, typo ("algoritm"), too vague |
| Slide 5: Encoder vs Decoder comparison | Irrelevant to this audience — save for ML conferences |
| Slide 7: Patch tokenization / multiscale convolution | Implementation detail, not needed for bio audience |
| Slide 8: Cross-attention weight diagrams | Condensed into one architecture overview |
| Slide 9: Multi-scale attention weight diagrams | Condensed into one architecture overview |
| Slides 10–11: LwF problem + solution (2 full slides) | Condensed to 1 slide with clinical framing |

### Slides ADDED (new content)
| New slide | Why added |
|---|---|
| **Why predict glucose?** (slide 2) | Clinical motivation — the audience needs to know WHY before HOW |
| **The challenge: multimodal data** (slide 4) | Frames the ML problem in biological terms (different sensor types) |
| **Datasets: Who did we study?** (slide 7) | Bioinformaticians care deeply about data provenance, cohort design |
| **Error tracks disease severity** (slide 10) | The per-cohort gradient is a biological finding, not just an ML number |
| **Sugar-Sugar human benchmarking** (slide 11) | Novel study design, most discussion-generating content |
| **Beyond glucose** (slide 13) | Shows applicability to the audience's own problems |
| **Conclusions & next steps** (slide 14) | Standard slide — was missing entirely |

### Slides KEPT (with refinements)
| Slide | Changes |
|---|---|
| Title (slide 1) | Added conference name, grant acknowledgment |
| CGMs — what are they? (slide 3) | Slightly condensed text, kept images |
| Results T2DM (slide 8) | Added clinical context (MARD vs sensor accuracy), metric definitions |
| Results T1DM (slide 9) | Added honest note about RMSE gap — scientists respect transparency |
| Architecture diagram (slide 6) | Kept the general schema figure, added key innovation callout |

## Reusing visuals from the Timisoara presentation

The original presentation has several excellent custom figures. Reuse them in the new structure:

1. **Slide 2 images (CGM photos + Dexcom diagram)** → New slide 3
2. **Slide 3 (CGM report)** → Can be used as backup/appendix if someone asks "what does CGM data look like?"
3. **Slide 6 (Architecture general schema)** → New slide 6 — use this full-size, it's the best figure
4. **Slide 8 (Cross-attention weights)** → New slide 5 as a small inset, or appendix
5. **Slide 12 (Bar charts)** → New slides 8–9, but consider adding the per-cohort table too
6. **Slide 11 (LwF solution diagram)** → New slide 12 as a small inset

## Presentation flow (15 slides, ~20 min talk)

```
1. Title                          (~30 sec)
2. Why predict glucose?           (~2 min)  ← NEW: clinical hook
3. CGMs explained                 (~1.5 min)
4. Multimodal challenge           (~1.5 min) ← NEW: frames the problem
5. Sugar I: how it works          (~2 min)  ← CONDENSED from 6 slides
6. Architecture diagram           (~1 min)  ← optional detail
7. Datasets                       (~2 min)  ← NEW: key for this audience
8. Results: T2DM                  (~1.5 min)
9. Results: T1DM                  (~1 min)
10. Per-cohort error gradient     (~2 min)  ← NEW: biological finding
11. Sugar-Sugar human study       (~2 min)  ← NEW: most novel aspect
12. Continual learning            (~1 min)  ← CONDENSED from 2 slides
13. Beyond glucose                (~1 min)  ← NEW: applicability
14. Conclusions                   (~1 min)
15. Thank you / Q&A
```

## Key talking points for Q&A preparation

**Q: How does this compare to closed-loop insulin pump systems?**
A: Closed-loop systems (like Medtronic 780G) use simpler predictive algorithms built into the pump. Sugar I could serve as a better prediction engine for next-gen pump systems. We also have a GluMindIC variant that incorporates insulin and carb data for this use case.

**Q: Why not include meal data?**
A: AI-READI doesn't have reliable meal logs. The GluMindIC variant (scripts/glumind_ic/) handles insulin and carb inputs. Self-reported meal data is notoriously unreliable in clinical studies.

**Q: Is ~197K parameters really enough?**
A: Yes — the model has a narrow task (12-step regression) vs. LLMs that must represent all of language. ~197K params is tiny by modern standards (GPT has billions). The small size enables edge deployment on wearable devices. Bigger models (GluFormer) actually performed worse.

**Q: Can I use this on my own biosignal data?**
A: The architecture is modular. Replace the 3-channel input (glucose, HR, steps) with your own channels. The cross-attention mechanism generalizes to any set of irregularly sampled time series.

**Q: What about privacy concerns with health data?**
A: AI-READI is a public dataset with proper consent. Sugar-Sugar has ethics approval from UMR Rostock. Model weights are on HuggingFace — inference can run locally without sending data to any server.

---

## FACT-CHECK: Issues found by auditing the codebase

### ERRORS in the original Timisoara presentation

| Claim | What the code/data actually says | Severity |
|---|---|---|
| **~48K parameters** (mentioned in Q&A prep) | **196,972 parameters** (from tuning.txt line 37: `Params: 196,972`). **Fixed in the new presentation.** | HIGH — 4x off |
| Slide 4 typo: "algoritm" | Should be "algorithm" | LOW |
| Slide 4 nearly empty | No substantive content on the slide | MEDIUM |

### DISCREPANCIES between abstract and actual run data

| Abstract claim | Codebase value | Explanation |
|---|---|---|
| MAE **11.39** (AI-READI) | **11.33** (val_metrics from AI Ready Only run) | Abstract likely rounded a slightly different run or used per-group average instead of overall. The new presentation uses 11.33 from the actual run CSV. |
| RMSE **17.62** (AI-READI) | **17.73** (val_metrics from AI Ready Only run) | Same issue — 0.6% difference. |
| NHITS MAE **19.88** / RMSE **31.81** | Our NHITS runs: MAE **20.60** / RMSE **34.45** | Abstract appears to cite the original GluMind paper's NHITS numbers, not our own runs. Our NHITS is actually worse, which makes Sugar I look even better. The new presentation uses our actual run numbers. |
| MARD **8.25%** (AI-READI) | Matches val_metrics: **8.2476%** | Correct (rounded). |
| **896 participants** across 4 cohorts | Not verifiable from codebase. Training windows: 4.45M train / 977K val / 984K test. | The 896 number comes from the AI-READI consortium documentation, not from our code. Livia should verify against the AI-READI paper. |
| **53 public CGM datasets** | Not found anywhere in the repo. No catalogue file exists. | This number may come from a separate dataset survey not in this repo. Livia needs to provide the source or we should remove this claim. |

### NAMING: "Sugar I" vs "GluMind"

- All code, configs, scripts, and reports use **"GluMind"**
- The abstract and presentation use **"Sugar I"**
- These appear to be the same model. "Sugar I" seems to be the public/presentation name, "GluMind" the internal/codebase name
- **Recommendation:** Be consistent. If using "Sugar I" in the talk, explain once that it's "our implementation of the GluMind architecture" to avoid confusion for anyone who reads the code later

### VERIFIED claims (correct)

- Sugar I/GluMind outperforms NHITS by 42-45% MAE on AI-READI: **CONFIRMED** (CROSS_MODEL_COMPARISON.md: 44.97%)
- Sugar I outperforms GluFormer by ~40% MAE: **CONFIRMED** (40.11%)
- Error gradient: Healthy < Pre-T2DM < Oral < Insulin < T1DM: **CONFIRMED** from test_metrics_by_study_group.csv
- Cross-attention: glucose queries HR and steps separately, averaged 0.5/0.5: **CONFIRMED** (glumind_model.py line 67)
- Multi-scale: DS=1, DS=2, DS=4 downsampling factors: **CONFIRMED** (glumind_model.py lines 73-127)
- Parallel architecture (cross-attn + multi-scale run simultaneously): **CONFIRMED** (glumind_model.py lines 130-149)
- Input: 80 steps (400 min) → output: 12 steps (60 min): **CONFIRMED** from config.json
- NHITS retains lower RMSE on T1DM (21.05 vs 23.00): **CONFIRMED** (CROSS_MODEL_COMPARISON.md line 125: delta = -1.9479)
- GluMindIC has learnable softmax mixing: **CONFIRMED** (glumind_ic_model.py line 65: `nn.Parameter(torch.zeros(3))`)
- Learning-without-forgetting lambda=0.3 sweet spot: **CONFIRMED** from reports

### Architecture details verified from code (for accuracy in slides)

| Parameter | Value | Source |
|---|---|---|
| d_model | 32 | config.json |
| n_heads | 4 | config.json |
| n_blocks | 3 | config.json |
| ff_units | 128 | config.json |
| dropout | 0.1 | config.json |
| input_steps | 80 (= 400 min at 5-min freq) | config.json |
| horizon | 12 (= 60 min) | config.json |
| total parameters | 196,972 | tuning.txt |
| optimizer | AdamW (lr=0.001, wd=0.0001) | config.json |
| precision | bf16 mixed | config.json |
| batch_size | 4096 | config.json |
| training epochs (best model) | 65 (early stopped from 120, patience=20) | tuning.txt |

---

## Generated figures

8 publication-quality PNG figures generated from actual codebase data:

| Figure | File | Use in slide |
|---|---|---|
| Training convergence curve | `fig_training_curve.png` | New slide (optional) or appendix |
| Per-cohort MAE comparison (4 models) | `fig_per_cohort_mae.png` | Slide 10 (Error tracks disease severity) |
| Overall T2DM metrics (3 panels) | `fig_overall_t2dm.png` | Slide 8 (Results: T2DM) |
| Overall T1DM metrics (3 panels) | `fig_overall_t1dm.png` | Slide 9 (Results: T1DM) |
| Error gradient by disease severity | `fig_error_gradient.png` | Slide 10 (standalone version) |
| Improvement heatmap (% over baselines) | `fig_improvement_heatmap.png` | Slide 10 or conclusions |
| Architecture diagram (simplified) | `fig_architecture.png` | Slide 5-6 |
| Checkpoint metrics evolution | `fig_checkpoint_metrics.png` | Optional / appendix |

To regenerate: `uv run python docs/presentation/generate_figures.py`
