# SugarJepa (proof of concept)

SugarOne + a pretrained [CGM-JEPA](https://github.com/cruiseresearchgroup/CGM-JEPA) glucose embedding as
a 4th cross-attention auxiliary stream (basal / bolus / carbs / **jepa**, learnable softmax mix). See
`sugar_jepa_model.py` for the architecture and `vendor/cgm_jepa/NOTICE.md` for what's vendored and why.

This is a separate, self-contained experiment folder — nothing in `src/sugar_one/`,
`scripts/glumind*/`, or `src/common/` is modified, only imported.

## Run tonight (exact command)

```bash
uv run python src/sugar_jepa/train_sugar_jepa.py \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda \
  --d-model 32 --n-heads 8 --n-blocks 5 --ff-units 128 --input-steps 128 --horizon 12 \
  --lr 0.0004 --weight-decay 0.00003 --batch-size 256 \
  --epochs 30 --patience 3 --val-every-n-epochs 5 --num-workers 0 \
  --out-dir data/output/runs/sugar_jepa
```

(These are also the defaults baked into `train_sugar_jepa.py`'s CLI — you can omit all of them and just
run `--csv ... --device cuda`. They're spelled out here so it's obvious at a glance where they come from.
Flags are kebab-case, e.g. `--batch-size` not `--batch_size` — run `--help` to see the full, generated
list.)

That's it — the JEPA encoder loads its pretrained weights from the local, already-downloaded
`src/sugar_jepa/pretrained/cgm_jepa/` (no network needed), stays **frozen** by default. The
**model architecture + optimizer** hyperparameters above (`d-model`/`n-heads`/`n-blocks`/`ff-units`/
`input-steps`/`horizon`/`lr`/`weight-decay`/`batch-size`) are copied verbatim from
[`src/sugar_one/tune_sugar_one_full.toml`](../sugar_one/tune_sugar_one_full.toml) — the best-tuned
SugarOne values — so results are directly comparable to the tuned SugarOne production run. The
**training-loop control** values (`epochs`/`patience`/`val-every-n-epochs`/`num-workers`), by contrast,
come from [`tune_sugar_one_dev.toml`](../sugar_one/tune_sugar_one_dev.toml), not `..._full.toml`:
`tune_sugar_one_full.toml`'s epochs=120/patience=10 are sized for a production run against the full
~1GB CSV and would take far too long here. Since this run is against the dev CSV for a fast comparison,
it uses dev.toml's shorter loop (epochs=30, patience=3, validate every 5 epochs, `num_workers=0` to avoid
Windows DataLoader worker-spawn stalls) instead. Both source values live side by side in
[`tune_sugar_jepa_full.toml`](tune_sugar_jepa_full.toml) (`[defaults]` from full.toml, `[train]` from
dev.toml) for reference.

It checkpoints every epoch to `data/output/runs/sugar_jepa/sugar_jepa_global_h12_<timestamp>/` (`best_model.pt`,
`last_checkpoint.pt`) and writes `val_metrics_overall.csv` / `test_metrics_overall.csv` at the end, so if
it's still running later you can `Ctrl+C` and resume with `--resume-from
data/output/runs/sugar_jepa/.../last_checkpoint.pt`, or just evaluate whatever `best_model.pt` exists so far.

Rough sizing: the dev CSV has 1050 series (55MB); with a 288-step JEPA lookback, ~59% of series are long
enough to contribute windows (shorter ones are skipped — this is logged). With epochs=30/patience=3 (vs.
the production 120/10), this should finish in well under an hour on a single consumer GPU (RTX
3060-class) — validate-every-5-epochs plus early stopping at patience=3 means it can stop as early as
~15-20 epochs in.

## Evaluate a finished run

```bash
uv run glucose evaluate \
  --run-dir data/output/runs/sugar_jepa/<run_name> \
  --model-type sugar_jepa \
  --data data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda \
  --no-plot
```

Reports MAE / RMSE / MARD. For a same-CSV SugarOne baseline:

```bash
uv run glucose evaluate \
  --data data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --run-dir <a SugarOne run trained on the same dev CSV> \
  --model-type sugar_one \
  --no-plot
```

## Fast smoke test (no GPU wait, ~1 minute)

Before a long run, sanity-check the pipeline end-to-end on a tiny slice:

```bash
uv run python src/sugar_jepa/train_sugar_jepa.py \
  --csv data/input/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
  --device cuda --epochs 2 --max-train-series 20 --max-eval-series 10 \
  --batch-size 32 --n-blocks 2 --out-dir data/output/runs/sugar_jepa_smoketest
```

Unit/shape tests: `uv run pytest tests/test_sugar_jepa_smoke.py -q`.

## What's different from SugarOne

- `CrossAttentionSugarJepaBlock` mixes **4** auxiliaries (basal, bolus, carbs, jepa) via a learnable
  softmax weight, vs. SugarOne's 3-way mix.
- `JepaEncoderWrapper` runs a pretrained CGM-JEPA `Encoder` (3 layers, embed_dim 96, vendored in
  `vendor/cgm_jepa/`) over a **separate, longer glucose-only lookback window** (`--jepa-window`, default
  288 steps = 24h — the window CGM-JEPA was pretrained on) each forward pass, independent of the model's
  own `--input-steps` (default 128, unchanged from SugarOne). Frozen by default (feature-extractor mode);
  pass `--finetune-jepa` to unfreeze it (with its own, smaller `--jepa-lr`, default 4e-5).
- The JEPA branch's glucose is **z-score normalized** (its own scaler, fit on train only), not
  MinMax-scaled like the rest of the model — CGM-JEPA was pretrained on z-scored inputs, and feeding it
  `[0,1]`-scaled values would be out of distribution for the pretrained weights.
- Everything else (multi-scale self-attention, flatten + output head, MSE loss, MinMax scalers for
  glucose/basal/bolus/carbs, imputation policy) is identical to `src/sugar_one/sugar_one_model.py` /
  `train_sugar_one.py`.
- Scope: `global` training mode only (no per_group / cohort_wise / continual+LwF) — this is a proof of
  concept, not a full port.

## Known limitations (proof of concept, not a final architecture)

- **~41% of series are too short for the native 288-step JEPA window** and contribute zero training
  windows (logged as "Skipped N series shorter than..."). This is a deliberate fidelity-over-coverage
  tradeoff — see the plan/design notes below — not a bug.
- **Domain shift is untested**: CGM-JEPA's pretraining cohort/device mix may not resemble this repo's
  Loop-pump T1DM-heavy population. Treat the first run's results as an ablation ("does a frozen,
  off-the-shelf glucose embedding help at all here?"), not a verdict on the architecture.
- No hyperparameter tuner (`tune_sugar_jepa.py`) — `tune_sugar_jepa_full.toml` is a reference config, not
  a script input; there's no random-search loop like `tune_sugar_one.py`'s yet.
- Evaluate via `uv run glucose evaluate --model-type sugar_jepa` (same unified path as other families).

## Why local pretrained weights instead of `Encoder.from_pretrained("CRUISEResearchGroup/CGM-JEPA", ...)`

This started as a workaround for a Windows TLS crash on the original dev machine; that crash is now
root-caused and fixed (see `src/common/network.py`), so Hub-based loading works fine there too. The
vendored copy under `src/sugar_jepa/pretrained/cgm_jepa/` (`config.json` + `model.safetensors`, ~4MB,
MIT license) stays the default anyway — it's already fast, offline, and reproducible, so there's no
reason to add a network dependency for this proof of concept. Passing `--jepa-weights-dir` a Hub repo id
still works (`from_pretrained` accepts a Hub id or local path interchangeably) if you want it. See
`vendor/cgm_jepa/NOTICE.md` for the full explanation of the two TLS issues that were fixed.

## Long-run thoughts: what's the best way to use this beyond tonight?

1. **Treat tonight's run as an ablation, not a final architecture.** If frozen embeddings help despite
   the untested domain gap, further investment (fine-tuning, more context) is very likely to help more;
   if they're a no-op or hurt, that's the signal to invest in domain-adaptive pretraining instead of
   iterating on fusion architecture.
2. **Fine-tune progressively, not all at once**, once the ablation is positive: keep the encoder frozen
   while the fusion weights converge first (tonight's default), then unfreeze the top 1 of 3 JEPA blocks
   with a small LR for a second phase.
3. **If domain shift looks like the limiting factor**, do continued self-supervised pretraining on this
   project's own unlabeled glucose data first (`pretrain/pretrain_cgm_jepa.py` in the upstream repo, not
   vendored here) before touching the forecasting head.
4. **Reconsider the window-length mismatch as a deliberate phase-2 experiment**: tonight's design keeps
   SugarOne's `input_steps=128` and gives JEPA its own 288-step lookback purely for comparability. A
   version where the whole model runs at 288 steps is cleaner architecturally but forfeits today's
   apples-to-apples comparison — worth trying only once there's a positive signal.
5. **Fold results into the repo's existing comparison-doc convention**
   (`docs/GLUMIND_VS_SUGARONE_COMPARISON.md`, the per-model/dataset `data/output/marked_runs/*/RUNS_ANALYSIS.md`
   writeups) rather than treating this as
   a one-off side experiment.

## Second architecture track: `JepaEncoder` trained in-house (not wired to `train_sugar_jepa.py` yet)

Alongside the frozen, upstream-pretrained `JepaEncoderWrapper` path above, this folder also has a second,
independent JEPA track that trains its own encoder from scratch on this project's data instead of loading
CGM-JEPA's pretrained weights:

- [`jepa_pretrain.py`](jepa_pretrain.py) — self-supervised masked-latent pretraining of `JepaEncoder`
  (Conv1d patchify + sinusoidal positions + pre-norm blocks) over the 128-step glucose window, EMA target
  encoder, smooth-L1 loss in latent space. Writes `encoder.pt` / `encoder_best.pt` / `pretrain_metrics.csv`
  / diagnostic plots (see [`encoder_plots.py`](encoder_plots.py)) under a timestamped run dir.
- [`glucodensity.py`](glucodensity.py) / [`export_glucodensity.py`](export_glucodensity.py) — maps glucose
  windows to glucodensity images (KDE, adapted from CGM-JEPA's `glucodensity_utils.py`) and precomputes
  paired `(window, image)` `.npz` arrays for offline/Colab use.
- [`x_jepa_pretrain.py`](x_jepa_pretrain.py) / [`x_jepa_train.py`](x_jepa_train.py) — X-CGM-JEPA: a
  cross-modal predictor trained jointly on the CGM window and its glucodensity image
  (`L_total = L_CGM + w * L_Glu`), reusing `jepa_pretrain.py`'s CGM half unchanged. Pretrained encoder
  checkpoints live under `pretrained/xcgm_jepa/`.

This track's own model class is `SugarJepaModel2` in [`sugar_jepa_model.py`](sugar_jepa_model.py) —
batch-first throughout (no permutes), single `forward(x)` signature, and a trainable (not frozen)
`JepaEncoder` instead of `JepaEncoderWrapper`. **`train_sugar_jepa.py` currently builds `SugarJepaModel`,
not `SugarJepaModel2`**, so `run_downstream.sh`'s `--jepa-init` flag and the encoder checkpoints this
track produces are not yet consumable by the CLI trainer above — wiring `SugarJepaModel2` (and a
`--jepa-init` flag) into `train_sugar_jepa.py`, or adding it as a `glucose evaluate`-compatible model
family in its own right, is open follow-up work.
