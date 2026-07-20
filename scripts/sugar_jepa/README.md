# SugarJepa

SugarOne + **our own** JEPA glucose encoder as a 4th cross-attention auxiliary stream
(basal / bolus / carbs / **jepa**, learnable softmax mix).

We port the *architecture*, not the weights. `JepaEncoder` in
[`sugar_jepa_model.py`](sugar_jepa_model.py) is a plain patch transformer — Conv1d patchify, sinusoidal
positions, pre-norm blocks — trained by us, either end-to-end from random init or from a self-supervised
checkpoint we produce ourselves with [`jepa_pretrain.py`](jepa_pretrain.py). No `from_pretrained`, no
`safetensors`, no network.

The JEPA branch reads its glucose from `x[..., 0]` — the **same 128-step lookback** the rest of the model
sees — so the dataset contract stays SugarOne's plain `(x, y)`: no second tensor, no second scaler, no
separate window. Every series long enough for SugarOne is long enough for SugarJepa.

Scope: `global` training mode only.

## The three commands

### 1. Train end-to-end (random init)

```bash
uv run python scripts/sugar_jepa/train_sugar_jepa.py \
  --csv data/input/loop_ai_ready_joined2_dev.csv \
  --device cuda \
  --d-model 32 --n-heads 8 --n-blocks 5 --ff-units 128 --input-steps 128 --horizon 12 \
  --lr 0.0004 --weight-decay 0.00003 --batch-size 256 \
  --epochs 30 --patience 3 --val-every-n-epochs 5 --num-workers 0 \
  --jepa-patch-size 8 --jepa-embed-dim 96 --jepa-layers 3 --jepa-heads 6 --jepa-lr 4e-5 \
  --out-dir runs/sugar_jepa
```

All of these are the CLI defaults, so `--csv ... --device cuda` alone reproduces it. Flags are kebab-case
(Typer); `--help` prints the full list.

The encoder **always trains** — there is no frozen mode. It just gets its own smaller-LR optimizer group
(`--jepa-lr`), which is why the startup banner reports two param counts:

```
Training 367,840 out of 367,840 SugarOne params @ lr=0.0004
Training 336,576 out of 336,576 JEPA params @ lr=4e-05
```

`input_steps` must be divisible by `jepa_patch_size` (128 / 8 = 16 patches of 40 min); it fails up front
if not.

### 2. Pretrain the encoder (JEPA self-supervision)

```bash
uv run python scripts/sugar_jepa/jepa_pretrain.py \
  --csv data/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv --device cuda \
  --window-stride 4 --epochs 50 --batch-size 256 \
  --patch-size 8 --embed-dim 96 --n-layers 3 --n-heads 6
```

Masked latent prediction: an EMA target encoder encodes all 16 patches, the context encoder sees only the
unmasked ones, and a narrow predictor — given just the *positions* of the masked blocks — must reproduce
their latents. Loss is smooth-L1 **in latent space**; nothing reconstructs glucose values.

Trains on the CSV's **train split only**. Val/test rows never enter this stage, or every forecasting number
downstream is leakage-contaminated. A slice of the *train* series (`--holdout-frac`) is held out to watch
the objective.

Each run gets its own timestamped directory under `runs/`, like the trainers — so a second pretrain
cannot silently overwrite the encoder an already-fine-tuned model was initialized from:

```
runs/jepa_encoder/
├── latest.txt                                       # points at the most recent run
└── jepa_encoder_w128_p8_d96_l3_h6_20260715_002330/
    ├── config.json           # the SSL config + final latent_std / eff_rank
    ├── encoder.pt            # last epoch — this is what --jepa-init loads
    ├── encoder_best.pt       # lowest holdout loss
    ├── pretrain_metrics.csv  # the per-epoch curve
    └── plots/epoch_001.png … # 4-panel encoder diagnostics
```

The directory name carries the encoder shape (`w128_p8_d96_l3_h6` = window / patch / dim / layers /
heads), because those must match the fine-tuning config exactly — the `--jepa-init` load is
`strict=True`.

**Watch `latent_std` and `eff_rank`, not the loss.** Representation collapse — the encoder emitting nearly
the same vector for every window — drives the loss toward zero and looks like a triumph. Both are logged
every epoch and to the CSV. Your reference: a *random-init* encoder at `embed_dim=96` gives
`latent_std ≈ 0.67`. Healthy means staying the same order (0.4–1.0); a monotone slide toward 0 is collapse
in progress, and below ~0.1 the run is dead.

Then fine-tune from it — same command as (1), plus:

```bash
  --jepa-init runs/jepa_encoder/jepa_encoder_w128_p8_d96_l3_h6_<timestamp>/encoder.pt
```

Encoder weights only; `jepa_proj` and the backbone stay randomly initialized. The load is `strict=True`, so
a shape mismatch between the SSL config and the training config is an error, not a silent partial load.

### 3. Evaluate

```bash
uv run python scripts/sugar_jepa/evaluate_sugar_jepa.py \
  --run-dir data/output/runs/sugar_jepa/<run_name> \
  --test-csv data/input/loop_ai_ready_joined2_dev.csv \
  --device cuda
```

Reports MAE / RMSE / MARD overall and per Study Group, and prints the checkpoint's learned mix weights.
Architecture hyperparameters come from the run's `config.json` / `tuning_meta.json`.

## Metrics CSVs

Both trainers write one row per epoch, flushed immediately — plottable mid-run, and a killed run keeps
what it finished.

- `<run_dir>/training_metrics.csv` — `epoch, train_loss, val_loss, lr, lr_group1, best_val_loss,
  epoch_seconds, mix_basal, mix_bolus, mix_carbs, mix_jepa`
- `runs/jepa_encoder/<run>/pretrain_metrics.csv` — `epoch, train_loss, holdout_loss, latent_std, eff_rank, embed_dim,
  ema_momentum, lr, epoch_seconds`

A skipped validation epoch writes a **blank**, not `0.0` (which would plot as a perfect score).

`mix_jepa` is the diagnostic worth watching: it's `softmax(mix_logits)[jepa]` averaged over blocks, i.e.
how much weight the model actually gives the JEPA stream. If it decays well below 0.25, the model is
routing around the branch and whatever it learns is coming from the SugarOne backbone.

## Smoke test

```bash
uv run pytest tests/test_sugar_jepa_batch_first.py tests/test_jepa_pretrain.py -q

uv run python scripts/sugar_jepa/train_sugar_jepa.py \
  --csv data/input/loop_ai_ready_joined2_dev.csv \
  --device cuda --epochs 2 --max-train-series 20 --max-eval-series 10 \
  --batch-size 32 --n-blocks 2 --out-dir data/output/runs/sugar_jepa_smoketest
  --csv data/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv \
```

`tests/test_sugar_jepa_smoke.py` is **stale and failing** — it tests the retired 288-step dataset. Rewrite
or delete it when the old path goes.

## What's different from SugarOne

- `CrossAttentionSugarJepaBlock` mixes **4** auxiliaries via a learnable softmax weight, vs. SugarOne's
  3-way mix.
- `JepaEncoder` encodes the same 128-step glucose window into 16 patch embeddings (dim 96), projected to
  `d_model` by `jepa_proj` to serve as the 4th K/V stream.
- The encoder **instance-normalizes each window** (per-window z-score) inside its own forward pass. A
  z-score is invariant to affine maps, and MinMax scaling is affine — so an encoder pretrained on raw
  mg/dL sees an identical input distribution when fine-tuned on MinMax-scaled `x[..., 0]`. That is what
  removes the need for a separate JEPA scaler.
- Its blocks run `batch_first=True` (the flag defaults to `False` for the legacy model, which still uses
  the `(seq, batch, d_model)` contract).
- Everything else — multi-scale self-attention, output head, MSE loss, MinMax scalers, imputation, the
  training loop — is SugarOne's, imported not copied.

## Known gaps

- **No baseline on disk.** The SugarOne run the old PoC was compared against is gone, so a plain SugarOne
  run at matched config on matched window counts is a prerequisite for any claim here.
- **The JEPA encoder is ~336k params against a ~368k backbone**, so a random-init win could just be extra
  capacity. A capacity-matched SugarOne control (bump `n-blocks`/`d-model` until params match) is not
  optional. `--jepa-embed-dim 64 --jepa-layers 2` gets the branch to ~100k if you'd rather shrink it.
- **No `--ablate-jepa`.** Doing it right means renormalizing the 4-way softmax over the three remaining
  auxiliaries — otherwise `w[3]` still multiplies a nonzero term and "ablating" perturbs basal/bolus/carbs
  too.
- No hyperparameter tuner; `tune_sugar_jepa_full.toml` is a reference config, not a script input.
- `evaluate_sugar_jepa.py` is standalone, not folded into the unified `evaluate-model` CLI.

## The retired path

`SugarJepaModel` + `JepaEncoderWrapper` + `vendor/cgm_jepa/` are the original proof of concept: a
**frozen, pretrained** CGM-JEPA encoder over a separate **288-step** lookback. They are kept only because
`tests/test_sugar_jepa_smoke.py` still imports them, and should be deleted with it.

That design gave the JEPA branch a 288-step window (what CGM-JEPA was pretrained on) while the backbone
kept 128, so the dataset built windows at `max(128, 288) = 288` and every series shorter than 300 steps
produced **zero** windows — ~41% of dev-CSV series dropped. SugarJepa was scored on 68,862 val windows vs.
SugarOne's 88,574. The two models were never evaluated on the same data, which is why the 4–5% MAE win in
`docs/SUGAR_JEPA_VS_SUGAR_ONE_DEV_COMPARISON.md` is not usable as a baseline. See
[`docs/JEPA_INTEGRATION_PLAN.md`](../../docs/JEPA_INTEGRATION_PLAN.md).
