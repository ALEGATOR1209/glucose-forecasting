# JEPA Integration Plan — fold into SugarOne at a single 128-step window

**Status:** proposed, not started
**Date:** 2026-07-13
**Supersedes:** the `scripts/sugar_jepa/` proof of concept (frozen, pretrained CGM-JEPA @ 288 steps)

## Decisions this plan is built on

1. **Fold into SugarOne.** The JEPA stream becomes an optional 4th cross-attention auxiliary inside
   `scripts/sugar_one/`, flag-gated (`--use-jepa`). `scripts/sugar_jepa/` is retired once this
   reproduces. One model family, one train CLI, one eval CLI.
2. **Our own JEPA implementation.** We port the *architecture*, not the weights. The vendored
   `scripts/sugar_jepa/vendor/cgm_jepa/` becomes a reference we read, not a dependency we import. No
   `from_pretrained`, no `safetensors`, no `huggingface_hub` in the model path.
3. **One window: 128 steps.** No separate `--jepa-window`. The JEPA branch reads the same lookback the
   rest of SugarOne reads.
4. **The encoder trains.** It is never frozen — it gets its own smaller-LR optimizer group. "Fine-tune"
   in stage B means "initialize from our SSL checkpoint and keep training", not "freeze".
5. **Staged.** Stage A = architecture only, random init, joint end-to-end training. Stage B = JEPA
   self-supervised pretraining on our own glucose, then init stage A's encoder from it. Stage B's
   headline number is meaningless without stage A as its control.
6. **SSL pretrains on the Loop CSV train split only** (glucose column, train rows). No val/test rows
   touch the SSL stage — otherwise the forecasting numbers are leakage-contaminated.

## Why the window change is the point, not a detail

The PoC gave JEPA a 288-step (24 h) lookback because that is what CGM-JEPA was pretrained on, while the
SugarOne backbone kept 128. Two consequences, both bad:

- **Coverage.** The dataset built windows at `lookback = max(128, 288) = 288`, so every series shorter
  than 300 steps produced zero training windows — ~41% of dev-CSV series dropped. It shows up directly
  in the recorded numbers: SugarJepa evaluated on 68,862 val windows vs. SugarOne's 88,574 on the same
  split. The two models were never scored on the same data, which quietly undermines the 4–5% MAE win
  reported in `docs/SUGAR_JEPA_VS_SUGAR_ONE_DEV_COMPARISON.md`.
- **Plumbing.** The 288-step view forced a second tensor through the dataset, the collate, the training
  loop, and the eval loop (`x, jepa, y` instead of SugarOne's `x, y`), which is exactly why
  `evaluate_sugar_jepa.py` had to be a standalone fork of `evaluate-model` instead of a `--model-type`.

Dropping to 128 fixes both at once. Once the JEPA lookback *is* the SugarOne lookback, the branch can
derive its input from `x[..., 0]` inside the model, the dataset contract stays `(x, y)`, and the whole
thing folds into the existing SugarOne train/eval path with no new tensor plumbing.

## Target design

### Normalization: per-window instance z-score, computed inside the model

The PoC carried a separate `StandardScaler` for the JEPA branch, fit on train, because the pretrained
CGM-JEPA weights expected z-scored inputs. We are dropping those weights, so that constraint is gone —
and with it, the reason to plumb a fifth scaler through the dataset, the checkpoint, and the eval script.

Instead: the JEPA branch standardizes **each window against its own mean/std** at the top of its forward
pass (RevIN-style instance norm). This keeps the dataset yielding `(x, y)` with SugarOne's existing
MinMax scalers untouched, keeps the SSL pretraining stage and the forecasting stage on an identical input
distribution for free, and makes the branch scale-invariant.

Tradeoff to be aware of: per-window standardization discards absolute glucose level from the JEPA branch
(a window at 250 mg/dL and one at 90 mg/dL with the same shape become identical). That is acceptable
*here* because the main glucose stream is still MinMax-scaled against a global scaler and carries the
absolute level into the same cross-attention block. Worth a `--jepa-norm {instance,global}` flag so this
assumption is testable rather than baked in.

### Patch size: 8 (16 patches of 40 min)

CGM-JEPA used 12-step patches (1 h) × 24 = 288. 128 is not divisible by 12. Since we own the
architecture now, we pick a patch size that divides 128 rather than truncating: **`patch_size=8` → 16
patches**, each 40 minutes. Assert `input_steps % jepa_patch_size == 0` at construction and fail loudly
rather than silently dropping a remainder. `patch_size=16` (8 patches of 80 min) is the obvious
alternative and should be a flag, not a rewrite.

Patch embedding simplifies too. Upstream's `ValueEmbedding` runs `Conv1d(1, d, kernel=3, stride=3)` over
each 12-sample patch, flattens the 4 outputs, and pushes them through a `Linear(4d → d)` — a shape that
only exists to match their checkpoint. Ours is a single `Conv1d(1, d, kernel=patch, stride=patch)`, which
is the standard patch-embed and what the upstream design collapses to anyway. Sinusoidal position
embeddings are length-agnostic, so nothing else cares about the sequence length change.

### Model wiring — backward compatibility is a hard requirement

`SugarOneModel` gains `use_jepa: bool = False`. When `False`, **no JEPA submodules are constructed and
the `state_dict` keys are byte-identical to today's**, so every existing SugarOne checkpoint keeps
loading. This is the single constraint that must not break; it gets its own regression test against the
bundled `test_model_sugar_one/best_model.pt`.

The mixing weight generalizes from a hardcoded 3-way softmax to `n_aux`-way: `mix_logits` is
`nn.Parameter(torch.zeros(3))` when `use_jepa=False` and `torch.zeros(4)` when `True` — same key name,
same shape as today in the default case.

CLAUDE.md notes the model files are "intentionally frozen" — this plan is the specific request that
unfreezes `sugar_one_model.py`. The JEPA encoder classes go *in that file* (not in `scripts/common/`) so
the repo's "a checkpoint loads with just the model file + `torch.load`" property survives; only the SSL
training machinery (predictor, EMA target, masking) lives elsewhere, since it is never needed at
inference.

## Phases

### Phase 0 — baseline the control (before writing any model code)

The PoC's 4–5% win was measured against a different window count, so it is not usable as a baseline. And
the SugarOne run it was compared against is **gone from disk** — `runs/` currently holds only
`sugar_jepa/`, there is no `marked_runs/`, and `runs/sugar_one_tune/explore_dev2/` does not exist. The
`trial_0001` numbers survive only as text in the comparison doc, with no checkpoint and no window counts.
So phase 0 is a real training run, not a lookup.

- Train SugarOne (3-aux, no JEPA) on `loop_ai_ready_joined2_dev.csv` at
  `d_model=32, n_heads=8, n_blocks=5, ff_units=128, dropout=0.1, input_steps=128, horizon=12, lr=4e-4, weight_decay=3e-5, batch_size=256`
  — the `trial_0001` config. Record val/test MAE/RMSE/MARD **and window counts**.
- Every number in phases 1–3 gets compared to this row, on identical window counts.

### Phase 1 — JEPA encoder as an architecture, random init, joint training (Stage A)

Files:

- **`scripts/sugar_one/sugar_one_model.py`** *(modified)*
  - `JepaPatchEmbed` — `Conv1d(1, embed_dim, kernel=patch, stride=patch)` + sinusoidal pos emb.
  - `JepaBlock` — pre-norm MHSA + MLP (`mlp_ratio=4`, GELU). Straight port of
    `vendor/cgm_jepa/modules.py:Block`, reimplemented, no mask path.
  - `JepaEncoder` — patch embed → `n_layers` × `JepaBlock` → `LayerNorm`. Returns
    `(batch, n_patches, embed_dim)`. Defaults `embed_dim=96, n_layers=3, n_heads=6` (upstream's shape;
    small relative to the ~1-2 M-param SugarOne backbone). Instance-norms its input.
    **No `proj` head** — upstream's `MLP(96 → 1024 → 48)` exists only for their SSL objective and is
    dead weight here; the SSL predictor in phase 2 lives in the pretraining module instead.
  - `JepaBranch` — `JepaEncoder` + `Linear(embed_dim → d_model)`, producing the
    `(n_patches, batch, d_model)` K/V stream. Reads `x[..., 0:1]`; takes no extra forward argument.
  - `CrossAttentionSugarOneBlock` — extend to an optional 4th `attn_jepa` head and an `n_aux`-way
    softmax mix. Guarded so `use_jepa=False` reconstructs today's exact module tree.
  - `SugarOneModel.forward(x)` — signature **unchanged**. When `use_jepa`, it computes the JEPA stream
    once (not per block) and threads it through the blocks.
- **`scripts/sugar_one/train_sugar_one.py`** *(modified)*
  - New flags: `--use-jepa`, `--jepa-patch-size` (8), `--jepa-embed-dim` (96), `--jepa-layers` (3),
    `--jepa-heads` (6), `--jepa-norm` (`instance`), `--jepa-lr` (4e-5), `--jepa-init` (path to an SSL
    checkpoint, empty = random init — used in phase 3).
  - Optimizer gains the two-group split (JEPA params at `--jepa-lr`, everything else at `--lr`),
    lifted from `train_sugar_jepa.py:make_optimizer_and_scheduler`.
  - Dataset, collate, training loop, imputation policy: **untouched**.
  - Log `softmax(mix_logits)` each validation — how much weight the model actually gives the JEPA
    stream is the single most informative diagnostic we get, and the PoC never recorded it.
- **`scripts/sugar_one/evaluate_model.py`** *(modified)*
  - Construct with `use_jepa` from the run's `config.json` / `tuning_meta.json`. Since the forward
    signature and dataset are unchanged, `_run_evaluate` needs no changes.
  - Add `"jepa"` to `SUGAR_ONE_COVARIATES` so `--zero-cov jepa` ablates the branch at inference.
- **`tests/test_sugar_one_jepa.py`** *(new)* — shape tests (`use_jepa` on/off), the
  `input_steps % patch_size` assertion, the two-group optimizer, and the **backward-compat test**: the
  bundled `test_model_sugar_one/best_model.pt` still loads into `use_jepa=False` with `strict=True`.
  (It is the only SugarOne checkpoint left on disk, so it is what guards the "existing checkpoints keep
  working" requirement.)

Deliverable: a `--use-jepa` run on the dev CSV at matched hyperparameters, scored on the **same window
count** as phase 0.

**Concrete code for this phase is in [Appendix A](#appendix-a--code-changes-for-the-128-step-jepa-branch)**,
written against the real signatures in `sugar_one_model.py`, `train_sugar_one.py` and `evaluate_model.py`.

**The control that makes phase 1 interpretable:** a random-init JEPA branch is, structurally, just a
second glucose encoder bolted onto the model — any gain could be raw extra capacity rather than anything
JEPA-shaped. So phase 1 reports two rows: `--use-jepa` and a capacity-matched SugarOne (bump `n_blocks`
or `d_model` until parameter counts match). If the JEPA row does not beat the capacity-matched row, the
architecture is not earning its keep and phase 2 should be reconsidered before it is written.

### Phase 2 — JEPA self-supervised pretraining on our own glucose (Stage B, part 1)

- **`scripts/sugar_one/jepa_pretrain.py`** *(new, Typer)* — the actual JEPA recipe:
  - Context encoder `E_θ` = `JepaEncoder` (imported from `sugar_one_model.py`).
  - Target encoder `E_ξ` = EMA copy of `E_θ`, `requires_grad=False`, momentum `0.996 → 1.0` on a cosine
    schedule.
  - Predictor `P` — a narrow transformer (`embed_dim // 2`, 2 layers) that takes context latents +
    mask tokens + target position embeddings and predicts target-block latents.
  - Masking over the 16-patch sequence: sample `M=4` target blocks of 2–4 contiguous patches; the
    context is the remaining patches with target positions removed. (I-JEPA multi-block, scaled down —
    16 patches is a much shorter sequence than an image's 196, so block counts/sizes need a sanity pass,
    not a copy-paste of the paper's constants.)
  - Loss: smooth-L1 between `P(context)` and `stop_grad(E_ξ(target))` in latent space. **No pixel/value
    reconstruction** — predicting in representation space is the whole point of JEPA.
  - Data: glucose-only 128-step sliding windows from the **train split** of
    `loop_ai_ready_joined2*.csv`, reusing `load_splits_streaming` + `impute_and_sort` from
    `scripts/common/data_loading.py`. Per-window instance norm, matching phase 1.
  - Guard against the classic JEPA failure: **representation collapse**. Log latent variance and the
    rank of the target-encoder output every epoch; a flat-lining variance means the run is dead and
    the numbers downstream are noise.
  - Output: `scripts/sugar_one/pretrained/jepa_128_p8/{config.json, encoder.pt}` — plain `state_dict`,
    key-compatible with `JepaEncoder`.
- **`tests/test_jepa_pretrain.py`** *(new)* — mask sampler produces disjoint context/target sets and
  non-empty blocks; EMA actually updates the target encoder; loss decreases on a tiny overfit batch.

### Phase 3 — initialize from SSL, fine-tune into SugarOne (Stage B, part 2)

- `--jepa-init scripts/sugar_one/pretrained/jepa_128_p8/encoder.pt` loads those weights into the branch;
  training proceeds exactly as phase 1 (encoder still trains, at `--jepa-lr`).
- Same config as phase 1, so the only delta is the encoder's initialization.

### Phase 4 — report and retire the fork

- Results table into `docs/` following the existing comparison-doc convention:

  | Run | Init | MAE | RMSE | MARD | Windows | Params | `softmax(mix)[jepa]` |
  |---|---|---|---|---|---|---|---|
  | SugarOne (phase 0) | — | | | | | | — |
  | SugarOne, capacity-matched | — | | | | | | — |
  | + JEPA branch (phase 1) | random | | | | | | |
  | + JEPA branch (phase 3) | SSL | | | | | | |

  All four rows on identical window counts, or the table is not a comparison.
- Mark `scripts/sugar_jepa/` deprecated in its README, pointing here. Delete the folder (and the
  vendored `pretrained/` weights) only once phase 3 has landed and the recorded PoC numbers are no
  longer the best JEPA result we have. `docs/SUGAR_JEPA_VS_SUGAR_ONE_DEV_COMPARISON.md` stays as
  history, with a note that its two rows were scored on different window counts.

## Risks and open questions

- **Does 128 steps starve the SSL stage?** Masked latent prediction over 16 patches is a much easier
  task than over 24 (and CGM-JEPA chose 24 h for a reason — that is where the circadian structure is).
  It is plausible that SSL at this window learns something too trivial to help. If phase 3 ≈ phase 1,
  this is the first suspect, and the answer is a phase-5 experiment where the *whole model* moves to 288
  steps — which the fixed-window design makes a one-flag change rather than a rewrite.
- **Random-init JEPA may win for the wrong reason** (extra capacity). Mitigated by the capacity-matched
  control in phase 1; do not skip it.
- **Instance norm discards absolute glucose level** in the JEPA branch. Mitigated by the main stream
  carrying it, and by `--jepa-norm global` existing as a fallback.
- **SSL collapse** is a real and quiet failure mode. Mitigated by variance/rank logging in phase 2.
- **Dev-CSV SSL data may simply be too small** for a self-supervised stage to pay off. If phase 3 is
  flat, rerunning the SSL stage against the full `loop_ai_ready_joined2.csv` train split (still no
  val/test rows) is cheaper than any architecture change and should be tried before concluding anything.

## Open questions for you

1. **Phase 0 costs a full SugarOne training run** (the old one is gone from disk). It is a prerequisite
   for every comparison downstream, but if you already know you want the JEPA code written regardless,
   phases 0 and 1 can be built in parallel — phase 0 just has to *finish* before any of the numbers mean
   anything.
2. **Dev CSV or full CSV?** Everything above assumes `loop_ai_ready_joined2_dev.csv` (fast iteration,
   matches the PoC). Both files are on disk; the full one changes runtimes by an order of magnitude and
   would need `epochs=120 / patience=10` per `tune_sugar_one_full.toml` rather than the dev loop's 30/3.
3. **Patch size 8 (16 patches × 40 min) vs. 16 (8 patches × 80 min).** I defaulted to 8. Both will be
   flags, so this only decides which one gets the GPU hours first.

---

# Appendix A — code changes for the 128-step JEPA branch

Written against the current signatures. Three source files change, one test file is new, and **no existing
SugarOne checkpoint, dataset, collate, or eval path is touched** when `use_jepa=False`.

A late addition from `docs/jepa_integration_guide_en.md`: the guide ranks the fusion options and says to
**start with concat at the bottleneck**, not cross-attention. Since both routes need the identical
encoder, the branch below is built once and the fusion point is a flag — `--jepa-inject concat|cross`
(`film` is a two-line extension of `concat`). Default is `concat`.

## A.1 `scripts/sugar_one/sugar_one_model.py`

New classes, appended after `PositionalEncoding` (which they reuse). Nothing above them changes.

```python
class JepaEncoder(nn.Module):
    """Patch-embed + pre-norm transformer over a glucose-only lookback.

    Architecture ported from CGM-JEPA (vendor/cgm_jepa/{embed,modules,encoder}.py);
    weights are NOT. Kept in this file so a checkpoint still loads with just the
    model file + torch.load, per the repo's checkpoint-friendliness rule.
    """

    def __init__(
        self,
        n_time_steps: int,
        patch_size: int = 8,
        embed_dim: int = 96,
        n_layers: int = 3,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm: str = "instance",
    ):
        super().__init__()
        if n_time_steps % patch_size != 0:
            raise ValueError(
                f"input_steps ({n_time_steps}) must be divisible by "
                f"jepa_patch_size ({patch_size}); 128 % 8 == 0, 128 % 12 != 0."
            )
        self.n_patches = n_time_steps // patch_size
        self.embed_dim = embed_dim
        self.norm_mode = norm

        # One conv over the raw series does patchify + embed in a single op.
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=self.n_patches)

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(embed_dim),
                        "attn": nn.MultiheadAttention(
                            embed_dim, n_heads, dropout=dropout, batch_first=True
                        ),
                        "norm2": nn.LayerNorm(embed_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, glucose: torch.Tensor) -> torch.Tensor:
        """glucose: (batch, n_time_steps) — MinMax-scaled, as it arrives in x[..., 0].

        Returns (batch, n_patches, embed_dim).
        """
        if self.norm_mode == "instance":
            # Per-window z-score. Replaces the PoC's separate StandardScaler:
            # nothing to fit, nothing to plumb through the dataset or checkpoint,
            # and SSL pretraining sees the exact same input distribution for free.
            mean = glucose.mean(dim=1, keepdim=True)
            std = glucose.std(dim=1, keepdim=True).clamp_min(1e-6)
            glucose = (glucose - mean) / std

        x = self.patch_embed(glucose.unsqueeze(1))       # (B, embed_dim, n_patches)
        x = self.pos_enc(x.transpose(1, 2))              # (B, n_patches, embed_dim)
        for blk in self.blocks:
            h = blk["norm1"](x)
            attn_out, _ = blk["attn"](h, h, h, need_weights=False)
            x = x + attn_out
            x = x + blk["mlp"](blk["norm2"](x))
        return self.norm_out(x)


class JepaBranch(nn.Module):
    """JepaEncoder + the projection each fusion route needs."""

    def __init__(
        self,
        n_time_steps: int,
        d_model: int,
        inject: str = "concat",
        reduce_dim: int = 16,
        **encoder_kwargs,
    ):
        super().__init__()
        self.inject = inject
        self.encoder = JepaEncoder(n_time_steps, **encoder_kwargs)
        e = self.encoder.embed_dim

        if inject == "cross":
            # sequence of patch embeddings as K/V for cross-attention
            self.proj = nn.Linear(e, d_model)
            self.out_dim = d_model
        else:
            # concat / film: mean-pool to one vector, then reduce + rescale.
            # The guide is explicit that 96 raw dims will swamp a 32-wide model.
            self.proj = nn.Linear(e, reduce_dim)
            self.scale = nn.LayerNorm(reduce_dim)  # keeps z in the same range as the rest
            self.out_dim = reduce_dim

    def forward(self, glucose: torch.Tensor) -> torch.Tensor:
        h = self.encoder(glucose)                    # (B, n_patches, embed_dim)
        if self.inject == "cross":
            return self.proj(h).permute(1, 0, 2)     # (n_patches, B, d_model) for MHA
        return self.scale(self.proj(h.mean(dim=1)))  # (B, reduce_dim)
```

`CrossAttentionSugarOneBlock` — the only edit to an existing class. The guard is what preserves
`state_dict` key-compatibility:

```python
-    def __init__(self, d_model, n_heads, ff_units, dropout=0.1):
+    def __init__(self, d_model, n_heads, ff_units, dropout=0.1, use_jepa=False):
         self.attn_basal = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
         self.attn_bolus = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
         self.attn_carbs = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
+        self.use_jepa = use_jepa
+        if use_jepa:
+            self.attn_jepa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

-        self.mix_logits = nn.Parameter(torch.zeros(3))
+        # (3,) when use_jepa=False — byte-identical to today's checkpoints.
+        self.mix_logits = nn.Parameter(torch.zeros(4 if use_jepa else 3))

-    def forward(self, glucose, basal, bolus, carbs):
+    def forward(self, glucose, basal, bolus, carbs, jepa=None):
         out_basal, _ = self.attn_basal(glucose, basal, basal)
         out_bolus, _ = self.attn_bolus(glucose, bolus, bolus)
         out_carbs, _ = self.attn_carbs(glucose, carbs, carbs)
+        outs = [out_basal, out_bolus, out_carbs]
+        if self.use_jepa:
+            out_jepa, _ = self.attn_jepa(glucose, jepa, jepa)   # 16 K/V positions vs 128 queries
+            outs.append(out_jepa)

-        res_basal = self.ln1(glucose + self.dropout(out_basal))
-        ...
-        merged = w[0] * res_basal + w[1] * res_bolus + w[2] * res_carbs
+        res = [self.ln1(glucose + self.dropout(o)) for o in outs]
+        w = F.softmax(self.mix_logits, dim=0)
+        merged = sum(w[i] * res[i] for i in range(len(res)))
```

`SugarOneParallelBlock.forward` gains a pass-through `jepa=None` argument. `SugarOneModel`:

```python
     def __init__(self, n_time_steps, n_features=4, d_model=32, n_heads=4,
                  ff_units=128, n_blocks=3, prediction_horizon=12, dropout=0.1,
+                 use_jepa=False, jepa_inject="concat", jepa_patch_size=8,
+                 jepa_embed_dim=96, jepa_layers=3, jepa_heads=6,
+                 jepa_reduce_dim=16, jepa_norm="instance"):
         ...
+        self.use_jepa = use_jepa
+        self.jepa_inject = jepa_inject if use_jepa else None
+        self.jepa_enabled = True     # runtime ablation switch, see evaluate_model.py
+        if use_jepa:
+            self.jepa = JepaBranch(
+                n_time_steps=n_time_steps, d_model=d_model, inject=jepa_inject,
+                reduce_dim=jepa_reduce_dim, patch_size=jepa_patch_size,
+                embed_dim=jepa_embed_dim, n_layers=jepa_layers,
+                n_heads=jepa_heads, dropout=dropout, norm=jepa_norm,
+            )

         self.flatten_fc = nn.Linear(d_model * n_time_steps, d_model)
-        self.out_fc = nn.Linear(d_model, prediction_horizon)
+        head_in = d_model + (self.jepa.out_dim if use_jepa and jepa_inject == "concat" else 0)
+        self.out_fc = nn.Linear(head_in, prediction_horizon)

     def forward(self, x):            # signature UNCHANGED — dataset still yields (x, y)
         ...
+        jepa_kv, z = None, None
+        if self.use_jepa and self.jepa_enabled:
+            out = self.jepa(x[..., 0])                     # reads the same 128-step window
+            if self.jepa_inject == "cross":
+                jepa_kv = out                              # (16, B, d_model)
+            else:
+                z = out                                    # (B, reduce_dim)

         out = g_e
         for block in self.blocks:
-            out = block(out, b_e, bo_e, c_e)
+            out = block(out, b_e, bo_e, c_e, jepa_kv)
         ...
         out = self.dropout(F.gelu(self.flatten_fc(out)))   # h : (B, d_model)
+        if z is not None:
+            out = torch.cat([out, z], dim=-1)              # (B, d_model + reduce_dim)
         return self.out_fc(out)
```

Ablation (`self.jepa_enabled = False`) has a trap in **each** route, and both must be handled or the
ablation is quietly wrong rather than loudly broken:

- Under `cross`, dropping the stream must renormalize the softmax over the three non-JEPA logits.
  Otherwise `w[3]` still multiplies `ln1(glucose + 0)` — a nonzero term that shrinks the other three
  weights, so "ablating" the branch also perturbs basal/bolus/carbs:

  ```python
          logits = self.mix_logits[:3] if (self.use_jepa and jepa is None) else self.mix_logits
          w = F.softmax(logits, dim=0)
  ```

- Under `concat`, `out_fc` was built with `in_features = d_model + reduce_dim`, so skipping `z` entirely
  makes the head's input the wrong width and the forward pass raises. The branch must be zeroed, not
  omitted:

  ```python
          if self.use_jepa and self.jepa_inject == "concat":
              z = self.jepa(x[..., 0]) if self.jepa_enabled else out.new_zeros(out.size(0), self.jepa.out_dim)
              out = torch.cat([out, z], dim=-1)
  ```

## A.2 `scripts/sugar_one/train_sugar_one.py`

`make_model` (line 666) and `_model_kwargs` (line 1047) — every new key uses `cfg.get(...)` with a
default, so **old `config.json` files resume unchanged**:

```python
 def make_model(input_steps, d_model, n_heads, ff_units, n_blocks, horizon,
-               dropout, compile_mode, device) -> SugarOneModel:
+               dropout, compile_mode, device, use_jepa=False, jepa_inject="concat",
+               jepa_patch_size=8, jepa_embed_dim=96, jepa_layers=3, jepa_heads=6,
+               jepa_reduce_dim=16, jepa_norm="instance", jepa_init="") -> SugarOneModel:
     model = SugarOneModel(
         n_time_steps=input_steps, n_features=N_FEATURES, d_model=d_model,
         n_heads=n_heads, ff_units=ff_units, n_blocks=n_blocks,
-        prediction_horizon=horizon, dropout=dropout,
+        prediction_horizon=horizon, dropout=dropout,
+        use_jepa=use_jepa, jepa_inject=jepa_inject, jepa_patch_size=jepa_patch_size,
+        jepa_embed_dim=jepa_embed_dim, jepa_layers=jepa_layers, jepa_heads=jepa_heads,
+        jepa_reduce_dim=jepa_reduce_dim, jepa_norm=jepa_norm,
     ).to(device)
+    if use_jepa and jepa_init:
+        # phase 3: initialize the encoder from SSL pretraining. Weights only —
+        # the fusion projections stay randomly initialized.
+        state = torch.load(jepa_init, map_location=device, weights_only=True)
+        model.jepa.encoder.load_state_dict(state, strict=True)   # raises on any shape/key mismatch
+        typer.echo(f"Loaded JEPA encoder init from {jepa_init}")
     if device.type == "cuda" and compile_mode != "none":
```

`make_optimizer_and_scheduler` (line 491) — the two-group split, so the encoder trains at its own smaller
LR (it is **never frozen**):

```python
-def make_optimizer_and_scheduler(model, lr, weight_decay, epochs):
-    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
+def make_optimizer_and_scheduler(model, lr, weight_decay, epochs, jepa_lr=None):
+    if jepa_lr is not None and getattr(model, "use_jepa", False):
+        jepa_ids = {id(p) for p in model.jepa.encoder.parameters()}
+        groups = [
+            {"params": [p for p in model.parameters() if id(p) not in jepa_ids], "lr": lr},
+            {"params": [p for p in model.parameters() if id(p) in jepa_ids], "lr": jepa_lr},
+        ]
+        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
+    else:
+        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
```

Call site (line 791): `make_optimizer_and_scheduler(model, cfg["lr"], cfg["weight_decay"], cfg["epochs"], cfg.get("jepa_lr"))`.

CLI (`main`, line 1065) and the `cfg` dict (line 1158) gain:

```python
    use_jepa: bool = typer.Option(False, help="Add the JEPA glucose-embedding branch."),
    jepa_inject: str = typer.Option("concat", help="concat | film | cross — where the branch fuses."),
    jepa_patch_size: int = typer.Option(8, help="Steps per patch; input_steps must divide by it."),
    jepa_embed_dim: int = typer.Option(96, help="JEPA encoder width."),
    jepa_layers: int = typer.Option(3, help="JEPA encoder blocks."),
    jepa_heads: int = typer.Option(6, help="JEPA encoder attention heads."),
    jepa_reduce_dim: int = typer.Option(16, help="Pooled embedding size for concat/film (try 8/16/32)."),
    jepa_norm: str = typer.Option("instance", help="instance | none — per-window z-score."),
    jepa_lr: float = typer.Option(4e-5, help="LR for the JEPA encoder's own param group."),
    jepa_init: str = typer.Option("", help="Path to an SSL-pretrained encoder.pt (empty = random init)."),
```

Diagnostic, logged at every validation — the PoC never recorded it and it is the cheapest signal we get
on whether the branch is doing anything:

```python
    if getattr(model, "use_jepa", False) and model.jepa_inject == "cross":
        w = F.softmax(model.blocks[0].cross_attn.mix_logits, dim=0)
        echo_plain(f"  mix weights (basal/bolus/carbs/jepa): {[round(v, 3) for v in w.tolist()]}")
```

Dataset, collate, imputation, training loop, the four training modes: **unchanged**.

## A.3 `scripts/sugar_one/evaluate_model.py`

`_build_model` (line 392) reads the new keys from the run's meta, defaulting to off:

```python
     if model_kind == "glumind":
         return GluMindModel(n_features=3, **common)
-    return SugarOneModel(n_features=4, **common)
+    return SugarOneModel(
+        n_features=4, **common,
+        use_jepa=meta.get("use_jepa", False),
+        jepa_inject=meta.get("jepa_inject", "concat"),
+        jepa_patch_size=meta.get("jepa_patch_size", 8),
+        jepa_embed_dim=meta.get("jepa_embed_dim", 96),
+        jepa_layers=meta.get("jepa_layers", 3),
+        jepa_heads=meta.get("jepa_heads", 6),
+        jepa_reduce_dim=meta.get("jepa_reduce_dim", 16),
+        jepa_norm=meta.get("jepa_norm", "instance"),
+    )
```

`_detect_model_kind` needs no change — it keys off `embed_basal.weight` / `embed_bolus.weight`, which are
present either way.

**Correction to the earlier draft of this plan:** it said to add `"jepa"` to `SUGAR_ONE_COVARIATES` so
`--zero-cov jepa` would ablate the branch. That is wrong. `SUGAR_ONE_COVARIATES` maps canonical names to
*CSV column names*, and the zeroing machinery (`_zero_covariates`) zeroes DataFrame columns — but the JEPA
stream is derived from glucose, which has no column of its own to zero. Zeroing glucose would destroy the
main input too. The branch needs a model-level switch instead:

```python
    ablate_jepa: bool = typer.Option(False, "--ablate-jepa",
        help="Disable the JEPA branch at inference (weights loaded, stream ignored)."),
    ...
    if ablate_jepa:
        model.jepa_enabled = False
```

Since the forward signature and the dataset are unchanged, `_run_evaluate` needs no edits at all.

## A.4 `tests/test_sugar_one_jepa.py` *(new)*

```python
def test_jepa_off_state_dict_is_unchanged():
    """The load-bearing test: every existing SugarOne checkpoint must still load."""
    state = torch.load("test_model_sugar_one/best_model.pt", weights_only=True)
    meta = json.loads(Path("test_model_sugar_one/tuning_meta.json").read_text())
    model = SugarOneModel(n_time_steps=meta["input_steps"], d_model=meta["d_model"], ...)
    model.load_state_dict(strip_compile_prefix(state), strict=True)   # must not raise
    assert model.blocks[0].cross_attn.mix_logits.shape == (3,)

@pytest.mark.parametrize("inject", ["concat", "cross"])
def test_forward_shapes(inject):
    m = SugarOneModel(n_time_steps=128, prediction_horizon=12, use_jepa=True, jepa_inject=inject)
    assert m(torch.randn(4, 128, 4)).shape == (4, 12)          # forward still takes x alone

def test_patch_size_must_divide_input_steps():
    with pytest.raises(ValueError, match="divisible"):
        SugarOneModel(n_time_steps=128, use_jepa=True, jepa_patch_size=12)   # the CGM-JEPA default

def test_optimizer_has_two_groups():
    m = SugarOneModel(n_time_steps=128, use_jepa=True)
    opt, _ = make_optimizer_and_scheduler(m, lr=4e-4, weight_decay=3e-5, epochs=10, jepa_lr=4e-5)
    assert [g["lr"] for g in opt.param_groups] == [4e-4, 4e-5]
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(m.parameters()))

def test_ablation_renormalizes_the_mix():
    m = SugarOneModel(n_time_steps=128, use_jepa=True, jepa_inject="cross")
    m.jepa_enabled = False
    assert m(torch.randn(4, 128, 4)).shape == (4, 12)
```

## A.5 What this buys, mechanically

- Every series ≥ 140 steps now produces windows (was ≥ 300), so phases 0–3 are finally scored on the
  same data. That alone invalidates the PoC's headline comparison and is the main point of the change.
- The dataset still yields `(x, y)`, so `evaluate-model`, the four training modes, `tune_sugar_one.py`,
  and the checkpoint format all keep working with no changes.
- Added parameters at `d_model=32, patch=8, embed_dim=96, layers=3`: roughly 340k in the encoder plus
  ~1.5k for the fusion — comparable to the SugarOne backbone, which is exactly why the capacity-matched
  control in phase 1 is not optional.

## A.6 Commands

```bash
# smoke (CPU, seconds)
uv run pytest tests/test_sugar_one_jepa.py -q

# phase 1 — concat route, random init (the guide's recommended starting rung)
uv run python scripts/sugar_one/train_sugar_one.py \
  --csv data/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv --device cuda \
  --d-model 32 --n-heads 8 --n-blocks 5 --ff-units 128 --input-steps 128 --horizon 12 \
  --lr 0.0004 --weight-decay 0.00003 --batch-size 256 --epochs 30 --patience 3 \
  --use-jepa --jepa-inject concat --jepa-patch-size 8 --jepa-reduce-dim 16 --jepa-lr 4e-5 \
  --out-dir runs/sugar_one_jepa

# same run, cross-attention route (the PoC's fusion, now at 128 steps)
  ... --use-jepa --jepa-inject cross

# evaluate, and ablate the branch on the same checkpoint
uv run evaluate-model --run-dir runs/sugar_one_jepa/<run> --model-type sugar_one \
  --test-csv data/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv
uv run evaluate-model --run-dir runs/sugar_one_jepa/<run> --model-type sugar_one \
  --test-csv data/loop_and_ai_ready/loop_ai_ready_joined2_dev.csv --ablate-jepa
```
