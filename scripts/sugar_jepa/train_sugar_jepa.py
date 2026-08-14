#!/usr/bin/env python3
"""
SugarJepa — SugarOne + our own JEPA glucose encoder as a 4th cross-attention
auxiliary stream. See scripts/sugar_jepa/sugar_jepa_model.py and
scripts/sugar_jepa/README.md.

Dataset: data/input/loop_ai_ready_joined2_dev.csv (or the full
loop_ai_ready_joined2.csv).

Two windows: the backbone reads `--input-steps` steps, the JEPA branch reads
`--jepa-window` (default 288 = 24h), and both end at the same instant. The
dataset builds ONE window of `max(input_steps, jepa_window)` steps and the model
slices each view out of it, so the batch stays SugarOne's plain `(x, y)` and
every model-agnostic piece below is still imported from train_sugar_one.py
rather than reimplemented. `--jepa-window` must match the `--window` the encoder
passed to `--jepa-init` was pretrained at.

Proof-of-concept scope: `global` mode only (one model, all study groups) —
per_group / cohort_wise / continual (LwF) from train_sugar_one.py are not
implemented here.

Imputation policy (identical to SugarOne):
  - Basal Rate: forward-fill then back-fill, then fill_null(0.0).
  - Bolus Insulin / Carbohydrates: fill_null(0.0) directly (discrete events).
  - Glucose: forward-fill then back-fill then fill_null(0.0).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

from scripts.sugar_one.console_log import echo_plain
from scripts.sugar_jepa.sugar_jepa_model import SugarJepaModel2

from scripts.common.data_loading import (
    limit_series,
    normalize_study_group_label,
    normalize_study_groups_column,
    resolve_num_workers,
)

# --- everything model-agnostic comes from SugarOne, unchanged ----------------
from scripts.sugar_one.train_sugar_one import (
    N_FEATURES,
    SugarOneWindowDataset,
    apply_split_scheme,
    build_datasets,
    compute_and_print_metrics,
    evaluate,
    impute_and_sort,
    load_full_checkpoint,
    load_splits_streaming,
    train_loop,
    update_latest_symlink,
)

# The dataset is now literally SugarOne's. Alias kept so existing imports of
# `SugarJepaWindowDataset` keep resolving.
SugarJepaWindowDataset = SugarOneWindowDataset

app = typer.Typer(
    name="train_sugar_jepa",
    add_completion=False,
    help="SugarJepa: SugarOne + our own JEPA glucose-embedding stream.",
)


# ============================================================================
#  MODEL-FACING PIECES — the only things that differ from SugarOne
# ============================================================================

def _jepa_encoder(model: SugarJepaModel2) -> nn.Module:
    """The JEPA encoder submodule, unwrapping torch.compile if present."""
    m = getattr(model, "_orig_mod", model)
    return m.jepa_encoder


def make_model(cfg: dict, device: torch.device) -> SugarJepaModel2:
    model = SugarJepaModel2(
        n_time_steps=cfg["input_steps"],
        n_features=N_FEATURES,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        ff_units=cfg["ff_units"],
        n_blocks=cfg["n_blocks"],
        prediction_horizon=cfg["horizon"],
        dropout=cfg["dropout"],
        jepa_window=cfg["jepa_window"],
        jepa_patch_size=cfg["jepa_patch_size"],
        jepa_embed_dim=cfg["jepa_embed_dim"],
        jepa_layers=cfg["jepa_layers"],
        jepa_heads=cfg["jepa_heads"],
        jepa_norm=cfg["jepa_norm"],
    ).to(device)

    if cfg["jepa_init"]:
        # Start the encoder from self-supervised pretraining. Encoder weights
        # only — jepa_proj and the forecasting backbone stay randomly initialised.
        state = torch.load(cfg["jepa_init"], map_location=device, weights_only=True)
        _jepa_encoder(model).load_state_dict(state, strict=True)
        typer.echo(f"JEPA encoder initialised from {cfg['jepa_init']}")
    else:
        typer.echo("JEPA encoder randomly initialised (no --jepa-init).")

    if device.type == "cuda" and cfg["compile_mode"] != "none":
        model = torch.compile(model, mode=cfg["compile_mode"])
        typer.echo(f"torch.compile enabled (mode={cfg['compile_mode']})")
    return model


def make_optimizer_and_scheduler(
    model: SugarJepaModel2,
    cfg: dict,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Two param groups: the pretrained-or-random JEPA encoder gets a smaller LR
    than the freshly-initialised forecasting model around it. The encoder always
    trains — there is no frozen mode."""
    jepa_ids = {id(p) for p in _jepa_encoder(model).parameters()}
    jepa_all = [p for p in model.parameters() if id(p) in jepa_ids]
    other_all = [p for p in model.parameters() if id(p) not in jepa_ids]
    jepa_params = [p for p in jepa_all if p.requires_grad]
    other_params = [p for p in other_all if p.requires_grad]
    # A silent empty group here means every param trains at the wrong LR.
    if not jepa_params or not other_params:
        raise RuntimeError(
            f"Optimizer param split is degenerate: {len(jepa_params)} JEPA tensors, "
            f"{len(other_params)} others — check _jepa_encoder()."
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg["lr"]},
            {"params": jepa_params, "lr": cfg["jepa_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    def _n(params: list[torch.nn.Parameter]) -> int:
        return sum(p.numel() for p in params)

    echo_plain(
        f"  Training {_n(other_params):,} out of {_n(other_all):,} SugarOne params "
        f"@ lr={cfg['lr']}"
    )
    echo_plain(
        f"  Training {_n(jepa_params):,} out of {_n(jepa_all):,} JEPA params "
        f"@ lr={cfg['jepa_lr']}"
    )
    return optimizer, scheduler


def _mix_weights(model: SugarJepaModel2) -> dict[str, float]:
    """Mean softmax mix weight per auxiliary, averaged over blocks — logged to the
    metrics CSV every epoch so the JEPA stream's weight is a curve, not one number
    at the end."""
    m = getattr(model, "_orig_mod", model)
    w = torch.stack(
        [torch.softmax(b.cross_attn.mix_logits.detach().float(), dim=0) for b in m.blocks]
    ).mean(dim=0)
    names = ("mix_basal", "mix_bolus", "mix_carbs", "mix_jepa")
    return {n: round(v, 4) for n, v in zip(names, w.tolist())}


def _log_mix_weights(model: SugarJepaModel2) -> None:
    """How much weight each block's softmax mix gives the JEPA stream.

    If the jepa column trends to ~0 the model is ignoring the branch, which is
    the cheapest signal we get that the architecture is not earning its keep.
    """
    m = getattr(model, "_orig_mod", model)
    echo_plain("  Learned mix weights (basal / bolus / carbs / jepa):")
    for i, block in enumerate(m.blocks):
        w = torch.softmax(block.cross_attn.mix_logits.detach().float(), dim=0)
        echo_plain(f"    block {i}: {[round(v, 3) for v in w.tolist()]}")


def run_train_and_eval(
    model: SugarJepaModel2,
    train_ds: SugarOneWindowDataset,
    val_ds: SugarOneWindowDataset | None,
    test_ds: SugarOneWindowDataset | None,
    cfg: dict,
    device: torch.device,
    run_name: str,
    out_dir: Path,
) -> SugarJepaModel2:
    """Same as SugarOne's, with our optimizer factory wired in."""
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Run directory: {run_dir}")

    meta = dict(cfg)
    meta.update({
        "model_type": "sugar_jepa",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds else 0,
        "test_samples": len(test_ds) if test_ds else 0,
        "start_time": datetime.now().isoformat(),
    })
    with open(run_dir / "tuning_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    update_latest_symlink(run_dir, out_dir)

    num_workers = resolve_num_workers(cfg["num_workers"], device)
    loader_kwargs: dict = dict(
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg["prefetch_factor"]

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        if val_ds is not None and len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        if test_ds is not None and len(test_ds) > 0
        else None
    )

    echo_plain(
        f"  DataLoader: train_batches/epoch={len(train_loader):,} | "
        f"val_batches={len(val_loader) if val_loader else 0:,} | "
        f"test_batches={len(test_loader) if test_loader else 0:,} | "
        f"batch_size={cfg['batch_size']} | num_workers={num_workers}"
    )

    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg)
    loss_fn = nn.MSELoss()
    use_amp = device.type == "cuda" and cfg["precision"] in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and cfg["precision"] == "fp16")
    )

    start_epoch = 1
    best_val_loss = float("inf")
    start_wait = 0
    start_best_epoch = 0
    if cfg.get("resume_from"):
        resume_path = Path(cfg["resume_from"])
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        last_done, best_val_loss, start_wait, start_best_epoch = load_full_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        start_epoch = last_done + 1
        if start_epoch > cfg["epochs"]:
            echo_plain(
                f"  Checkpoint already at epoch {last_done} >= max {cfg['epochs']}; "
                "skipping training loop."
            )
            start_epoch = cfg["epochs"] + 1

    typer.echo(f"\n{'=' * 60}")
    typer.echo(
        f"Training: {len(train_ds):,} windows | "
        f"Val: {len(val_ds) if val_ds else 0:,} | "
        f"Test: {len(test_ds) if test_ds else 0:,} | "
        f"Params: {sum(p.numel() for p in model.parameters()):,}"
    )
    typer.echo(f"{'=' * 60}")

    model = train_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        loss_fn, device, cfg["epochs"], cfg["patience"], run_dir, cfg,
        verbose_every=cfg["log_every"],
        ckpt_every_n_epochs=cfg["ckpt_every_n_epochs"],
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        start_wait=start_wait,
        start_best_epoch=start_best_epoch,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        scaler=grad_scaler,
        val_every_n_epochs=cfg["val_every_n_epochs"],
        batch_log_every=int(cfg.get("batch_log_every", 0)),
        eval_batch_log_every=int(cfg.get("eval_batch_log_every", 0)),
        metrics_csv=run_dir / "training_metrics.csv",
        extra_metrics_fn=lambda: _mix_weights(model),
    )

    _log_mix_weights(model)

    if val_loader is not None:
        _, vt, vp = evaluate(
            model, val_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype,
            batch_log_every=int(cfg.get("eval_batch_log_every", 0)), split_label="val",
        )
        compute_and_print_metrics(vt, vp, train_ds.scaler_glucose, "val", run_dir, val_ds)

    if test_loader is not None:
        _, tt, tp = evaluate(
            model, test_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype,
            batch_log_every=int(cfg.get("eval_batch_log_every", 0)), split_label="test",
        )
        compute_and_print_metrics(tt, tp, train_ds.scaler_glucose, "test", run_dir, test_ds)

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return model


def _mode_global(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> None:
    typer.echo("\n=== MODE: GLOBAL ===")
    # One window per sample, long enough for both views; the model slices the
    # backbone's shorter one out of it (see SugarJepaModel2.forward).
    lookback = max(cfg["input_steps"], cfg["jepa_window"])
    if lookback > cfg["input_steps"]:
        typer.echo(
            f"  Lookback {lookback} steps (jepa_window={cfg['jepa_window']} > "
            f"input_steps={cfg['input_steps']}) — series shorter than "
            f"{lookback + cfg['horizon']} rows contribute no windows."
        )
    train_ds, val_ds, test_ds = build_datasets(
        train_df, val_df, test_df, lookback, cfg["horizon"]
    )
    # An empty training set here is silent otherwise: the loader yields no
    # batches and every epoch reports a loss of 0.0. Easy to hit now that
    # --jepa-window drives the lookback well past --input-steps.
    if len(train_ds) == 0:
        raise typer.BadParameter(
            f"No training windows: every series is shorter than "
            f"{lookback + cfg['horizon']} rows (lookback {lookback} + horizon "
            f"{cfg['horizon']}). Lower --jepa-window, or train on longer series."
        )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"sugar_jepa_global_h{cfg['horizon']}_{ts}"

    model = make_model(cfg, device)
    run_train_and_eval(model, train_ds, val_ds, test_ds, cfg, device, run_name, out_dir)


# ============================================================================
#  CLI
# ============================================================================

@app.command()
def main(
    csv: Path = typer.Option(..., help="Path to loop_ai_ready_joined2*.csv."),
    unique_id: str = typer.Option("sequence_id", help="sequence_id or user_id."),
    max_train_series: int = typer.Option(0, help="Limit training series (0 = all)."),
    max_eval_series: int = typer.Option(0, help="Limit evaluation series (0 = all)."),
    drop_interpolated: bool = typer.Option(False, help="Drop Interpolated rows."),
    study_groups: str = typer.Option("", help="Comma-separated Study Group filter (empty = all)."),
    split_scheme: str = typer.Option("classic", help="classic | trainval_test_as_val."),
    horizon: int = typer.Option(12, help="Prediction horizon steps (12 = 60 min at 5-min freq)."),
    input_steps: int = typer.Option(128, help="Backbone input window steps."),
    d_model: int = typer.Option(32, help="Embedding dimension."),
    n_heads: int = typer.Option(8, help="Attention heads."),
    n_blocks: int = typer.Option(5, help="Parallel transformer blocks."),
    ff_units: int = typer.Option(128, help="FFN hidden units."),
    dropout: float = typer.Option(0.1, help="Dropout rate."),
    epochs: int = typer.Option(30, help="Training epochs."),
    batch_size: int = typer.Option(256, help="Batch size."),
    precision: str = typer.Option("bf16", help="fp32 | bf16 | fp16."),
    compile_mode: str = typer.Option("none", help="none | default | reduce-overhead | max-autotune."),
    disable_tf32: bool = typer.Option(False, help="Disable TF32 on CUDA."),
    num_workers: int = typer.Option(0, help="DataLoader workers (-1 = auto; 0 avoids Windows worker-spawn stalls)."),
    prefetch_factor: int = typer.Option(4, help="DataLoader prefetch factor."),
    lr: float = typer.Option(4e-4, help="Learning rate (everything except the JEPA encoder)."),
    weight_decay: float = typer.Option(3e-5, help="Weight decay."),
    patience: int = typer.Option(3, help="Early stopping patience (0 = disabled)."),
    log_every: int = typer.Option(1, help="Print every N epochs."),
    ckpt_every_n_epochs: int = typer.Option(0, help="Save checkpoint every N epochs (0 = off)."),
    val_every_n_epochs: int = typer.Option(5, help="Run validation every N epochs."),
    resume_from: str = typer.Option("", help="Path to checkpoint.pt to resume from."),
    batch_log_every: int = typer.Option(200, help="Log train progress every N batches (0 = off)."),
    eval_batch_log_every: int = typer.Option(300, help="Log eval progress every N batches (0 = off)."),
    # --- JEPA branch ---------------------------------------------------------
    # The branch fuses as a 4th cross-attention auxiliary (K/V = patch embeddings).
    jepa_window: int = typer.Option(
        288,
        help="Glucose-only lookback for the JEPA branch (288 = 24h at 5-min). Must match the "
             "--window the encoder was pretrained at. Independent of --input-steps; when longer, "
             "the dataset window grows to it and short series drop out.",
    ),
    jepa_patch_size: int = typer.Option(8, help="Steps per JEPA patch; jepa_window must divide by it."),
    jepa_embed_dim: int = typer.Option(96, help="JEPA encoder width."),
    jepa_layers: int = typer.Option(3, help="JEPA encoder blocks."),
    jepa_heads: int = typer.Option(6, help="JEPA encoder attention heads."),
    jepa_norm: str = typer.Option("instance", help="instance | none — per-window z-score inside the encoder."),
    jepa_lr: float = typer.Option(4e-5, help="LR for the JEPA encoder's own param group."),
    jepa_init: str = typer.Option("", help="Path to a self-supervised encoder.pt (empty = random init)."),
    device_name: str = typer.Option("cuda", "--device", help="cpu | mps | cuda."),
    seed: int = typer.Option(42, help="Random seed."),
    out_dir: Path = typer.Option(Path("data/output/runs/sugar_jepa"), help="Output directory."),
) -> None:
    """Train SugarJepa (global mode only) — SugarOne + our own JEPA glucose encoder."""
    # Fail before the (slow) CSV load rather than at model construction.
    if jepa_window % jepa_patch_size != 0:
        raise typer.BadParameter(
            f"--jepa-window ({jepa_window}) must be divisible by "
            f"--jepa-patch-size ({jepa_patch_size})."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device_name == "mps" and not torch.backends.mps.is_available():
        typer.echo("MPS not available, falling back to CPU.")
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        typer.echo("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    if device.type == "cuda":
        if not disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            typer.echo("TF32 enabled.")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    typer.echo(f"Device: {device}")

    train_df, val_df, test_df = load_splits_streaming(csv, unique_id, drop_interpolated)
    typer.echo(f"Loaded: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    train_df = normalize_study_groups_column(train_df)
    val_df = normalize_study_groups_column(val_df)
    test_df = normalize_study_groups_column(test_df)

    if study_groups:
        group_list = [
            normalize_study_group_label(g.strip())
            for g in study_groups.split(",") if g.strip()
        ]
        train_df = train_df.filter(pl.col("study_group").is_in(group_list))
        val_df = val_df.filter(pl.col("study_group").is_in(group_list))
        test_df = test_df.filter(pl.col("study_group").is_in(group_list))
        typer.echo(f"Filtered to {group_list}: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")

    train_df, val_df, test_df = apply_split_scheme(train_df, val_df, test_df, split_scheme)

    if max_train_series > 0:
        train_df = limit_series(train_df, max_train_series)
    if max_eval_series > 0:
        val_df = limit_series(val_df, max_eval_series)
        test_df = limit_series(test_df, max_eval_series)

    train_df = impute_and_sort(train_df)
    val_df = impute_and_sort(val_df)
    test_df = impute_and_sort(test_df)

    typer.echo(f"After limits: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    typer.echo(f"Study groups in train: {sorted(train_df['study_group'].unique().to_list())}")

    cfg = {
        "csv": str(csv), "unique_id": unique_id, "drop_interpolated": drop_interpolated,
        "study_groups": study_groups, "split_scheme": split_scheme, "mode": "global",
        "model_type": "sugar_jepa",
        "horizon": horizon, "input_steps": input_steps,
        "d_model": d_model, "n_heads": n_heads, "n_blocks": n_blocks,
        "ff_units": ff_units, "dropout": dropout,
        "epochs": epochs, "batch_size": batch_size, "precision": precision,
        "compile_mode": compile_mode, "disable_tf32": disable_tf32,
        "num_workers": num_workers, "prefetch_factor": prefetch_factor,
        "lr": lr, "weight_decay": weight_decay, "patience": patience,
        "log_every": log_every, "ckpt_every_n_epochs": ckpt_every_n_epochs,
        "val_every_n_epochs": val_every_n_epochs, "resume_from": resume_from,
        "batch_log_every": batch_log_every, "eval_batch_log_every": eval_batch_log_every,
        "jepa_window": jepa_window,
        "jepa_patch_size": jepa_patch_size, "jepa_embed_dim": jepa_embed_dim,
        "jepa_layers": jepa_layers, "jepa_heads": jepa_heads,
        "jepa_norm": jepa_norm, "jepa_lr": jepa_lr, "jepa_init": jepa_init,
        "device": device_name, "seed": seed, "out_dir": str(out_dir),
    }

    _mode_global(train_df, val_df, test_df, cfg, device, out_dir)
    typer.echo("\nDone.")


if __name__ == "__main__":
    app()
