#!/usr/bin/env python3
"""Legacy argparse entry point for GluMind training.

Reusable training, evaluation, model-factory, and mode-runner functions live
in :mod:`glucose_forecasting.training.glumind` and are re-exported here for
backward compatibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from glucose_forecasting.training.glumind import *  # noqa: F403


def parse_args() -> argparse.Namespace:
    """Parse the unchanged legacy GluMind command-line interface."""
    ap = argparse.ArgumentParser(
        description="GluMind: Multimodal Parallel-Attention Transformer for Blood Glucose Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", type=Path, required=True, help="Path to processed dataset CSV.")
    ap.add_argument("--unique_id", choices=["sequence_id", "user_id"], default="sequence_id")
    ap.add_argument("--chunk_size", type=int, default=1_000_000)
    ap.add_argument("--max_train_series", type=int, default=0, help="Limit training series (0 = all).")
    ap.add_argument("--max_eval_series", type=int, default=0, help="Limit evaluation series (0 = all).")
    ap.add_argument("--drop_interpolated", action="store_true")
    ap.add_argument("--mask_interpolated_targets", action="store_true")
    ap.add_argument(
        "--study_groups",
        type=str,
        default="",
        help="Comma-separated list of Study Group values. Empty = all groups.",
    )
    ap.add_argument(
        "--split_scheme",
        choices=["classic", "trainval_test_as_val"],
        default="classic",
        help=(
            "Data split policy. 'classic' uses Recommended Split as-is. "
            "'trainval_test_as_val' merges train+val for training and uses test as validation (test eval is disabled)."
        ),
    )
    ap.add_argument(
        "--mode",
        choices=["global", "per_group", "cohort_wise", "continual"],
        default="global",
    )
    ap.add_argument("--horizon", type=int, default=12, help="Prediction horizon in steps (12=60min, 6=30min, 1=5min).")
    ap.add_argument("--input_steps", type=int, default=80, help="Input window in steps (80 = 400 min at 5-min freq).")
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--ff_units", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16", help="Mixed precision mode on CUDA.")
    ap.add_argument(
        "--compile_mode",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        default="none",
        help="Enable torch.compile for model graph optimization.",
    )
    ap.add_argument("--disable_tf32", action="store_true", help="Disable TF32 on CUDA matmul/cuDNN.")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto; cuda->cpu_count/2 capped at 8).",
    )
    ap.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="DataLoader prefetch factor (only when num_workers>0).",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience (0 = disabled).")
    ap.add_argument("--log_every", type=int, default=10, help="Print loss every N epochs.")
    ap.add_argument(
        "--ckpt_every_n_epochs",
        type=int,
        default=0,
        help="Save checkpoint + run full eval every N epochs (0 = disabled). Results saved to checkpoints/epoch_NNNN/ subdirs.",
    )
    ap.add_argument("--val_every_n_epochs", type=int, default=1, help="Run validation every N epochs (1 = every epoch).")
    ap.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to a checkpoint.pt file to resume training from. Restores model, optimizer, scheduler, and epoch number.",
    )
    ap.add_argument("--lwf_lambda", type=float, default=0.5, help="LwF distillation weight for continual mode.")
    ap.add_argument(
        "--continual_order",
        type=str,
        choices=["default", "reverse"],
        default="default",
        help="Group order in continual mode: 'default' follows Healthy->Pre-T2DM->Oral-T2DM->Insulin-T2DM->T1DM; 'reverse' runs the opposite order.",
    )
    ap.add_argument(
        "--continual_val_scope",
        type=str,
        choices=["current_group", "all_groups"],
        default="current_group",
        help="Validation set scope in continual mode: 'current_group' validates only on the active cohort; 'all_groups' validates on the full validation split each step.",
    )
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=Path("data/output/runs/glumind"))
    ap.add_argument("--save_predictions", action="store_true")
    return ap.parse_args()


def main() -> None:
    """Run legacy GluMind CLI data preparation and mode dispatch."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        if not args.disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled.")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")
    train_df, val_df, test_df = load_splits_streaming(args.csv, args.unique_id, args.drop_interpolated)  # noqa: F405
    print(f"Loaded: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    train_df = normalize_study_groups_column(train_df)  # noqa: F405
    val_df = normalize_study_groups_column(val_df)  # noqa: F405
    test_df = normalize_study_groups_column(test_df)  # noqa: F405
    if args.study_groups:
        groups = [normalize_study_group_label(group.strip()) for group in args.study_groups.split(",") if group.strip()]  # noqa: F405
        train_df = train_df.filter(pl.col("study_group").is_in(groups))
        val_df = val_df.filter(pl.col("study_group").is_in(groups))
        test_df = test_df.filter(pl.col("study_group").is_in(groups))
        print(f"Filtered to groups {groups}: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    train_df, val_df, test_df = apply_split_scheme(train_df, val_df, test_df, args.split_scheme)  # noqa: F405
    train_df = impute_and_sort(train_df)  # noqa: F405
    val_df = impute_and_sort(val_df)  # noqa: F405
    test_df = impute_and_sort(test_df)  # noqa: F405
    if args.max_train_series > 0:
        train_df = limit_series(train_df, args.max_train_series)  # noqa: F405
    if args.max_eval_series > 0:
        val_df = limit_series(val_df, args.max_eval_series)  # noqa: F405
        test_df = limit_series(test_df, args.max_eval_series)  # noqa: F405
    print(f"After limits: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}")
    print(f"Study groups in train: {sorted(train_df['study_group'].unique().to_list())}")
    {
        "global": mode_global,  # noqa: F405
        "per_group": mode_per_group,  # noqa: F405
        "cohort_wise": mode_cohort_wise,  # noqa: F405
        "continual": mode_continual,  # noqa: F405
    }[args.mode](train_df, val_df, test_df, args, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
