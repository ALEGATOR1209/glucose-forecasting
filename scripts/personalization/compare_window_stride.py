#!/usr/bin/env python3
"""Compare dense vs sparse train window stride for personalization fine-tuning.

Hypothesis: with input_steps=128 and horizon=12, consecutive sliding windows
overlap heavily. Training on every 6th window (~30 min apart at 5-min sampling)
cuts train windows ~6× while each target timestep still appears in multiple
windows. Expect similar test MAE with much faster epochs.

Val/test always use dense windows (stride=1) for comparable metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from scripts.common.console import init_cli_console, safe_echo
from scripts.personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_SEED,
    DENSE_WINDOW_STRIDE,
    SPARSE_WINDOW_STRIDE,
)
from scripts.personalization.finetune import run_finetune

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _row_from_run(
    *,
    label: str,
    train_window_stride: int,
    run_dir: Path,
    results: dict[str, Any],
) -> dict[str, Any]:
    cfg = results.get("config", {})
    zs = results.get("zero_shot_test") or {}
    ft_test = results.get("finetuned_test") or {}
    return {
        "label": label,
        "train_window_stride": train_window_stride,
        "run_dir": str(run_dir),
        "train_windows": cfg.get("train_windows"),
        "val_windows": cfg.get("val_windows"),
        "test_windows": cfg.get("test_windows"),
        "wall_time_s": results.get("wall_time_s"),
        "zero_shot_test_mae": zs.get("mae"),
        "finetuned_test_mae": ft_test.get("mae"),
        "finetuned_test_rmse": ft_test.get("rmse"),
        "finetuned_test_mard": ft_test.get("mard"),
        "lwf_lambda": cfg.get("lwf_lambda"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
    }


@app.command()
def main(
    base_run_dir: Path = typer.Option(
        Path(DEFAULT_BASE_RUN_DIR),
        "--base-run-dir",
    ),
    personal_csv: Path = typer.Option(..., "--personal-csv"),
    out_dir: Path = typer.Option(
        Path("data/output/runs/personalization/livia/window_stride_compare"),
        "--out-dir",
    ),
    sparse_stride: int = typer.Option(
        SPARSE_WINDOW_STRIDE,
        "--sparse-stride",
        help="Train window stride for sparse run (default: 6 = 30 min).",
    ),
    dense_stride: int = typer.Option(
        DENSE_WINDOW_STRIDE,
        "--dense-stride",
        help="Train window stride for dense baseline (default: 1).",
    ),
    run_sparse: bool = typer.Option(True, "--run-sparse/--no-run-sparse"),
    run_dense: bool = typer.Option(True, "--run-dense/--no-run-dense"),
    lwf_lambda: float = typer.Option(0.3, "--lwf-lambda"),
    lr: Optional[float] = typer.Option(None, "--lr"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay"),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    precision: str = typer.Option("fp32", "--precision"),
) -> None:
    """Run sparse and/or dense window-stride fine-tunes; write comparison JSON."""
    init_cli_console()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    common = dict(
        base_run_dir=base_run_dir,
        personal_csv=personal_csv,
        out_dir=out_dir,
        lwf_lambda=lwf_lambda,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        precision=precision,
        eval_zero_shot=True,
    )

    if run_sparse:
        safe_echo(f"\n===== sparse train windows (stride={sparse_stride}) =====")
        sparse_dir, sparse_results = run_finetune(
            **common,
            run_name=f"sparse_stride{sparse_stride}",
            train_window_stride=sparse_stride,
        )
        rows.append(
            _row_from_run(
                label="sparse",
                train_window_stride=sparse_stride,
                run_dir=sparse_dir,
                results=sparse_results,
            )
        )

    if run_dense:
        safe_echo(f"\n===== dense train windows (stride={dense_stride}) =====")
        dense_dir, dense_results = run_finetune(
            **common,
            run_name=f"dense_stride{dense_stride}",
            train_window_stride=dense_stride,
        )
        rows.append(
            _row_from_run(
                label="dense",
                train_window_stride=dense_stride,
                run_dir=dense_dir,
                results=dense_results,
            )
        )

    comparison = {
        "hypothesis": (
            "Sparse train windows (stride=6, 30 min) reduce epochs ~6× with minimal "
            "test MAE loss because horizon=12 still overlaps neighbouring windows."
        ),
        "sparse_stride": sparse_stride,
        "dense_stride": dense_stride,
        "runs": rows,
    }
    if len(rows) == 2:
        sparse_mae = rows[0].get("finetuned_test_mae")
        dense_mae = rows[1].get("finetuned_test_mae")
        sparse_t = rows[0].get("wall_time_s")
        dense_t = rows[1].get("wall_time_s")
        if sparse_mae is not None and dense_mae is not None:
            comparison["mae_delta_sparse_minus_dense"] = float(sparse_mae) - float(dense_mae)
        if sparse_t is not None and dense_t is not None and sparse_t > 0:
            comparison["wall_time_speedup_dense_over_sparse"] = float(dense_t) / float(sparse_t)

    out_path = out_dir / "window_stride_comparison.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    safe_echo(f"\nComparison written: {out_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
