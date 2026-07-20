#!/usr/bin/env python3
"""Step 2: sweep LwF, learning rate, and weight_decay on full personal train data.

Uses all personal train rows (excluding val/test).

LwF grid is centered on GluMind type-1 best (``lwf_lambda=0.3`` from
``reports/glumind/AI_READY_PLUS_TYPE_1_TUNED_MODELS_RUNS_ANALYSIS.md``).
LR and weight_decay grids use 0.5× / 1× / 2× of base model values.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Optional

import typer

from scripts.common.console import init_cli_console, safe_echo
from scripts.personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_FT_PATIENCE,
    DEFAULT_LWF_LAMBDAS,
    DEFAULT_LR_MULTIPLIERS,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY_MULTIPLIERS,
    GLUMIND_BEST_LWF_TYPE1,
)
from scripts.personalization.finetune import run_finetune
from scripts.personalization.sweep_utils import (
    flatten_metrics,
    lr_grid_from_base,
    pick_best_row,
    weight_decay_grid,
    write_best_recipe,
    write_summary,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _parse_floats(raw: str | None, default: tuple[float, ...]) -> list[float]:
    if raw is None:
        return list(default)
    return [float(p.strip()) for p in raw.split(",") if p.strip()]


@app.command()
def main(
    base_run_dir: Path = typer.Option(
        Path(DEFAULT_BASE_RUN_DIR),
        "--base-run-dir",
    ),
    personal_csv: Path = typer.Option(..., "--personal-csv"),
    out_dir: Path = typer.Option(
        Path("data/output/runs/personalization/livia/sweeps/hyperparams"),
        "--out-dir",
    ),
    lwf_lambdas: Optional[str] = typer.Option(
        None,
        "--lwf-lambdas",
        help="Comma-separated LwF weights (default: 0.2,0.25,0.3,0.35 around GluMind type-1 best).",
    ),
    lr_multipliers: Optional[str] = typer.Option(
        None,
        "--lr-multipliers",
        help="Comma-separated multipliers of base model lr, e.g. '0.5,1,2'.",
    ),
    weight_decay_multipliers: Optional[str] = typer.Option(
        None,
        "--weight-decay-multipliers",
        help="Comma-separated multipliers of 3e-5, e.g. '0.5,1,2'.",
    ),
    epochs: int = typer.Option(30, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    subject: str = typer.Option("livia", "--subject"),
) -> None:
    """Grid over LwF × LR × weight_decay on full personal train data."""
    init_cli_console()
    lwf_grid = _parse_floats(lwf_lambdas, DEFAULT_LWF_LAMBDAS)
    lr_mults = tuple(_parse_floats(lr_multipliers, DEFAULT_LR_MULTIPLIERS))
    wd_mults = tuple(_parse_floats(weight_decay_multipliers, DEFAULT_WEIGHT_DECAY_MULTIPLIERS))
    lr_grid = lr_grid_from_base(base_run_dir, multipliers=lr_mults)
    wd_grid = weight_decay_grid(wd_mults)
    patience = DEFAULT_FT_PATIENCE

    safe_echo(f"GluMind type-1 LwF starting point: {GLUMIND_BEST_LWF_TYPE1}")
    safe_echo(f"LwF grid: {lwf_grid}")
    safe_echo(f"LR grid: {lr_grid}")
    safe_echo(f"weight_decay grid: {wd_grid}")
    safe_echo(f"Patience: {patience}")

    rows: list[dict] = []
    for lwf, lr, wd in itertools.product(lwf_grid, lr_grid, wd_grid):
        label = f"lwf{lwf:g}_lr{lr:g}_wd{wd:g}"
        safe_echo(f"\n===== hyperparams {label} (full train data) =====")
        try:
            run_dir, results = run_finetune(
                base_run_dir=base_run_dir,
                personal_csv=personal_csv,
                out_dir=out_dir / label,
                run_name=f"{subject}_{label}",
                personal_days=None,
                lwf_lambda=lwf,
                epochs=epochs,
                lr=lr,
                weight_decay=wd,
                patience=patience,
                batch_size=batch_size,
                seed=seed,
                device=device,
                eval_zero_shot=True,
            )
        except ValueError as exc:
            safe_echo(f"Skipping {label}: {exc}", err=True)
            rows.append(
                {
                    "subject": subject,
                    "personal_days": "all",
                    "lwf_lambda": lwf,
                    "lr": lr,
                    "weight_decay": wd,
                    "status": "skipped",
                    "error": str(exc),
                }
            )
            continue

        row = {
            "subject": subject,
            "personal_days": "all",
            "lwf_lambda": lwf,
            "lr": lr,
            "weight_decay": wd,
            "patience": patience,
            "epochs": epochs,
            "run_dir": str(run_dir),
            "status": "ok",
            **flatten_metrics("zs_test", results.get("zero_shot_test")),
            **flatten_metrics("ft_test", results.get("finetuned_test")),
            **flatten_metrics("ft_val", results.get("finetuned_val")),
        }
        rows.append(row)

    summary_path = write_summary(rows, out_dir)
    safe_echo(f"Wrote {summary_path}")

    best = pick_best_row([r for r in rows if r.get("status") == "ok"])
    if best:
        recipe_out = {
            "subject": subject,
            "sweep": "hyperparams",
            "personal_days": None,
            "lwf_lambda": float(best["lwf_lambda"]),
            "lr": float(best["lr"]),
            "weight_decay": float(best["weight_decay"]),
            "patience": int(best["patience"]),
            "epochs": epochs,
            "base_run_dir": str(base_run_dir),
            "glumind_lwf_start": GLUMIND_BEST_LWF_TYPE1,
            "ft_test_mae": best.get("ft_test_mae"),
            "zs_test_mae": best.get("zs_test_mae"),
            "source_run_dir": best.get("run_dir"),
        }
        recipe_path = out_dir / "best_recipe.json"
        write_best_recipe(recipe_path, recipe_out)
        subject_root = out_dir.parents[1] if out_dir.name == "hyperparams" else out_dir
        write_best_recipe(subject_root / "best_recipe.json", recipe_out)
        safe_echo(
            f"Best lwf={best['lwf_lambda']} lr={best['lr']} wd={best['weight_decay']} "
            f"MAE={best.get('ft_test_mae')} -> {recipe_path}"
        )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
