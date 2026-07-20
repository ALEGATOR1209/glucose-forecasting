#!/usr/bin/env python3
"""Step 3: sweep personal train-day budgets using the best LwF/LR recipe.

After hyperparameters are fixed on full personal train data, measure how test
MAE improves as more days of personal data are used for fine-tuning. Estimates
plateau day for optimal dataset size.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from scripts.personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_DATA_SIZE_DAYS,
    DEFAULT_FT_PATIENCE,
    DEFAULT_SEED,
)
from scripts.personalization.finetune import run_finetune
from scripts.personalization.sweep_utils import (
    estimate_plateau_day,
    flatten_metrics,
    load_best_recipe,
    write_best_recipe,
    write_summary,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _parse_days_grid(raw: str | None) -> list[int | None]:
    if raw is None:
        items = list(DEFAULT_DATA_SIZE_DAYS)
    else:
        items = [p.strip() for p in raw.split(",") if p.strip()]
    out: list[int | None] = []
    for item in items:
        if str(item).lower() == "all":
            out.append(None)
        else:
            out.append(int(item))
    return out


@app.command()
def main(
    base_run_dir: Path = typer.Option(
        Path(DEFAULT_BASE_RUN_DIR),
        "--base-run-dir",
    ),
    personal_csv: Path = typer.Option(..., "--personal-csv"),
    out_dir: Path = typer.Option(
        Path("data/output/runs/personalization/livia/sweeps/data_size"),
        "--out-dir",
    ),
    recipe_json: Path = typer.Option(
        ...,
        "--recipe-json",
        help="best_recipe.json from hyperparameter sweep (LwF + LR).",
    ),
    days: Optional[str] = typer.Option(
        None,
        "--days",
        help="Comma-separated day budgets, e.g. '1,3,7,14,30,60,all'.",
    ),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    subject: str = typer.Option("livia", "--subject"),
) -> None:
    """Run data-size learning curve with fixed LwF/LR; estimate plateau."""
    recipe = load_best_recipe(recipe_json)
    lwf = float(recipe.get("lwf_lambda", 0.3))
    lr = float(recipe.get("lr", 4e-4))
    wd = float(recipe.get("weight_decay", 3e-5))
    patience = int(recipe.get("patience", DEFAULT_FT_PATIENCE))
    recipe_epochs = int(epochs if epochs is not None else recipe.get("epochs", 30))

    typer.echo(
        f"Using recipe: lwf={lwf} lr={lr} weight_decay={wd} patience={patience} from {recipe_json}"
    )

    grid = _parse_days_grid(days)
    rows: list[dict] = []

    for day_budget in grid:
        label = "all" if day_budget is None else str(day_budget)
        typer.echo(f"\n===== data-size days={label} =====")
        try:
            run_dir, results = run_finetune(
                base_run_dir=base_run_dir,
                personal_csv=personal_csv,
                out_dir=out_dir / f"days_{label}",
                run_name=f"{subject}_days_{label}",
                personal_days=day_budget,
                lwf_lambda=lwf,
                epochs=recipe_epochs,
                lr=lr,
                weight_decay=wd,
                patience=patience,
                batch_size=batch_size,
                seed=seed,
                device=device,
                eval_zero_shot=True,
            )
        except ValueError as exc:
            typer.echo(f"Skipping days={label}: {exc}", err=True)
            rows.append(
                {
                    "subject": subject,
                    "personal_days": label,
                    "lwf_lambda": lwf,
                    "lr": lr,
                    "status": "skipped",
                    "error": str(exc),
                }
            )
            continue

        row = {
            "subject": subject,
            "personal_days": label,
            "lwf_lambda": lwf,
            "lr": lr,
            "weight_decay": wd,
            "patience": patience,
            "run_dir": str(run_dir),
            "status": "ok",
            **flatten_metrics("zs_test", results.get("zero_shot_test")),
            **flatten_metrics("ft_test", results.get("finetuned_test")),
            **flatten_metrics("ft_val", results.get("finetuned_val")),
        }
        rows.append(row)

    summary_path = write_summary(rows, out_dir)
    plateau_info = estimate_plateau_day(rows)
    plateau_path = out_dir / "plateau_analysis.json"
    with plateau_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(plateau_info, f, indent=2)

    typer.echo(f"Wrote {summary_path}")
    typer.echo(
        f"Plateau estimate: optimal_day={plateau_info.get('optimal_day')} "
        f"plateau_day={plateau_info.get('plateau_day')}"
    )

    recipe_out = {
        **recipe,
        "sweep": "data_size",
        "plateau_day": plateau_info.get("plateau_day"),
        "optimal_day": plateau_info.get("optimal_day"),
        "best_mae": plateau_info.get("best_mae"),
    }
    write_best_recipe(out_dir / "best_from_data_size.json", recipe_out)
    subject_root = out_dir.parents[1] if out_dir.name == "data_size" else out_dir
    write_best_recipe(subject_root / "best_recipe_with_days.json", recipe_out)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
