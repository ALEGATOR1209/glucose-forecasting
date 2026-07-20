#!/usr/bin/env python3
"""Step 5: validate Livia recipe on Loop holdouts + compare data-size curves.

Phase A — apply frozen Livia LwF/LR on full personal train data per holdout.
Phase B — repeat data-size sweep per holdout with same LwF/LR; compare plateau
curves to Livia and document differences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from scripts.personalization.constants import (
    DEFAULT_BASE_RUN_DIR,
    DEFAULT_DATA_SIZE_DAYS,
    DEFAULT_FT_PATIENCE,
    DEFAULT_SEED,
    LOOP_HOLDOUT_QUALITY_USERS,
)
from scripts.personalization.finetune import run_finetune
from scripts.personalization.prepare_personal_csv import prepare_person_frame
from scripts.personalization.splits import write_split_meta
from scripts.personalization.sweep_data_size import _parse_days_grid
from scripts.personalization.sweep_utils import (
    estimate_plateau_day,
    flatten_metrics,
    load_best_recipe,
    write_summary,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _ensure_holdout_csv(
    loop_csv: Path,
    user_id: str,
    out_dir: Path,
    test_fraction: float,
    val_fraction_of_remainder: float,
) -> Path:
    import polars as pl

    from scripts.personalization.constants import COL_USER

    out_csv = out_dir / f"loop_{user_id}_chronological.csv"
    if out_csv.exists():
        return out_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    person = (
        pl.scan_csv(loop_csv, infer_schema_length=10_000)
        .with_columns(pl.col(COL_USER).cast(pl.Utf8))
        .filter(pl.col(COL_USER) == user_id)
        .collect()
    )
    if person.is_empty():
        raise ValueError(f"User {user_id} not found in {loop_csv}")

    labeled, meta = prepare_person_frame(
        person,
        test_fraction=test_fraction,
        val_fraction_of_remainder=val_fraction_of_remainder,
        personal_days=None,
        study_group="T1DM",
    )
    meta["source"] = str(loop_csv)
    meta["subject"] = f"loop_{user_id}"
    meta["user_ids"] = [user_id]
    labeled.write_csv(out_csv)
    write_split_meta(out_dir / f"loop_{user_id}_split_meta.json", meta)
    return out_csv


def _run_data_size_for_subject(
    *,
    base_run_dir: Path,
    personal_csv: Path,
    out_dir: Path,
    subject: str,
    lwf: float,
    lr: float,
    weight_decay: float,
    patience: int,
    epochs: int,
    days_grid: list[int | None],
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for day_budget in days_grid:
        label = "all" if day_budget is None else str(day_budget)
        try:
            run_dir, results = run_finetune(
                base_run_dir=base_run_dir,
                personal_csv=personal_csv,
                out_dir=out_dir / f"days_{label}",
                run_name=f"{subject}_days_{label}",
                personal_days=day_budget,
                lwf_lambda=lwf,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                patience=patience,
                batch_size=batch_size,
                seed=seed,
                device=device,
                eval_zero_shot=True,
            )
        except ValueError as exc:
            rows.append(
                {
                    "subject": subject,
                    "personal_days": label,
                    "status": "skipped",
                    "error": str(exc),
                }
            )
            continue
        rows.append(
            {
                "subject": subject,
                "personal_days": label,
                "lwf_lambda": lwf,
                "lr": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "run_dir": str(run_dir),
                "status": "ok",
                **flatten_metrics("zs_test", results.get("zero_shot_test")),
                **flatten_metrics("ft_test", results.get("finetuned_test")),
            }
        )
    plateau = estimate_plateau_day(rows)
    return rows, plateau


@app.command()
def main(
    base_run_dir: Path = typer.Option(
        Path(DEFAULT_BASE_RUN_DIR),
        "--base-run-dir",
    ),
    recipe_json: Path = typer.Option(
        ...,
        "--recipe-json",
        help="Frozen Livia recipe (best_recipe.json from hyperparameter sweep).",
    ),
    livia_data_size_summary: Optional[Path] = typer.Option(
        None,
        "--livia-data-size-summary",
        help="Livia data_size/summary.csv for curve comparison.",
    ),
    loop_csv: Path = typer.Option(
        Path("data/loop_and_ai_ready/loop.csv"),
        "--loop-csv",
    ),
    holdout_dir: Path = typer.Option(
        Path("data/personalization/holdouts"),
        "--holdout-dir",
    ),
    out_dir: Path = typer.Option(
        Path("data/output/runs/personalization/holdout_validation"),
        "--out-dir",
    ),
    users: Optional[str] = typer.Option(None, "--users"),
    days: Optional[str] = typer.Option(
        None,
        "--days",
        help="Day grid for holdout data-size curves (default: same as Livia).",
    ),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
    batch_size: int = typer.Option(256, "--batch-size"),
    seed: int = typer.Option(DEFAULT_SEED, "--seed"),
    device: str = typer.Option("cpu", "--device"),
    test_fraction: float = typer.Option(0.25, "--test-fraction"),
    val_fraction_of_remainder: float = typer.Option(0.15, "--val-fraction-of-remainder"),
    skip_prepare: bool = typer.Option(False, "--skip-prepare"),
    skip_data_size: bool = typer.Option(
        False,
        "--skip-data-size",
        help="Only run full-data params validation (phase A).",
    ),
) -> None:
    """Validate Livia hyperparameters and data-size curves on holdout users."""
    recipe = load_best_recipe(recipe_json)
    lwf = float(recipe.get("lwf_lambda", 0.3))
    lr = float(recipe.get("lr", 4e-4))
    wd = float(recipe.get("weight_decay", 3e-5))
    patience = int(recipe.get("patience", DEFAULT_FT_PATIENCE))
    recipe_epochs = int(epochs if epochs is not None else recipe.get("epochs", 30))
    days_grid = _parse_days_grid(days)

    user_list = (
        [u.strip() for u in users.split(",") if u.strip()]
        if users
        else list(LOOP_HOLDOUT_QUALITY_USERS)
    )

    params_rows: list[dict[str, Any]] = []
    data_size_by_user: dict[str, list[dict[str, Any]]] = {}
    plateau_by_user: dict[str, dict[str, Any]] = {}

    for uid in user_list:
        subject = f"loop_{uid}"
        typer.echo(f"\n===== holdout user={uid} =====")
        try:
            if skip_prepare:
                personal_csv = holdout_dir / f"loop_{uid}_chronological.csv"
                if not personal_csv.exists():
                    raise ValueError(f"Missing prepared CSV: {personal_csv}")
            else:
                if not loop_csv.exists():
                    raise ValueError(f"loop CSV not found: {loop_csv}")
                personal_csv = _ensure_holdout_csv(
                    loop_csv,
                    uid,
                    holdout_dir,
                    test_fraction,
                    val_fraction_of_remainder,
                )

            # Phase A: full train data with frozen Livia LwF/LR
            typer.echo("Phase A: full-data fine-tune with Livia recipe")
            run_dir, results = run_finetune(
                base_run_dir=base_run_dir,
                personal_csv=personal_csv,
                out_dir=out_dir / "params" / subject,
                run_name=f"{subject}_full_recipe",
                personal_days=None,
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
            zs = results.get("zero_shot_test") or {}
            ft = results.get("finetuned_test") or {}
            delta_mae = None
            if zs.get("mae") is not None and ft.get("mae") is not None:
                delta_mae = float(ft["mae"]) - float(zs["mae"])

            params_rows.append(
                {
                    "user_id": uid,
                    "subject": subject,
                    "phase": "params_validation",
                    "status": "ok",
                    "lwf_lambda": lwf,
                    "lr": lr,
                    "weight_decay": wd,
                    "patience": patience,
                    "personal_days": "all",
                    "run_dir": str(run_dir),
                    **flatten_metrics("zs_test", results.get("zero_shot_test")),
                    **flatten_metrics("ft_test", results.get("finetuned_test")),
                    "delta_mae_ft_minus_zs": delta_mae,
                    "improved": (delta_mae is not None and delta_mae < 0),
                }
            )

            # Phase B: data-size curve with same LwF/LR
            if not skip_data_size:
                typer.echo("Phase B: data-size sweep with frozen Livia LwF/LR")
                ds_rows, plateau = _run_data_size_for_subject(
                    base_run_dir=base_run_dir,
                    personal_csv=personal_csv,
                    out_dir=out_dir / "data_size" / subject,
                    subject=subject,
                    lwf=lwf,
                    lr=lr,
                    weight_decay=wd,
                    patience=patience,
                    epochs=recipe_epochs,
                    days_grid=days_grid,
                    batch_size=batch_size,
                    seed=seed,
                    device=device,
                )
                data_size_by_user[uid] = ds_rows
                plateau_by_user[uid] = plateau
                write_summary(ds_rows, out_dir / "data_size" / subject)

        except (ValueError, FileNotFoundError) as exc:
            typer.echo(f"Skipping user {uid}: {exc}", err=True)
            params_rows.append({"user_id": uid, "phase": "params_validation", "status": "skipped", "error": str(exc)})
            continue

    params_path = write_summary(params_rows, out_dir / "params", name="summary")

    livia_plateau: dict[str, Any] | None = None
    if livia_data_size_summary is not None and livia_data_size_summary.exists():
        import polars as pl

        livia_rows = pl.read_csv(livia_data_size_summary).to_dicts()
        livia_plateau = estimate_plateau_day(livia_rows)

    comparison: list[dict[str, Any]] = []
    for uid, plateau in plateau_by_user.items():
        entry: dict[str, Any] = {
            "user_id": uid,
            "holdout_plateau_day": plateau.get("plateau_day"),
            "holdout_optimal_day": plateau.get("optimal_day"),
            "holdout_best_mae": plateau.get("best_mae"),
        }
        if livia_plateau:
            entry["livia_plateau_day"] = livia_plateau.get("plateau_day")
            entry["livia_optimal_day"] = livia_plateau.get("optimal_day")
            entry["livia_best_mae"] = livia_plateau.get("best_mae")
            h_opt = plateau.get("optimal_day")
            l_opt = livia_plateau.get("optimal_day")
            if h_opt is not None and l_opt is not None:
                try:
                    entry["optimal_day_delta"] = float(h_opt) - float(l_opt)
                except (TypeError, ValueError):
                    entry["optimal_day_delta"] = None
        comparison.append(entry)

    meta = {
        "recipe_json": str(recipe_json),
        "recipe": recipe,
        "users": user_list,
        "params_summary": str(params_path),
        "n_params_ok": sum(1 for r in params_rows if r.get("status") == "ok"),
        "n_params_improved": sum(1 for r in params_rows if r.get("improved")),
        "livia_plateau": livia_plateau,
        "holdout_plateau_by_user": plateau_by_user,
        "curve_comparison": comparison,
    }
    with (out_dir / "validation_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    typer.echo(f"Wrote {params_path}")
    typer.echo(
        f"Params improved vs zero-shot: {meta['n_params_improved']}/{meta['n_params_ok']}"
    )
    if comparison:
        typer.echo("Curve comparison vs Livia:")
        for row in comparison:
            typer.echo(f"  user {row['user_id']}: {row}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
