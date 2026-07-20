#!/usr/bin/env python3
"""
evaluate_model.py — Evaluate GluMind or SugarOne checkpoints on arbitrary CSV data.

Supports cross-model / cross-dataset comparison:
  - Models: scripts/glumind/glumind_model.py (HR + steps) or
            scripts/sugar_one/sugar_one_model.py (basal + bolus + carbs)
  - Datasets: ai_ready (Glucose Value, HR, Step Count) or
              loop_ai_ready (Glucose, Basal Rate, Bolus, Carbs)

Missing covariates are filled with 0.0 before imputation.
Use --zero-cov to force all non-glucose covariates to 0.0 after imputation
(for fair comparison with models trained without covariates on the same dataset).
Use --include-cov / --exclude-cov for finer-grained covariate ablation at inference.
Use --covariates to inspect which covariate columns exist in a CSV (no model run).
If the CSV has a 'Recommended Split' column, rows with split == --test-split
(default: test) are evaluated; otherwise all rows are used.

Model resolution (same as evaluate_glumind.py):
  --run-dir       directory with tuning_meta.json and best_model.pt
  --registry-dir  auto-picks lowest val_mae from _analysis_registry.csv
  --checkpoint    explicit .pt file (requires --run-dir for architecture meta)

Examples:
  uv run evaluate-model \\
      --run-dir test_model_sugar_one \\
      --model-type sugar_one \\
      --test-csv test_data/livia_sugar_one_ready.csv \\
      --train-csv test_data/livia_sugar_one_ready.csv \\
      --test-split ''

  uv run evaluate-model \\
      --run-dir test_model_sugar_one \\
      --model-type sugar_one \\
      --test-csv data/input/loop_ai_ready_joined2.csv \\
      --train-csv data/input/loop_ai_ready_joined2.csv
"""
from __future__ import annotations

import csv
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader

from glucose_forecasting.data.glumind import (
    COL_GROUP,
    COL_SEQ,
    COL_SPLIT,
    COL_TS as GLUMIND_COL_TS,
    COL_USER,
    TS_FORMAT,
    GlucoseWindowDataset,
    apply_split_scheme as apply_split_scheme_glumind,
    impute_and_sort as impute_and_sort_glumind,
    load_splits_streaming as load_splits_glumind,
)
from glucose_forecasting.data.sugar_one import (
    COL_TS as IC_COL_TS,
    SugarOneWindowDataset,
    apply_split_scheme as apply_split_scheme_ic,
    impute_and_sort as impute_and_sort_ic,
    load_splits_streaming as load_splits_ic,
)
from glucose_forecasting.models.glumind import GluMindModel
from glucose_forecasting.training.glumind import (
    mae_rmse_mard,
)
from scripts.sugar_one.sugar_one_model import SugarOneModel
from scripts.common.checkpoint import strip_compile_prefix
from scripts.common.registry import (
    find_best_run_dir as _common_find_best_run_dir,
    load_run_meta as _load_meta,
    resolve_checkpoint as _common_resolve_checkpoint,
    resolve_csv_path as _common_resolve_csv_path,
)
from scripts.common.evaluation import (
    COVARIATE_NAME_ALIASES,
    GLUMIND_COVARIATES,
    SUGAR_ONE_COVARIATES,
    _alias_to_canonical,
    _load_csv_flexible as _common_load_csv_flexible,
    _parse_covariate_names,
    _pick_header_column,
    _resolve_covariate_zeroing,
    _run_evaluate as _common_run_evaluate,
    _split_cov_arg,
    _zero_covariates,
    DEFAULT_INFERENCE_LOG_INTERVAL_S,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

ModelKind = Literal["glumind", "sugar_one"]

COL_EVENT = "Event Type"

TS_ALIASES = [GLUMIND_COL_TS, IC_COL_TS]


class ModelTypeChoice(str, Enum):
    auto = "auto"
    glumind = "glumind"
    sugar_one = "sugar_one"


# ---------------------------------------------------------------------------
# Registry / checkpoint helpers (shared with evaluate_glumind.py via
# scripts.common.registry; thin wrappers below bind project_root).
# ---------------------------------------------------------------------------

def _find_best_run_dir(registry_dir: Path) -> tuple[Path, dict]:
    return _common_find_best_run_dir(registry_dir, project_root)


def _resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    return _common_resolve_checkpoint(run_dir, checkpoint)


def _resolve_csv_path(csv_value: str | Path) -> Path:
    return _common_resolve_csv_path(csv_value, project_root)


def _detect_model_kind(meta: dict, state: dict[str, torch.Tensor]) -> ModelKind:
    explicit = meta.get("model_type") or meta.get("model")
    if explicit is not None:
        explicit_norm = str(explicit).lower().replace("_", "").replace("-", "")
        if explicit_norm in ("sugarone", "glumindic"):
            return "sugar_one"
        if explicit_norm in ("glumind",):
            return "glumind"

    normalized_keys = {k.removeprefix("_orig_mod.") for k in state}
    ic_keys = ("embed_basal.weight", "embed_bolus.weight", "embed_carbs.weight")
    glumind_keys = ("embed_hr.weight", "embed_steps.weight")
    if any(k in normalized_keys for k in ic_keys):
        return "sugar_one"
    if any(k in normalized_keys for k in glumind_keys):
        return "glumind"

    typer.echo(
        "Error: Could not auto-detect model type from checkpoint. "
        "Pass --model-type glumind or sugar_one.",
        err=True,
    )
    raise typer.Exit(1)


def _covariate_map(model_kind: ModelKind) -> dict[str, list[str]]:
    return GLUMIND_COVARIATES if model_kind == "glumind" else SUGAR_ONE_COVARIATES


def _canonical_feature_cols(model_kind: ModelKind) -> list[str]:
    if model_kind == "glumind":
        return ["glucose", "hr", "steps"]
    return ["glucose", "basal", "bolus", "carbs"]


def _non_glucose_covariate_cols(model_kind: ModelKind) -> list[str]:
    return [c for c in _canonical_feature_cols(model_kind) if c != "glucose"]


def _zero_non_glucose_covariates(df: pl.DataFrame, model_kind: ModelKind) -> pl.DataFrame:
    """Replace all non-glucose covariates with 0.0 (applied after imputation)."""
    return _zero_covariates(df, _non_glucose_covariate_cols(model_kind))


def _is_filled_expr(source_col: str) -> pl.Expr:
    return pl.col(source_col).is_not_null() & (
        pl.col(source_col).cast(pl.Utf8).str.strip_chars() != ""
    )


def _read_csv_header(csv_path: Path) -> list[str]:
    with open(csv_path, newline="") as f:
        return next(csv.reader(f))


def _covariate_column_stats(
    csv_path: Path,
    source_col: str,
    eval_split: str | None,
) -> tuple[int, int]:
    """Return (total_rows, filled_rows) optionally filtered by split."""
    header = _read_csv_header(csv_path)
    has_split = COL_SPLIT in header
    lf = pl.scan_csv(
        csv_path,
        infer_schema_length=10_000,
        schema_overrides={source_col: pl.Utf8},
    )
    if eval_split and has_split:
        lf = lf.filter(pl.col(COL_SPLIT) == eval_split)
    stats = lf.select([
        pl.len().alias("total"),
        _is_filled_expr(source_col).sum().alias("filled"),
    ]).collect()
    return int(stats["total"][0]), int(stats["filled"][0])


def _print_dataset_covariates(
    csv_path: Path,
    model_kind: ModelKind | None,
    eval_split: str | None,
) -> None:
    """Print covariate column mapping and fill stats for the target CSV."""
    header = _read_csv_header(csv_path)
    kinds: list[ModelKind] = (
        [model_kind] if model_kind is not None else ["glumind", "sugar_one"]
    )
    split_label = eval_split if eval_split else "all rows"
    typer.echo(f"Dataset : {csv_path}")
    typer.echo(f"Split   : {split_label}")
    typer.echo("")

    for kind in kinds:
        cov_map = _covariate_map(kind)
        typer.echo(f"Model type: {kind}")
        typer.echo(f"  Feature channels: {', '.join(_canonical_feature_cols(kind))}")
        typer.echo(
            f"  Non-glucose covariates (--include-cov / --exclude-cov): "
            f"{', '.join(_non_glucose_covariate_cols(kind))}"
        )
        typer.echo("  Columns:")
        for canonical, aliases in cov_map.items():
            source_col = _pick_header_column(header, aliases)
            if source_col is None:
                typer.echo(f"    {canonical:8s}  missing  (loaded as 0.0)")
                continue
            total, filled = _covariate_column_stats(csv_path, source_col, eval_split)
            pct = 100.0 * filled / total if total else 0.0
            typer.echo(
                f"    {canonical:8s}  {source_col!r}  "
                f"filled {filled:,}/{total:,} ({pct:.1f}%)"
            )
        typer.echo("  Accepted aliases:")
        for canonical in _non_glucose_covariate_cols(kind):
            aliases = COVARIATE_NAME_ALIASES.get(canonical, [canonical])
            typer.echo(f"    {canonical}: {', '.join(aliases)}")
        typer.echo("")


def _load_csv_flexible(
    csv_path: Path,
    model_kind: ModelKind,
    unique_id_choice: str,
    drop_interpolated: bool,
    eval_split: str | None,
    train_only: bool,
) -> pl.DataFrame:
    """Load CSV with canonical columns; missing covariates become 0.0."""
    return _common_load_csv_flexible(
        csv_path,
        model_kind,
        unique_id_choice,
        drop_interpolated,
        eval_split,
        train_only,
        col_seq=COL_SEQ,
        col_user=COL_USER,
        col_split=COL_SPLIT,
        col_group=COL_GROUP,
        ts_aliases=TS_ALIASES,
        ts_format=TS_FORMAT,
    )


def _load_train_for_scalers(
    test_csv_path: Path,
    model_kind: ModelKind,
    meta: dict,
    train_csv_override: Path | None,
) -> pl.DataFrame:
    """Load training rows for scaler fitting."""
    scaler_csv = train_csv_override or _resolve_csv_path(meta["csv"])
    typer.echo(f"Fitting scalers from: {scaler_csv}")

    split_scheme = meta.get("split_scheme", "classic")
    unique_id = meta.get("unique_id", "sequence_id")
    drop_interpolated = meta.get("drop_interpolated", False)
    impute = impute_and_sort_glumind if model_kind == "glumind" else impute_and_sort_ic

    if scaler_csv == test_csv_path:
        train_df = _load_csv_flexible(
            scaler_csv,
            model_kind=model_kind,
            unique_id_choice=unique_id,
            drop_interpolated=drop_interpolated,
            eval_split=None,
            train_only=True,
        )
        if train_df.is_empty():
            typer.echo(
                "Warning: No train split rows in CSV — fitting scalers on all rows.",
                err=True,
            )
            train_df = _load_csv_flexible(
                scaler_csv,
                model_kind=model_kind,
                unique_id_choice=unique_id,
                drop_interpolated=drop_interpolated,
                eval_split=None,
                train_only=False,
            )
        return impute(train_df)

    load_splits = load_splits_glumind if model_kind == "glumind" else load_splits_ic
    apply_split = apply_split_scheme_glumind if model_kind == "glumind" else apply_split_scheme_ic

    train_df_raw, val_df_raw, test_df_raw = load_splits(
        scaler_csv,
        unique_id_choice=unique_id,
        drop_interpolated=drop_interpolated,
    )
    train_df_raw, _, _ = apply_split(train_df_raw, val_df_raw, test_df_raw, split_scheme)

    if train_df_raw.is_empty():
        typer.echo(
            "Warning: Training split empty after load_splits_streaming — "
            "using flexible loader on all rows.",
            err=True,
        )
        train_df = _load_csv_flexible(
            scaler_csv,
            model_kind=model_kind,
            unique_id_choice=unique_id,
            drop_interpolated=drop_interpolated,
            eval_split=None,
            train_only=False,
        )
        return impute(train_df)

    return impute(train_df_raw)


def _build_train_dataset(
    train_df: pl.DataFrame,
    model_kind: ModelKind,
    meta: dict,
) -> GlucoseWindowDataset | SugarOneWindowDataset:
    if model_kind == "glumind":
        return GlucoseWindowDataset(
            train_df,
            input_steps=meta["input_steps"],
            horizon=meta["horizon"],
            fit_scalers=True,
        )
    return SugarOneWindowDataset(
        train_df,
        input_steps=meta["input_steps"],
        horizon=meta["horizon"],
        fit_scalers=True,
    )


def _build_eval_dataset(
    eval_df: pl.DataFrame,
    train_ds: GlucoseWindowDataset | SugarOneWindowDataset,
    model_kind: ModelKind,
    meta: dict,
) -> GlucoseWindowDataset | SugarOneWindowDataset:
    if model_kind == "glumind":
        assert isinstance(train_ds, GlucoseWindowDataset)
        return GlucoseWindowDataset(
            eval_df,
            input_steps=meta["input_steps"],
            horizon=meta["horizon"],
            scaler_glucose=train_ds.scaler_glucose,
            scaler_hr=train_ds.scaler_hr,
            scaler_steps=train_ds.scaler_steps,
            fit_scalers=False,
        )
    assert isinstance(train_ds, SugarOneWindowDataset)
    return SugarOneWindowDataset(
        eval_df,
        input_steps=meta["input_steps"],
        horizon=meta["horizon"],
        scaler_glucose=train_ds.scaler_glucose,
        scaler_basal=train_ds.scaler_basal,
        scaler_bolus=train_ds.scaler_bolus,
        scaler_carbs=train_ds.scaler_carbs,
        fit_scalers=False,
    )


def _build_model(model_kind: ModelKind, meta: dict) -> nn.Module:
    common = dict(
        n_time_steps=meta["input_steps"],
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        ff_units=meta["ff_units"],
        n_blocks=meta["n_blocks"],
        prediction_horizon=meta["horizon"],
        dropout=meta.get("dropout", 0.1),
    )
    if model_kind == "glumind":
        return GluMindModel(n_features=3, **common)
    return SugarOneModel(n_features=4, **common)


def _load_model_weights(
    model: nn.Module,
    ckpt_path: Path,
    device: str,
) -> None:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = strip_compile_prefix(state)
    model.load_state_dict(state)
    model.to(device)
    model.eval()


def _run_evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_windows: int,
    log_interval_s: float = DEFAULT_INFERENCE_LOG_INTERVAL_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with periodic progress logs and ETA."""
    return _common_run_evaluate(model, loader, device, n_windows, log_interval_s)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    test_csv: Path = typer.Option(
        ...,
        "--test-csv",
        help="CSV file to evaluate on.",
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        help="Run directory with tuning_meta.json and best_model.pt.",
    ),
    registry_dir: Path | None = typer.Option(
        None,
        "--registry-dir",
        help="Directory with _analysis_registry.csv; picks lowest val_mae run.",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Explicit .pt weights file (requires --run-dir for architecture meta).",
    ),
    train_csv: Path | None = typer.Option(
        None,
        "--train-csv",
        help="CSV for scaler fitting (default: csv from tuning_meta.json).",
    ),
    model_type: ModelTypeChoice = typer.Option(
        ModelTypeChoice.auto,
        "--model-type",
        help="Model architecture: glumind (HR+steps) or sugar_one (insulin+carbs).",
    ),
    test_split: str | None = typer.Option(
        "test",
        "--test-split",
        help=(
            "Evaluate rows where 'Recommended Split' equals this value. "
            "Use --test-split='' to disable split filtering."
        ),
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="DataLoader batch size (default: from tuning_meta.json).",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference.",
    ),
    output_json: Path | None = typer.Option(
        None,
        "--output-json",
        help="Optional path to write metrics as JSON for batch comparisons.",
    ),
    log_interval: float = typer.Option(
        DEFAULT_INFERENCE_LOG_INTERVAL_S,
        "--log-interval",
        help="Seconds between inference progress log lines (0 = log first and last only).",
    ),
    zero_cov: bool = typer.Option(
        False,
        "--zero-cov",
        help=(
            "Zero all non-glucose covariates at evaluation time (after imputation). "
            "Equivalent to excluding every non-glucose covariate. "
            "Mutually exclusive with --include-cov / --exclude-cov."
        ),
    ),
    include_cov: str | None = typer.Option(
        None,
        "--include-cov",
        help=(
            "Comma-separated covariates to keep at inference; all other "
            "non-glucose covariates are zeroed after imputation. "
            "Example: --include-cov basal,bolus"
        ),
    ),
    exclude_cov: str | None = typer.Option(
        None,
        "--exclude-cov",
        help=(
            "Comma-separated covariates to zero at inference; remaining "
            "non-glucose covariates are kept. Example: --exclude-cov carbs"
        ),
    ),
    covariates: bool = typer.Option(
        False,
        "--covariates",
        help=(
            "Print covariate columns available in --test-csv and exit "
            "(no checkpoint or inference required)."
        ),
    ),
) -> None:
    test_path = _resolve_csv_path(test_csv)
    eval_split = test_split if test_split else None

    if covariates:
        resolved_kind: ModelKind | None
        if model_type == ModelTypeChoice.auto:
            resolved_kind = None
        else:
            resolved_kind = model_type.value  # type: ignore[assignment]
        _print_dataset_covariates(test_path, resolved_kind, eval_split)
        raise typer.Exit(0)

    if run_dir is None and registry_dir is None:
        typer.echo("Error: Provide at least one of --run-dir or --registry-dir.", err=True)
        raise typer.Exit(1)

    if run_dir is not None:
        resolved_run_dir = run_dir
    else:
        resolved_run_dir, _ = _find_best_run_dir(registry_dir)  # type: ignore[arg-type]

    if not resolved_run_dir.exists():
        typer.echo(f"Error: Run directory does not exist: {resolved_run_dir}", err=True)
        raise typer.Exit(1)

    meta = _load_meta(resolved_run_dir)
    ckpt_path = _resolve_checkpoint(resolved_run_dir, checkpoint)

    typer.echo(f"Run directory: {resolved_run_dir}")
    typer.echo(f"Checkpoint   : {ckpt_path}")
    typer.echo(f"Test CSV     : {test_path}")

    state_probe = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if model_type == ModelTypeChoice.auto:
        resolved_kind = _detect_model_kind(meta, state_probe)
    else:
        resolved_kind = model_type.value  # type: ignore[assignment]

    typer.echo(f"Model type   : {resolved_kind}")

    try:
        active_cov, zeroed_cov = _resolve_covariate_zeroing(
            resolved_kind,
            zero_cov=zero_cov,
            include_cov=include_cov,
            exclude_cov=exclude_cov,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if zeroed_cov:
        if zero_cov:
            typer.echo(f"  --zero-cov: covariates set to 0.0: {', '.join(zeroed_cov)}")
        elif include_cov:
            typer.echo(
                f"  --include-cov {include_cov}: active={', '.join(active_cov)}; "
                f"zeroed={', '.join(zeroed_cov)}"
            )
        else:
            typer.echo(
                f"  --exclude-cov {exclude_cov}: active={', '.join(active_cov)}; "
                f"zeroed={', '.join(zeroed_cov)}"
            )

    train_df = _load_train_for_scalers(
        test_path,
        model_kind=resolved_kind,
        meta=meta,
        train_csv_override=train_csv,
    )
    train_ds = _build_train_dataset(train_df, resolved_kind, meta)
    typer.echo(
        f"Scalers fitted on {len(train_df):,} training rows, {len(train_ds):,} windows."
    )

    impute = impute_and_sort_glumind if resolved_kind == "glumind" else impute_and_sort_ic
    typer.echo(f"Loading evaluation data from: {test_path}")
    eval_df = _load_csv_flexible(
        test_path,
        model_kind=resolved_kind,
        unique_id_choice=meta.get("unique_id", "sequence_id"),
        drop_interpolated=meta.get("drop_interpolated", False),
        eval_split=eval_split,
        train_only=False,
    )
    eval_df = impute(eval_df)
    if zeroed_cov:
        eval_df = _zero_covariates(eval_df, zeroed_cov)

    if eval_df.is_empty():
        typer.echo("Error: Evaluation dataframe is empty after loading/filtering.", err=True)
        if eval_split is not None:
            typer.echo(
                f"Hint: no rows have Recommended Split == {eval_split!r}. "
                "Demo CSVs under test_data/ use an empty split — pass --test-split '' "
                "to score all rows.",
                err=True,
            )
        raise typer.Exit(1)

    eval_ds = _build_eval_dataset(eval_df, train_ds, resolved_kind, meta)
    if len(eval_ds) == 0:
        typer.echo(
            f"Error: No windows could be built. Each series needs at least "
            f"{meta['input_steps'] + meta['horizon']} rows.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Evaluation windows: {len(eval_ds):,}")
    resolved_batch_size = batch_size or meta.get("batch_size", 4096)
    eval_loader = DataLoader(eval_ds, batch_size=resolved_batch_size, shuffle=False)

    model = _build_model(resolved_kind, meta)
    _load_model_weights(model, ckpt_path, device)

    typer.echo("Running inference...")
    log_interval_s = max(0.0, log_interval)
    y_true_scaled, y_pred_scaled = _run_evaluate(
        model,
        eval_loader,
        device,
        n_windows=len(eval_ds),
        log_interval_s=log_interval_s,
    )

    y_true = train_ds.scaler_glucose.inverse_transform(
        y_true_scaled.ravel().reshape(-1, 1)
    ).ravel()
    y_pred = train_ds.scaler_glucose.inverse_transform(
        y_pred_scaled.ravel().reshape(-1, 1)
    ).ravel()

    mae, rmse, mard = mae_rmse_mard(y_true, y_pred)

    split_used = eval_split if eval_split else "all"
    typer.echo("\n" + "=" * 50)
    typer.echo("EVALUATION RESULTS")
    typer.echo(f"  Model type : {resolved_kind}")
    typer.echo(f"  Test CSV   : {test_path}")
    typer.echo(f"  Split used : {split_used}")
    typer.echo(f"  Zero cov   : {zero_cov}")
    if active_cov:
        typer.echo(f"  Active cov : {', '.join(active_cov)}")
    if zeroed_cov:
        typer.echo(f"  Zeroed cov : {', '.join(zeroed_cov)}")
    typer.echo(f"  Checkpoint : {ckpt_path}")
    typer.echo(f"  Windows    : {len(eval_ds):,}")
    typer.echo("-" * 50)
    typer.echo(f"  MAE : {mae:.4f}")
    typer.echo(f"  RMSE: {rmse:.4f}")
    typer.echo(f"  MARD: {mard:.4f}%")
    typer.echo("=" * 50)

    if output_json is not None:
        payload = {
            "model_type": resolved_kind,
            "test_csv": str(test_path),
            "run_dir": str(resolved_run_dir),
            "checkpoint": str(ckpt_path),
            "split_used": split_used,
            "zero_cov": zero_cov,
            "include_cov": _split_cov_arg(include_cov) or None,
            "exclude_cov": _split_cov_arg(exclude_cov) or None,
            "active_covariates": active_cov,
            "zeroed_covariates": zeroed_cov,
            "windows": len(eval_ds),
            "mae": mae,
            "rmse": rmse,
            "mard": mard,
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(payload, f, indent=2)
        typer.echo(f"Metrics written to {output_json}")


if __name__ == "__main__":
    app()
