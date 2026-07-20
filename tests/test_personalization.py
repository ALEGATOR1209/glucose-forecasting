"""Tests for Milestone 8 personalization package."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import torch
from typer.testing import CliRunner

from scripts.personalization.constants import SPARSE_WINDOW_STRIDE
from scripts.personalization.finetune import run_finetune
from scripts.personalization.prepare_personal_csv import app as prepare_app
from scripts.personalization.splits import chronological_split_labels, limit_train_days
from scripts.personalization.sweep_utils import (
    estimate_plateau_day,
)
from scripts.sugar_one.train_sugar_one import SugarOneWindowDataset
from scripts.sugar_one.sugar_one_model import SugarOneModel
from tests.conftest import (
    TINY_D_MODEL,
    TINY_FF_UNITS,
    TINY_HORIZON,
    TINY_INPUT_STEPS,
    TINY_N_BLOCKS,
    TINY_N_HEADS,
)

runner = CliRunner()


def _write_continuous_person_csv(path: Path, *, n_rows: int = 400, user_id: str = "Subject000") -> None:
    start = datetime(2024, 1, 1, 0, 0, 0)
    rows: list[dict[str, object]] = []
    for i in range(n_rows):
        ts = start + timedelta(minutes=5 * i)
        rows.append(
            {
                "sequence_id": f"seq_{user_id}",
                "Timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "Event Type": "EGV",
                "User ID": user_id,
                "Glucose (mg/dL)": 100.0 + (i % 50) * 0.5,
                "Basal Rate (U/h)": "1.0",
                "Bolus Insulin (U)": "2.0" if i % 20 == 0 else "",
                "Carbohydrates (g)": "15.0" if i % 30 == 0 else "",
            }
        )
    pl.DataFrame(rows).write_csv(path)


def _make_tiny_base_run(tmp_path: Path, *, lr: float = 4e-4, patience: int = 10) -> Path:
    run_dir = tmp_path / "base_sugar_one"
    run_dir.mkdir(parents=True)
    meta = {
        "model_type": "sugar_one",
        "input_steps": TINY_INPUT_STEPS,
        "horizon": TINY_HORIZON,
        "d_model": TINY_D_MODEL,
        "n_heads": TINY_N_HEADS,
        "ff_units": TINY_FF_UNITS,
        "n_blocks": TINY_N_BLOCKS,
        "dropout": 0.0,
        "lr": lr,
        "patience": patience,
        "weight_decay": 3e-5,
    }
    with (run_dir / "tuning_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    model = SugarOneModel(
        n_time_steps=TINY_INPUT_STEPS,
        n_features=4,
        d_model=TINY_D_MODEL,
        n_heads=TINY_N_HEADS,
        ff_units=TINY_FF_UNITS,
        n_blocks=TINY_N_BLOCKS,
        prediction_horizon=TINY_HORIZON,
        dropout=0.0,
    )
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    return run_dir


def test_chronological_split_and_day_limit() -> None:
    start = datetime(2024, 1, 1)
    rows = [
        {
            "Timestamp": start + timedelta(minutes=5 * i),
            "User ID": "u1",
            "Glucose (mg/dL)": 100.0,
        }
        for i in range(1000)
    ]
    df = pl.DataFrame(rows)
    labeled = chronological_split_labels(df, test_fraction=0.25, val_fraction_of_remainder=0.2)
    limited = limit_train_days(labeled, personal_days=1)
    train = limited.filter(pl.col("Recommended Split") == "train")
    assert train.height < labeled.filter(pl.col("Recommended Split") == "train").height


def test_prepare_livia_cli(tmp_path: Path) -> None:
    raw = tmp_path / "raw_person.csv"
    _write_continuous_person_csv(raw, n_rows=200)
    out_dir = tmp_path / "prepared"
    result = runner.invoke(
        prepare_app,
        ["livia", "--input", str(raw), "--out-dir", str(out_dir), "--out-name", "person.csv"],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "person.csv").exists()


def test_window_stride_reduces_train_windows(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    _write_continuous_person_csv(raw, n_rows=300)
    prepared_dir = tmp_path / "prepared"
    prep = runner.invoke(
        prepare_app,
        ["livia", "--input", str(raw), "--out-dir", str(prepared_dir), "--out-name", "p.csv"],
    )
    assert prep.exit_code == 0, prep.output
    from scripts.personalization.finetune import _load_split_frames

    train_df, _, _ = _load_split_frames(prepared_dir / "p.csv")
    dense = SugarOneWindowDataset(train_df, TINY_INPUT_STEPS, TINY_HORIZON, fit_scalers=True)
    sparse = SugarOneWindowDataset(
        train_df,
        TINY_INPUT_STEPS,
        TINY_HORIZON,
        scaler_glucose=dense.scaler_glucose,
        scaler_basal=dense.scaler_basal,
        scaler_bolus=dense.scaler_bolus,
        scaler_carbs=dense.scaler_carbs,
        window_stride=SPARSE_WINDOW_STRIDE,
    )
    assert len(sparse) < len(dense)
    assert len(sparse) >= len(dense) // SPARSE_WINDOW_STRIDE - 1


def _prepare_finetune_case(tmp_path: Path) -> tuple[Path, Path]:
    base = _make_tiny_base_run(tmp_path)
    raw = tmp_path / "raw.csv"
    _write_continuous_person_csv(raw, n_rows=300)
    prepared_dir = tmp_path / "prepared"
    result = runner.invoke(
        prepare_app,
        ["livia", "--input", str(raw), "--out-dir", str(prepared_dir), "--out-name", "p.csv"],
    )
    assert result.exit_code == 0, result.output
    return base, prepared_dir / "p.csv"


@pytest.mark.parametrize(
    ("name", "window_stride", "lwf_lambda", "eval_zero_shot", "artifact"),
    [
        ("sparse", SPARSE_WINDOW_STRIDE, 0.0, False, "personalization_metrics.json"),
        ("lwf", 1, 0.5, True, "tuning_meta.json"),
        ("no_lwf", 1, 0.0, False, "personalization_metrics.json"),
    ],
)
def test_finetune_smoke(
    tmp_path: Path,
    name: str,
    window_stride: int,
    lwf_lambda: float,
    eval_zero_shot: bool,
    artifact: str,
) -> None:
    base, personal_csv = _prepare_finetune_case(tmp_path)
    run_dir, results = run_finetune(
        base_run_dir=base,
        personal_csv=personal_csv,
        out_dir=tmp_path / "finetune",
        run_name=name,
        personal_days=2,
        train_window_stride=window_stride,
        lwf_lambda=lwf_lambda,
        epochs=1,
        patience=0,
        batch_size=8,
        device="cpu",
        num_workers=0,
        eval_zero_shot=eval_zero_shot,
    )
    assert (run_dir / artifact).exists()
    assert results["config"]["train_window_stride"] == window_stride
    assert results["config"]["lwf_lambda"] == lwf_lambda
    if eval_zero_shot:
        assert results.get("finetuned_test") is not None


def test_plateau_estimation() -> None:
    rows = [
        {"status": "ok", "personal_days": 1, "ft_test_mae": 15.0},
        {"status": "ok", "personal_days": 7, "ft_test_mae": 11.0},
        {"status": "ok", "personal_days": 14, "ft_test_mae": 10.6},
        {"status": "ok", "personal_days": 30, "ft_test_mae": 10.55},
    ]
    info = estimate_plateau_day(rows)
    assert info["optimal_day"] in (14, 30)
    assert info["plateau_day"] is not None


def test_aggregate_cli(tmp_path: Path) -> None:
    from scripts.personalization.aggregate_results import app as agg_app

    root = tmp_path / "runs"
    sweep = root / "livia" / "sweeps" / "hyperparams"
    sweep.mkdir(parents=True)
    pl.DataFrame(
        [{"subject": "livia", "lwf_lambda": 0.5, "ft_test_mae": 11.0, "status": "ok"}]
    ).write_csv(sweep / "summary.csv")
    out_json = tmp_path / "summary.json"
    result = runner.invoke(agg_app, ["--root", str(root), "--out", str(out_json)])
    assert result.exit_code == 0, result.output
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["sections"]["hyperparams"]["n_rows"] == 1
