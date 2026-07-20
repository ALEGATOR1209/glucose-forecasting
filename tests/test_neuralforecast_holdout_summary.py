"""Tests for aggregate fixed-split NeuralForecast holdout reports."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from glucose_forecasting.cli import app
from glucose_forecasting.backends.neuralforecast.evaluations.holdout import (
    summarize_holdout_runs,
)
from glucose_forecasting.backends.neuralforecast.reporting import collect_training_history

runner = CliRunner()


def _write_holdout_run(
    parent: Path,
    *,
    model: str,
    mae: float,
    input_hours: float = 128 * 5 / 60,
) -> Path:
    run_dir = parent / f"{model}_20260719T000000Z"
    run_dir.mkdir()
    config = {
        "csv": "data/input/loop.csv",
        "profile": "loop",
        "models": "auto",
        "evaluation": "holdout",
        "holdout_protocol": "sugarone-compatible",
        "out_dir": "data/output/runs",
        "unique_id": "sequence_id",
        "split_scheme": "classic",
        "global_model": True,
        "study_groups": [],
        "h_minutes": 60,
        "freq": "5min",
        "input_hours": input_hours,
        "train_tail_val_hours": 24.0,
        "max_steps": 10,
        "val_check_steps": 5,
        "batch_size": 8,
        "valid_batch_size": 8,
        "windows_batch_size": 32,
        "inference_windows_batch_size": 32,
        "step_size": 1,
        "learning_rate": 0.001,
        "max_train_series": 0,
        "max_eval_series": 0,
        "max_points_per_series": 0,
        "n_windows": 3,
        "drop_interpolated": False,
        "mask_interpolated_targets": False,
        "save_predictions": False,
        "plot": True,
        "max_plot_series": 3,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "training.json").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "message_type": "train_epoch_completed",
                        "model": model,
                        "epoch": 1,
                        "train_loss": 3.0,
                    }
                ),
                json.dumps(
                    {
                        "message_type": "validation_epoch_completed",
                        "model": model,
                        "epoch": 1,
                        "valid_loss": 2.5,
                    }
                ),
                json.dumps(
                    {
                        "message_type": "train_epoch_completed",
                        "model": model,
                        "epoch": 2,
                        "train_loss": 2.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    for split_name in ("val", "test"):
        pl.DataFrame({"mae": [mae], "rmse": [mae + 1], "mard": [mae + 2]}).write_csv(
            run_dir / f"{split_name}_metrics_overall.csv"
        )
        pl.DataFrame(
            {
                "study_group": ["T1DM"],
                "n_points": [2],
                "mae": [mae],
                "rmse": [mae + 1],
                "mard": [mae + 2],
            }
        ).write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")
        pl.DataFrame(
            {
                "unique_id": ["series-1", "series-1"],
                "ds": ["2026-01-01T00:00:00", "2026-01-01T00:05:00"],
                "y": [100.0, 110.0],
                "yhat": [100.0 + mae, 110.0 + mae],
                "study_group": ["T1DM", "T1DM"],
                "event_type": ["EGV", "EGV"],
            }
        ).write_csv(run_dir / f"{split_name}_predictions.csv")
    return run_dir


def test_summarize_holdout_runs_writes_metrics_predictions_and_config(tmp_path: Path) -> None:
    group_dir = tmp_path / "__ALL__"
    group_dir.mkdir()
    tft = _write_holdout_run(group_dir, model="TFT", mae=2.0)
    nhits = _write_holdout_run(group_dir, model="NHITS", mae=1.0)
    summary_dir = summarize_holdout_runs(
        [tft, nhits],
        output_dir=group_dir / "summaries" / "20260719T000001Z",
        plot=False,
    )

    summary = pl.read_csv(summary_dir / "test_metrics_summary.csv")
    assert summary["model"].to_list() == ["NHITS", "TFT"]
    assert pl.read_csv(summary_dir / "test_predictions.csv").height == 4
    config = json.loads((summary_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["selected_models"] == ["TFT", "NHITS"]
    manifest = json.loads((summary_dir / "model_runs.json").read_text(encoding="utf-8"))
    assert [entry["model"] for entry in manifest] == ["TFT", "NHITS"]
    assert (summary_dir / "plots" / "metrics.html").is_file()
    assert (summary_dir / "plots" / "study_group_metrics.html").is_file()
    assert (summary_dir / "plots" / "training.html").is_file()
    assert pl.read_csv(summary_dir / "study_group_metrics.csv").height == 4
    assert pl.read_csv(summary_dir / "training_history.csv").height == 4


def test_summarize_holdout_runs_writes_split_dashboards(tmp_path: Path) -> None:
    group_dir = tmp_path / "__ALL__"
    group_dir.mkdir()
    tft = _write_holdout_run(group_dir, model="TFT", mae=2.0)
    nhits = _write_holdout_run(group_dir, model="NHITS", mae=1.0)

    summary_dir = summarize_holdout_runs(
        [tft, nhits],
        output_dir=group_dir / "summaries" / "20260719T000001Z",
        plot=True,
    )

    assert (
        summary_dir
        / "plots"
        / "diagnostic_examples"
        / "comparison"
        / "val"
        / "model_comparison.html"
    ).is_file()
    assert (
        summary_dir
        / "plots"
        / "diagnostic_examples"
        / "comparison"
        / "test"
        / "model_comparison.html"
    ).is_file()


def test_summarize_holdout_runs_rejects_incompatible_geometry(tmp_path: Path) -> None:
    group_dir = tmp_path / "__ALL__"
    group_dir.mkdir()
    tft = _write_holdout_run(group_dir, model="TFT", mae=2.0)
    nhits = _write_holdout_run(group_dir, model="NHITS", mae=1.0, input_hours=6)

    with pytest.raises(ValueError, match="incompatible configurations"):
        summarize_holdout_runs(
            [tft, nhits],
            output_dir=group_dir / "summaries" / "20260719T000001Z",
            plot=False,
        )


def test_summarize_holdout_cli_writes_requested_output(tmp_path: Path) -> None:
    group_dir = tmp_path / "__ALL__"
    group_dir.mkdir()
    tft = _write_holdout_run(group_dir, model="TFT", mae=2.0)
    nhits = _write_holdout_run(group_dir, model="NHITS", mae=1.0)
    output_dir = tmp_path / "summary"

    result = runner.invoke(
        app,
        [
            "neuralforecast",
            "summarize-holdout",
            "--run-dir",
            str(tft),
            "--run-dir",
            str(nhits),
            "--out",
            str(output_dir),
            "--no-plot",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_dir.is_dir()
    assert "holdout summary written" in result.output


def test_training_history_does_not_fill_missing_validation(tmp_path: Path) -> None:
    group_dir = tmp_path / "__ALL__"
    group_dir.mkdir()
    tft = _write_holdout_run(group_dir, model="TFT", mae=2.0)

    history = collect_training_history([tft], ["TFT"])

    assert history["epoch"].to_list() == [1, 2]
    assert history["valid_loss"].to_list() == [2.5, None]
