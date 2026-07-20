"""End-to-end smoke tests chaining a tiny CPU-trained checkpoint into the
evaluate-model and evaluate-glumind Typer CLIs, asserting exit 0 and that
MAE/RMSE/MARD are printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.sugar_one.train_sugar_one import app as train_sugar_one_app
from tests.conftest import (
    TINY_TRAIN_SERIES,
    tiny_train_args,
    write_glumind_csv,
    write_sugar_one_csv,
)

runner = CliRunner()


def _train_sugar_one(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "sugar_one_mini.csv"
    write_sugar_one_csv(csv_path, series=TINY_TRAIN_SERIES)
    out_dir = tmp_path / "runs"
    result = runner.invoke(
        train_sugar_one_app,
        ["--csv", str(csv_path), *tiny_train_args("kebab", out_dir)],
    )
    assert result.exit_code == 0, result.output
    run_dirs = [p for p in out_dir.glob("sugar_one_global_*") if p.is_dir()]
    assert len(run_dirs) == 1
    return csv_path, run_dirs[0]


def test_evaluate_model_cli_smoke(tmp_path: Path) -> None:
    from scripts.sugar_one.evaluate_model import app as evaluate_model_app

    csv_path, run_dir = _train_sugar_one(tmp_path)

    result = runner.invoke(
        evaluate_model_app,
        [
            "--test-csv", str(csv_path),
            "--run-dir", str(run_dir),
            "--model-type", "sugar_one",
            "--test-split", "test",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "MAE" in result.output
    assert "RMSE" in result.output
    assert "MARD" in result.output


def test_evaluate_glumind_cli_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.glumind.evaluate_glumind import app as evaluate_glumind_app
    from scripts.glumind.inference_glumind import app as inference_glumind_app
    from scripts.glumind.train_glumind import main as train_glumind_main
    csv_path = tmp_path / "glumind_mini.csv"
    write_glumind_csv(csv_path, series=TINY_TRAIN_SERIES)
    out_dir = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_glumind.py",
            "--csv",
            str(csv_path),
            *tiny_train_args("snake", out_dir),
        ],
    )
    train_glumind_main()

    run_dirs = [p for p in out_dir.glob("glumind_global_*") if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    result = runner.invoke(
        evaluate_glumind_app,
        [
            "--run-dir", str(run_dir),
            "--test-csv", str(csv_path),
            "--test-split", "test",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "MAE" in result.output
    assert "RMSE" in result.output
    assert "MARD" in result.output

    result = runner.invoke(
        inference_glumind_app,
        ["--run-dir", str(run_dir), "--device", "cpu"],
    )
    assert result.exit_code == 0, result.output
    assert "MAE" in result.output
    assert "RMSE" in result.output
    assert "MARD" in result.output
