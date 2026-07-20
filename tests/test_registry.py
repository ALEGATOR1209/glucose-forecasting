"""Unit tests for scripts/common/registry.py."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from scripts.common.registry import (
    find_best_run_dir,
    resolve_csv_path,
)


def _write_registry_csv(path: Path, rows: list[dict[str, str]]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_find_best_run_dir_picks_lowest_metric_and_final_step(tmp_path: Path) -> None:
    project_root = tmp_path
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _write_registry_csv(
        registry_dir / "_analysis_registry.csv",
        [
            {"run_dir": "runs/a", "val_mae": "5.0"},
            {
                "run_dir": "runs/b",
                "val_mae": "2.0",
                "final_step": "step_03_T1DM",
            },
            {"run_dir": "runs/c", "val_mae": "3.0"},
        ],
    )
    step_dir, row = find_best_run_dir(registry_dir, project_root)
    assert step_dir == project_root / "runs" / "b" / "step_03_T1DM"
    assert row["run_dir"] == "runs/b"


def test_resolve_csv_path_accepts_existing_absolute_and_project_paths(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n")
    project_root = tmp_path / "root"
    project_root.mkdir()
    rooted_csv = project_root / "rooted.csv"
    rooted_csv.write_text("a,b\n1,2\n")
    assert resolve_csv_path(csv_path, project_root) == csv_path
    assert resolve_csv_path("rooted.csv", project_root) == rooted_csv


@pytest.mark.parametrize(
    "recorded_path",
    [
        r"D:\01_1_LIVIA\sources\glucose-forecasting\data\loop_and_ai_ready\loop_ai_ready_joined2.csv",
        "data/loop_and_ai_ready/loop_ai_ready_joined2.csv",
    ],
)
def test_resolve_csv_path_remaps_legacy_locations(
    tmp_path: Path,
    recorded_path: str,
) -> None:
    project_root = tmp_path / "root"
    input_dir = project_root / "data" / "input"
    input_dir.mkdir(parents=True)
    local = input_dir / "loop_ai_ready_joined2.csv"
    local.write_text("a,b\n1,2\n")
    assert resolve_csv_path(recorded_path, project_root) == local
