"""Parametrized CPU train smokes for GluMind / SugarOne / GluMind-Uni.

One table drives argv construction, CSV writing, run-dir globs, and the
checkpoint config-key each trainer still uses.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch
from typer.testing import CliRunner

from tests.conftest import (
    TINY_TRAIN_SERIES,
    tiny_train_args,
    write_glumind_csv,
    write_sugar_one_csv,
)

runner = CliRunner()

def _import_uniglumind_app():
    uni_dir = str(Path(__file__).resolve().parents[1] / "scripts" / "glumind_uni")
    if uni_dir not in sys.path:
        sys.path.insert(0, uni_dir)
    from train_uniglumind import app  # noqa: PLC0415

    return app


@dataclass(frozen=True)
class TrainSmokeSpec:
    name: str
    csv_writer: Callable[..., None]
    run_glob: str
    config_key: str
    invoke: Literal["argparse", "typer"]
    flag_style: Literal["snake", "kebab"]
    app_loader: Callable[[], object] | None = None
    expect_tuning_meta: bool = False
    expect_test_metrics: bool = True
    expect_latest: bool = False
    stringify_out_dir: bool = False


def _load_sugar_one_app():
    from scripts.sugar_one.train_sugar_one import app

    return app


_SPECS = [
    TrainSmokeSpec(
        name="glumind",
        csv_writer=write_glumind_csv,
        run_glob="glumind_global_*",
        config_key="args",
        invoke="argparse",
        flag_style="snake",
        expect_tuning_meta=True,
        expect_latest=True,
        stringify_out_dir=True,
    ),
    TrainSmokeSpec(
        name="sugar_one",
        csv_writer=write_sugar_one_csv,
        run_glob="sugar_one_global_*",
        config_key="config",
        invoke="typer",
        flag_style="kebab",
        app_loader=_load_sugar_one_app,
    ),
    TrainSmokeSpec(
        name="glumind_uni",
        csv_writer=write_glumind_csv,
        run_glob="glumind_uni_global_*",
        config_key="cfg",
        invoke="typer",
        flag_style="kebab",
        app_loader=_import_uniglumind_app,
        expect_test_metrics=False,
    ),
]


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_train_cli_smoke_cpu(spec: TrainSmokeSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / f"{spec.name}_mini.csv"
    spec.csv_writer(csv_path, series=TINY_TRAIN_SERIES)
    out_dir = tmp_path / "runs"
    flag_args = tiny_train_args(spec.flag_style, out_dir)

    if spec.invoke == "argparse":
        from scripts.glumind.train_glumind import main

        monkeypatch.setattr(sys, "argv", [f"train_{spec.name}.py", "--csv", str(csv_path), *flag_args])
        main()
    else:
        assert spec.app_loader is not None
        result = runner.invoke(spec.app_loader(), ["--csv", str(csv_path), *flag_args])
        assert result.exit_code == 0, result.output

    run_dirs = [p for p in out_dir.glob(spec.run_glob) if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    for artifact in ("best_model.pt", "last_model.pt", "last_checkpoint.pt", "config.json", "val_metrics_overall.csv"):
        assert (run_dir / artifact).exists(), artifact
    if spec.expect_test_metrics:
        assert (run_dir / "test_metrics_overall.csv").exists()
    if spec.expect_tuning_meta:
        assert (run_dir / "tuning_meta.json").exists()
    if spec.expect_latest:
        latest = out_dir / "latest.txt"
        assert latest.exists()
        assert latest.read_text(encoding="utf-8").strip() == str(run_dir)

    ckpt = torch.load(run_dir / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert spec.config_key in ckpt
    assert int(ckpt["epoch"]) >= 0
    if spec.stringify_out_dir:
        assert isinstance(ckpt[spec.config_key]["out_dir"], str)
        assert ckpt[spec.config_key]["out_dir"] == str(out_dir)
