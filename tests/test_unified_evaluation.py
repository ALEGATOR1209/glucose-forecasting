"""Tests for the unified run-directory evaluation pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from glucose_forecasting.evaluation.comparison import write_comparison_report
from glucose_forecasting.evaluation.detect import detect_run_dir, infer_model_name
from glucose_forecasting.evaluation.readers import read_precomputed_result
from glucose_forecasting.evaluation.runner import evaluate_run_dir
from glucose_forecasting.evaluation.types import (
    RunDirKind,
    SingleModelResult,
    SplitMetrics,
)
from glucose_forecasting.backends.neuralforecast.benchmark import RegressionMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def nf_run_dir(tmp_path: Path) -> Path:
    """A minimal NeuralForecast-style run directory."""
    run = tmp_path / "TFT_20260718T223910Z"
    run.mkdir()
    (run / "neuralforecast").mkdir()
    (run / "run_config.json").write_text(
        json.dumps({"selected_models": ["TFT"], "evaluation": "holdout"}),
        encoding="utf-8",
    )
    pl.DataFrame({"mae": [20.6], "rmse": [30.7], "mard": [15.8]}).write_csv(
        run / "test_metrics_overall.csv"
    )
    pl.DataFrame(
        {
            "study_group": ["healthy", "T1DM"],
            "n_points": [36, 1213],
            "mae": [7.7, 22.4],
            "rmse": [9.1, 32.7],
            "mard": [7.4, 17.9],
        }
    ).write_csv(run / "test_metrics_by_study_group.csv")
    return run


@pytest.fixture()
def pytorch_run_dir(tmp_path: Path) -> Path:
    """A minimal custom PyTorch-style run directory with ``n_windows``."""
    run = tmp_path / "sugar_jepa_dev"
    run.mkdir()
    (run / "best_model.pt").write_bytes(b"fake")
    (run / "tuning_meta.json").write_text(
        json.dumps({"model_type": "sugar_jepa", "csv": "data.csv"}),
        encoding="utf-8",
    )
    pl.DataFrame({"mae": [21.6], "rmse": [31.9], "mard": [17.0]}).write_csv(
        run / "test_metrics_overall.csv"
    )
    pl.DataFrame(
        {
            "study_group": ["healthy", "T1DM"],
            "n_windows": [4880, 29666],
            "mae": [11.5, 26.8],
            "rmse": [17.0, 37.6],
            "mard": [10.2, 22.8],
        }
    ).write_csv(run / "test_metrics_by_study_group.csv")
    return run


@pytest.fixture()
def precomputed_run_dir(tmp_path: Path) -> Path:
    """A run directory with only metrics CSVs (no checkpoint or config)."""
    run = tmp_path / "some_model"
    run.mkdir()
    pl.DataFrame({"mae": [25.0], "rmse": [35.0], "mard": [20.0]}).write_csv(
        run / "test_metrics_overall.csv"
    )
    return run


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestDetectRunDir:
    def test_neuralforecast(self, nf_run_dir: Path) -> None:
        assert detect_run_dir(nf_run_dir) == RunDirKind.NEURALFORECAST

    def test_custom_pytorch(self, pytorch_run_dir: Path) -> None:
        assert detect_run_dir(pytorch_run_dir) == RunDirKind.CUSTOM_PYTORCH

    def test_precomputed(self, precomputed_run_dir: Path) -> None:
        assert detect_run_dir(precomputed_run_dir) == RunDirKind.PRECOMPUTED

    def test_unknown_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="cannot detect"):
            detect_run_dir(empty)

    def test_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            detect_run_dir(tmp_path / "nope")


class TestInferModelName:
    def test_nf_from_config(self, nf_run_dir: Path) -> None:
        assert infer_model_name(nf_run_dir, RunDirKind.NEURALFORECAST) == "TFT"

    def test_nf_from_dirname(self, tmp_path: Path) -> None:
        run = tmp_path / "NHITS_20260718T222107Z"
        run.mkdir()
        (run / "neuralforecast").mkdir()
        (run / "run_config.json").write_text("{}", encoding="utf-8")
        assert infer_model_name(run, RunDirKind.NEURALFORECAST) == "NHITS"

    def test_pytorch_from_meta(self, pytorch_run_dir: Path) -> None:
        assert infer_model_name(pytorch_run_dir, RunDirKind.CUSTOM_PYTORCH) == "sugar_jepa"

    def test_label_takes_precedence(self, nf_run_dir: Path) -> None:
        assert infer_model_name(nf_run_dir, RunDirKind.NEURALFORECAST, label="MyModel") == "MyModel"


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


class TestReadPrecomputedResult:
    def test_reads_overall_metrics(self, nf_run_dir: Path) -> None:
        result = read_precomputed_result(nf_run_dir, "TFT")
        assert "test" in result.split_results
        assert result.split_results["test"].overall.mae == pytest.approx(20.6, abs=0.1)

    def test_normalizes_n_windows_to_n_points(self, pytorch_run_dir: Path) -> None:
        result = read_precomputed_result(pytorch_run_dir, "SugarJEPA")
        group_frame = result.split_results["test"].by_study_group
        assert "n_points" in group_frame.columns
        assert "n_windows" not in group_frame.columns

    def test_preserves_n_points_when_present(self, nf_run_dir: Path) -> None:
        result = read_precomputed_result(nf_run_dir, "TFT")
        group_frame = result.split_results["test"].by_study_group
        assert "n_points" in group_frame.columns

    def test_no_metrics_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_run"
        empty.mkdir()
        with pytest.raises(ValueError, match="no precomputed metrics"):
            read_precomputed_result(empty, "Model")


# ---------------------------------------------------------------------------
# Comparison report tests
# ---------------------------------------------------------------------------


class TestWriteComparisonReport:
    def _make_result(self, name: str, mae: float, run_dir: Path) -> SingleModelResult:
        return SingleModelResult(
            model_name=name,
            run_dir=run_dir,
            kind=RunDirKind.PRECOMPUTED,
            split_results={
                "test": SplitMetrics(
                    overall=RegressionMetrics(mae=mae, rmse=mae * 1.4, mard=mae * 0.8),
                    by_study_group=pl.DataFrame(
                        {
                            "study_group": ["healthy", "T1DM"],
                            "n_points": [100, 500],
                            "mae": [mae * 0.5, mae * 1.2],
                            "rmse": [mae * 0.7, mae * 1.6],
                            "mard": [mae * 0.3, mae * 1.0],
                        }
                    ),
                ),
            },
        )

    def test_produces_summary_csvs(self, tmp_path: Path) -> None:
        results = [
            self._make_result("ModelA", 20.0, tmp_path / "a"),
            self._make_result("ModelB", 25.0, tmp_path / "b"),
        ]
        for r in results:
            r.run_dir.mkdir(exist_ok=True)
        out = tmp_path / "report"
        write_comparison_report(results, out, plot=False)
        assert (out / "test_metrics_summary.csv").is_file()
        assert (out / "study_group_metrics.csv").is_file()
        assert (out / "run_manifest.json").is_file()

    def test_summary_csv_sorted_by_mae(self, tmp_path: Path) -> None:
        results = [
            self._make_result("Worse", 30.0, tmp_path / "w"),
            self._make_result("Better", 15.0, tmp_path / "b"),
        ]
        for r in results:
            r.run_dir.mkdir(exist_ok=True)
        out = tmp_path / "report"
        write_comparison_report(results, out, plot=False)
        summary = pl.read_csv(out / "test_metrics_summary.csv")
        assert summary["model"].to_list() == ["Better", "Worse"]

    def test_study_group_csv_schema(self, tmp_path: Path) -> None:
        results = [self._make_result("M", 20.0, tmp_path / "m")]
        results[0].run_dir.mkdir(exist_ok=True)
        out = tmp_path / "report"
        write_comparison_report(results, out, plot=False)
        sg = pl.read_csv(out / "study_group_metrics.csv")
        assert set(sg.columns) == {"split", "model", "study_group", "n_points", "mae", "rmse", "mard"}

    def test_empty_results_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one result"):
            write_comparison_report([], tmp_path / "report", plot=False)


# ---------------------------------------------------------------------------
# End-to-end evaluate_run_dir tests
# ---------------------------------------------------------------------------


class TestEvaluateRunDir:
    def test_precomputed_nf(self, nf_run_dir: Path) -> None:
        result = evaluate_run_dir(nf_run_dir)
        assert result.model_name == "TFT"
        assert result.split_results["test"].overall.mae == pytest.approx(20.6, abs=0.1)

    def test_precomputed_pytorch(self, pytorch_run_dir: Path) -> None:
        result = evaluate_run_dir(pytorch_run_dir, label="SugarJEPA")
        assert result.model_name == "SugarJEPA"
        assert "test" in result.split_results

    def test_no_data_no_metrics_raises(self, tmp_path: Path) -> None:
        run = tmp_path / "empty"
        run.mkdir()
        (run / "best_model.pt").write_bytes(b"fake")
        (run / "tuning_meta.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="no precomputed metrics"):
            evaluate_run_dir(run)

    def test_data_dispatches_pytorch_adapter(self, pytorch_run_dir: Path, tmp_path: Path, monkeypatch) -> None:
        """When --data is given for a CUSTOM_PYTORCH dir, dispatch to pytorch_adapter."""
        fake_csv = tmp_path / "data.csv"
        fake_csv.write_text("col\n1\n")
        sentinel = SingleModelResult(
            model_name="mock_sugar_jepa",
            run_dir=pytorch_run_dir,
            kind=RunDirKind.CUSTOM_PYTORCH,
            split_results={"test": SplitMetrics(
                overall=RegressionMetrics(mae=1.0, rmse=2.0, mard=3.0),
                by_study_group=pl.DataFrame(),
            )},
        )

        def mock_adapter(run_dir, model_name, *, data, train_data=None, device="auto", output_dir=None):
            return sentinel

        monkeypatch.setattr(
            "glucose_forecasting.evaluation.runner.evaluate_pytorch_run_dir",
            mock_adapter,
            raising=False,
        )
        import glucose_forecasting.evaluation.pytorch_adapter as pa_mod
        monkeypatch.setattr(pa_mod, "evaluate_pytorch_run_dir", mock_adapter)

        from glucose_forecasting.evaluation import runner as runner_mod
        monkeypatch.setattr(
            runner_mod,
            "_evaluate_with_inference",
            runner_mod._evaluate_with_inference,
        )

        result = evaluate_run_dir(pytorch_run_dir, data=fake_csv)
        assert result.model_name == "mock_sugar_jepa"
        assert result.split_results["test"].overall.mae == 1.0

    def test_data_dispatches_nf_adapter(self, nf_run_dir: Path, tmp_path: Path, monkeypatch) -> None:
        """When --data is given for a NEURALFORECAST dir, dispatch to nf_adapter."""
        fake_csv = tmp_path / "data.csv"
        fake_csv.write_text("col\n1\n")
        sentinel = SingleModelResult(
            model_name="mock_TFT",
            run_dir=nf_run_dir,
            kind=RunDirKind.NEURALFORECAST,
            split_results={"test": SplitMetrics(
                overall=RegressionMetrics(mae=5.0, rmse=6.0, mard=7.0),
                by_study_group=pl.DataFrame(),
            )},
        )

        def mock_adapter(run_dir, model_name, *, data, output_dir=None):
            return sentinel

        import glucose_forecasting.evaluation.nf_adapter as nf_mod
        monkeypatch.setattr(nf_mod, "evaluate_nf_run_dir", mock_adapter)

        result = evaluate_run_dir(nf_run_dir, data=fake_csv)
        assert result.model_name == "mock_TFT"
        assert result.split_results["test"].overall.mae == 5.0

    def test_precomputed_fallback_without_data(self, pytorch_run_dir: Path) -> None:
        """Without --data, reads precomputed metrics even for CUSTOM_PYTORCH dirs."""
        result = evaluate_run_dir(pytorch_run_dir)
        assert result.model_name == "sugar_jepa"
        assert result.split_results["test"].overall.mae == pytest.approx(21.6, abs=0.1)

    def test_train_data_kwarg_passes_through(self, pytorch_run_dir: Path, tmp_path: Path, monkeypatch) -> None:
        """--train-data is passed through to the pytorch adapter."""
        fake_csv = tmp_path / "data.csv"
        fake_csv.write_text("col\n1\n")
        train_csv = tmp_path / "train.csv"
        train_csv.write_text("col\n1\n")
        received = {}

        def mock_adapter(run_dir, model_name, *, data, train_data=None, device="auto", output_dir=None):
            received["train_data"] = train_data
            return SingleModelResult(
                model_name=model_name,
                run_dir=run_dir,
                kind=RunDirKind.CUSTOM_PYTORCH,
                split_results={"test": SplitMetrics(
                    overall=RegressionMetrics(mae=1.0, rmse=2.0, mard=3.0),
                    by_study_group=pl.DataFrame(),
                )},
            )

        import glucose_forecasting.evaluation.pytorch_adapter as pa_mod
        monkeypatch.setattr(pa_mod, "evaluate_pytorch_run_dir", mock_adapter)

        evaluate_run_dir(pytorch_run_dir, data=fake_csv, train_data=train_csv)
        assert received["train_data"] == train_csv
