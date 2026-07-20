"""Live inference adapter for NeuralForecast bundle run directories.

Re-evaluates a saved NeuralForecast bundle on the same or different data,
producing a ``SingleModelResult`` with MAE/RMSE/MARD metrics.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from glucose_forecasting.evaluation.readers import read_precomputed_result
from glucose_forecasting.evaluation.types import RunDirKind, SingleModelResult


def evaluate_nf_run_dir(
    run_dir: Path,
    model_name: str,
    *,
    data: Path,
    output_dir: Path | None = None,
) -> SingleModelResult:
    """Re-evaluate a NeuralForecast bundle on *data*.

    Reads the original ``run_config.json``, overrides the CSV path with
    *data*, finds the saved bundle under ``run_dir/neuralforecast/``,
    and runs ``run_loaded_holdout`` into a fresh output directory.
    """
    from glucose_forecasting.backends.neuralforecast.config import NeuralForecastRunConfig
    from glucose_forecasting.backends.neuralforecast.evaluations.holdout import run_loaded_holdout

    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise ValueError(f"no run_config.json in {run_dir}")

    bundle_dir = run_dir / "neuralforecast"
    if not bundle_dir.is_dir():
        raise ValueError(f"no neuralforecast/ bundle directory in {run_dir}")

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["csv"] = str(data)
    config = NeuralForecastRunConfig(**raw)

    if output_dir is not None:
        eval_dir = output_dir / model_name
    else:
        eval_dir = Path(tempfile.mkdtemp(prefix=f"nf_eval_{model_name}_"))

    print(f"  re-evaluating NF bundle {model_name} on {data} → {eval_dir}", file=sys.stderr)
    run_loaded_holdout(config, bundle_dir=bundle_dir, run_dir=eval_dir)

    result = read_precomputed_result(eval_dir, model_name, kind=RunDirKind.NEURALFORECAST)
    return result
