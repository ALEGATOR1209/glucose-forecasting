"""Fixed-split held-out NeuralForecast evaluation."""
from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch
from eliot import start_action

from glucose_forecasting.backends.neuralforecast.adapter import (
    PreparedSplits,
    filter_minimum_length,
    prepare_splits,
)
from glucose_forecasting.backends.neuralforecast.benchmark import (
    calculate_metrics,
    to_neuralforecast_frame,
)
from glucose_forecasting.backends.neuralforecast.catalog import (
    ModelDefinition,
    create_model,
    resolve_models,
)
from glucose_forecasting.backends.neuralforecast.config import (
    ModelSuiteConfig,
    NeuralForecastRunConfig,
    frequency_minutes,
    neuralforecast_frequency,
)
from glucose_forecasting.backends.neuralforecast.plotting import (
    write_comparison_dashboard,
    write_prediction_charts,
)
from glucose_forecasting.backends.neuralforecast.reporting import (
    write_holdout_report_visuals,
)
from glucose_forecasting.backends.neuralforecast.telemetry import (
    EpochProgressCallback,
    announce,
    configure_run_logs,
)
from glucose_forecasting.common.data_loading import limit_series


def resolve_device(requested: str) -> str:
    """Select a supported accelerator, preferring CUDA in automatic mode."""
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if not available[requested]:
        raise ValueError(f"requested device {requested!r} is not available")
    return requested


def run_holdout(
    config: NeuralForecastRunConfig,
    *,
    suites: ModelSuiteConfig,
    suites_yaml: str,
) -> list[Path]:
    """Fit selected models and evaluate fixed validation/test CSV splits."""
    if config.evaluation != "holdout":
        raise ValueError("run_holdout requires evaluation='holdout'")
    splits = prepare_splits(
        config.csv,
        profile_name=config.profile,
        unique_id_choice=config.unique_id,
        split_scheme=config.split_scheme,
        drop_interpolated=config.drop_interpolated,
        max_train_series=config.max_train_series,
        max_points_per_series=config.max_points_per_series,
    )
    resolved_profile = splits.profile.name
    models = _resolve_selected_models(config, suites, resolved_profile)
    device = resolve_device(config.device)
    horizon, input_size, validation_size = _window_sizes(config)
    prepared = PreparedSplits(
        profile=splits.profile,
        train=filter_minimum_length(splits.train, input_size + validation_size + horizon),
        validation=filter_minimum_length(splits.validation, input_size + horizon),
        test=filter_minimum_length(splits.test, input_size + horizon),
    )
    if prepared.train.is_empty():
        raise ValueError("no training series remain after minimum-length filtering")

    groups = _select_groups(prepared, config)
    runs: list[Path] = []
    for group in groups:
        group_splits = _for_group(prepared, group)
        group_runs: list[Path] = []
        for definition in models:
            group_runs.append(
                _fit_one(
                    config,
                    definition=definition,
                    splits=group_splits,
                    group=group,
                    device=device,
                    horizon=horizon,
                    input_size=input_size,
                    validation_size=validation_size,
                    suites_yaml=suites_yaml,
                )
            )
        runs.extend(group_runs)
        if len(group_runs) > 1:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            summary_dir = config.out_dir / "nf_holdout" / group / "summaries" / timestamp
            summarize_holdout_runs(
                group_runs,
                output_dir=summary_dir,
                plot=config.plot,
            )
            announce(
                f"Holdout summary written to {summary_dir}.",
                group=group,
                run_directory=str(summary_dir),
            )
            runs.append(summary_dir)
    return runs


def summarize_holdout_runs(
    run_dirs: list[Path],
    *,
    output_dir: Path,
    plot: bool,
) -> Path:
    """Combine compatible per-model fixed-split holdout artifacts into one report."""
    if not run_dirs:
        raise ValueError("at least one holdout run directory is required")
    if len({run_dir.parent for run_dir in run_dirs}) != 1:
        raise ValueError("holdout summary requires runs from the same study-group directory")
    source_configs = [_load_run_config(run_dir) for run_dir in run_dirs]
    _validate_compatible_run_configs(source_configs)
    models = [_model_name(run_dir) for run_dir in run_dirs]
    if len(models) != len(set(models)):
        raise ValueError("holdout summary requires at most one run per model")

    output_dir.mkdir(parents=True, exist_ok=False)
    for split_name in ("val", "test"):
        metrics = _read_split_metrics(run_dirs, models, split_name)
        metrics.write_csv(output_dir / f"{split_name}_metrics_summary.csv")
        predictions = _read_split_predictions(run_dirs, models, split_name)
        predictions.write_csv(output_dir / f"{split_name}_predictions.csv")
        if plot and len(models) > 1:
            _write_split_dashboard(
                predictions,
                models=models,
                split_name=split_name,
                output_dir=output_dir,
            )

    common_config = dict(source_configs[0])
    common_config["selected_models"] = models
    common_config["source_runs"] = [str(run_dir) for run_dir in run_dirs]
    (output_dir / "run_config.json").write_text(
        json.dumps(common_config, indent=2),
        encoding="utf-8",
    )
    (output_dir / "model_runs.json").write_text(
        json.dumps(
            [
                {
                    "model": model,
                    "run_dir": str(run_dir),
                    "run_config": config,
                }
                for model, run_dir, config in zip(models, run_dirs, source_configs, strict=True)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    write_holdout_report_visuals(output_dir, run_dirs=run_dirs, models=models)
    return output_dir


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load one saved per-model holdout configuration."""
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise ValueError(f"run config not found: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid run config: {config_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"run config must contain an object: {config_path}")
    if payload.get("evaluation") != "holdout":
        raise ValueError(f"run is not a holdout evaluation: {run_dir}")
    return payload


def _validate_compatible_run_configs(configs: list[dict[str, Any]]) -> None:
    """Require summary candidates to share every evaluation-affecting setting."""
    ignored_fields = {"models", "out_dir", "plot", "max_plot_series"}
    reference = {key: value for key, value in configs[0].items() if key not in ignored_fields}
    for config in configs[1:]:
        candidate = {key: value for key, value in config.items() if key not in ignored_fields}
        if candidate != reference:
            raise ValueError(
                "holdout runs have incompatible configurations; choose runs with the same "
                "data, split, protocol, geometry, and evaluation settings"
            )


def _model_name(run_dir: Path) -> str:
    """Infer the saved model name from the standard holdout run directory."""
    model_name, separator, _ = run_dir.name.rpartition("_")
    if not separator or not model_name:
        raise ValueError(f"holdout run directory must end with _<UTC timestamp>: {run_dir}")
    return model_name


def _read_split_metrics(
    run_dirs: list[Path],
    models: list[str],
    split_name: str,
) -> pl.DataFrame:
    """Read and sort one overall-metrics row per model."""
    rows = [
        pl.read_csv(run_dir / f"{split_name}_metrics_overall.csv").with_columns(
            pl.lit(model).alias("model")
        )
        for model, run_dir in zip(models, run_dirs, strict=True)
    ]
    return pl.concat(rows, how="vertical").select(["model", "mae", "rmse", "mard"]).sort("mae")


def _read_split_predictions(
    run_dirs: list[Path],
    models: list[str],
    split_name: str,
) -> pl.DataFrame:
    """Read model predictions as one long-form comparison frame."""
    frames = [
        pl.read_csv(run_dir / f"{split_name}_predictions.csv", try_parse_dates=True).with_columns(
            pl.lit(model).alias("model")
        )
        for model, run_dir in zip(models, run_dirs, strict=True)
    ]
    return pl.concat(frames, how="vertical")


def _write_split_dashboard(
    predictions: pl.DataFrame,
    *,
    models: list[str],
    split_name: str,
    output_dir: Path,
) -> Path:
    """Write a dashboard comparing every model on one holdout split."""
    actual = predictions.select(["unique_id", "ds", "y"]).unique(
        subset=["unique_id", "ds"],
        keep="first",
        maintain_order=True,
    )
    predictions_by_model = {
        model: predictions.filter(pl.col("model") == model).drop("model").rename({"yhat": model})
        for model in models
    }
    return write_comparison_dashboard(
        actual,
        predictions_by_model,
        output_dir=output_dir,
        max_sequences=3,
        dashboard_subdir=f"diagnostic_examples/comparison/{split_name}",
    )


def run_loaded_holdout(
    config: NeuralForecastRunConfig,
    *,
    bundle_dir: Path,
    run_dir: Path,
) -> Path:
    """Evaluate one saved NeuralForecast bundle without fitting it again."""
    from neuralforecast import NeuralForecast

    neuralforecast = NeuralForecast.load(str(bundle_dir))
    neuralforecast.freq = neuralforecast_frequency(config.freq)
    _validate_loaded_bundle(neuralforecast, config)
    splits = prepare_splits(
        config.csv,
        profile_name=config.profile,
        unique_id_choice=config.unique_id,
        split_scheme=config.split_scheme,
        drop_interpolated=config.drop_interpolated,
        max_train_series=config.max_train_series,
        max_points_per_series=config.max_points_per_series,
    )
    horizon, input_size, _ = _window_sizes(config)
    prepared = PreparedSplits(
        profile=splits.profile,
        train=splits.train,
        validation=filter_minimum_length(splits.validation, input_size + horizon),
        test=filter_minimum_length(splits.test, input_size + horizon),
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    configure_run_logs(run_dir)
    for split_name, frame in (("val", prepared.validation), ("test", prepared.test)):
        _evaluate_split(
            neuralforecast,
            split_name=split_name,
            frame=frame,
            profile_exogenous=prepared.profile.historical_exogenous,
            horizon=horizon,
            input_size=input_size,
            run_dir=run_dir,
            config=config,
        )
    (run_dir / "run_config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "source_bundle.txt").write_text(f"{bundle_dir}\n", encoding="utf-8")
    (run_dir / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "source_bundle": str(bundle_dir),
                "holdout_protocol": config.holdout_protocol,
                "sugarone_comparable": config.holdout_protocol == "sugarone-compatible",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def _fit_one(
    config: NeuralForecastRunConfig,
    *,
    definition: ModelDefinition,
    splits: PreparedSplits,
    group: str,
    device: str,
    horizon: int,
    input_size: int,
    validation_size: int,
    suites_yaml: str,
) -> Path:
    from neuralforecast import NeuralForecast
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.out_dir / "nf_holdout" / group / f"{definition.name.value}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = configure_run_logs(run_dir)
    announce(
        f"Training {definition.name.value} for group {group} on {device}.",
        model=definition.name.value,
        group=group,
        device=device,
        max_steps=config.max_steps,
        log_directory=str(logs_dir),
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="step-{step}",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer_kwargs: dict[str, Any] = {
        "accelerator": device,
        "devices": 1,
        "default_root_dir": str(run_dir),
        "callbacks": [
            checkpoint,
            EarlyStopping(monitor="valid_loss", mode="min", patience=10),
            EpochProgressCallback(model_name=definition.name.value),
        ],
        "enable_checkpointing": True,
        "logger": False,
        "enable_progress_bar": True,
    }
    model = create_model(
        definition,
        horizon=horizon,
        input_size=input_size,
        max_steps=config.max_steps,
        historical_exogenous=splits.profile.historical_exogenous,
        trainer_kwargs=trainer_kwargs,
        learning_rate=config.learning_rate,
        val_check_steps=config.val_check_steps,
        batch_size=config.batch_size,
        valid_batch_size=config.valid_batch_size,
        windows_batch_size=config.windows_batch_size,
        inference_windows_batch_size=config.inference_windows_batch_size,
        step_size=config.step_size,
    )
    neuralforecast = NeuralForecast(models=[model], freq=neuralforecast_frequency(config.freq))
    with start_action(
        action_type="neuralforecast_fit",
        model=definition.name.value,
        group=group,
        max_steps=config.max_steps,
    ):
        neuralforecast.fit(
            df=to_neuralforecast_frame(splits.train, splits.profile.historical_exogenous),
            val_size=validation_size,
        )
    announce(
        f"Finished training {definition.name.value}; saving model bundle.",
        model=definition.name.value,
        group=group,
    )
    neuralforecast.save(str(run_dir / "neuralforecast"), overwrite=True)
    for split_name, frame in (("val", splits.validation), ("test", splits.test)):
        announce(
            f"Evaluating {definition.name.value} on {split_name}.",
            model=definition.name.value,
            group=group,
            split=split_name,
        )
        evaluated = _evaluate_split(
            neuralforecast,
            split_name=split_name,
            frame=frame,
            profile_exogenous=splits.profile.historical_exogenous,
            horizon=horizon,
            input_size=input_size,
            run_dir=run_dir,
            config=config,
        )
        if config.plot and evaluated is not None:
            announce(
                f"Writing {split_name} charts for {definition.name.value}.",
                model=definition.name.value,
                group=group,
                split=split_name,
            )
            write_prediction_charts(
                frame,
                evaluated,
                model_name=definition.name.value,
                output_dir=run_dir,
                max_sequences=config.max_plot_series,
                title_prefix=f"{split_name.title()} — ",
                split_name=split_name,
            )
    (run_dir / "run_config.json").write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_dir / "model_suites.yaml").write_text(suites_yaml, encoding="utf-8")
    announce(
        f"Completed {definition.name.value}; artifacts: {run_dir}",
        model=definition.name.value,
        group=group,
        run_directory=str(run_dir),
    )
    return run_dir


def _evaluate_split(
    neuralforecast: Any,
    *,
    split_name: str,
    frame: pl.DataFrame,
    profile_exogenous: tuple[str, ...],
    horizon: int,
    input_size: int,
    run_dir: Path,
    config: NeuralForecastRunConfig,
) -> pl.DataFrame | None:
    frame = limit_series(frame, config.max_eval_series)
    frame = filter_minimum_length(frame, input_size + horizon)
    if frame.is_empty():
        return None
    if config.holdout_protocol in {"sugarone-compatible", "dense"}:
        evaluated = _evaluate_dense_split(
            neuralforecast,
            frame=frame,
            profile_exogenous=profile_exogenous,
            input_size=input_size,
        )
    else:
        evaluated = _evaluate_tail_split(
            neuralforecast,
            frame=frame,
            profile_exogenous=profile_exogenous,
            horizon=horizon,
        )
    if config.mask_interpolated_targets:
        evaluated = evaluated.filter(pl.col("event_type") != "Interpolated")
    if evaluated.is_empty():
        return None
    metrics = calculate_metrics(evaluated)
    announce(
        (
            f"{split_name} metrics: MAE={metrics.overall.mae:.3f}, "
            f"RMSE={metrics.overall.rmse:.3f}, MARD={metrics.overall.mard:.3f}%."
        ),
        split=split_name,
        mae=metrics.overall.mae,
        rmse=metrics.overall.rmse,
        mard=metrics.overall.mard,
    )
    pl.DataFrame(
        [{
            "mae": metrics.overall.mae,
            "rmse": metrics.overall.rmse,
            "mard": metrics.overall.mard,
        }]
    ).write_csv(run_dir / f"{split_name}_metrics_overall.csv")
    metrics.by_study_group.write_csv(run_dir / f"{split_name}_metrics_by_study_group.csv")
    evaluated.write_csv(run_dir / f"{split_name}_predictions.csv")
    return evaluated


def _evaluate_tail_split(
    neuralforecast: Any,
    *,
    frame: pl.DataFrame,
    profile_exogenous: tuple[str, ...],
    horizon: int,
) -> pl.DataFrame:
    """Score one final horizon per series for legacy experimental holdouts."""
    truth = frame.group_by("unique_id", maintain_order=True).tail(horizon)
    history = frame.join(
        truth.select(["unique_id", "ds"]),
        on=["unique_id", "ds"],
        how="anti",
    )
    predictions = neuralforecast.predict(
        df=to_neuralforecast_frame(history, profile_exogenous)
    )
    prediction_columns = [column for column in predictions.columns if column not in {"unique_id", "ds"}]
    if len(prediction_columns) != 1:
        raise RuntimeError("expected exactly one prediction column for one selected model")
    predicted = _as_polars(predictions).rename({prediction_columns[0]: "yhat"})
    return truth.join(predicted, on=["unique_id", "ds"], how="inner")


def _evaluate_dense_split(
    neuralforecast: Any,
    *,
    frame: pl.DataFrame,
    profile_exogenous: tuple[str, ...],
    input_size: int,
) -> pl.DataFrame:
    """Score every valid stride-1 forecast origin, matching SugarOne evaluation."""
    series_lengths = frame.group_by("unique_id").len()
    evaluated_parts: list[pl.DataFrame] = []
    for length_value in sorted(series_lengths["len"].unique().to_list()):
        length = int(length_value)
        series_ids = series_lengths.filter(pl.col("len") == length).select("unique_id")
        same_length_frame = frame.join(series_ids, on="unique_id", how="semi")
        predictions = neuralforecast.cross_validation(
            df=to_neuralforecast_frame(same_length_frame, profile_exogenous),
            n_windows=None,
            test_size=length - input_size,
            step_size=1,
            use_fitted=True,
            refit=False,
            verbose=False,
        )
        prediction_columns = [
            column
            for column in predictions.columns
            if column not in {"unique_id", "ds", "cutoff", "y"}
        ]
        if len(prediction_columns) != 1:
            raise RuntimeError("expected exactly one prediction column for one selected model")
        predicted = (
            _as_polars(predictions)
            .select(["unique_id", "ds", "cutoff", prediction_columns[0]])
            .rename({prediction_columns[0]: "yhat", "cutoff": "forecast_origin"})
        )
        metadata = same_length_frame.select(["unique_id", "ds", "y", "study_group", "event_type"])
        evaluated_parts.append(metadata.join(predicted, on=["unique_id", "ds"], how="inner"))
    return pl.concat(evaluated_parts, how="vertical")


def _validate_loaded_bundle(neuralforecast: Any, config: NeuralForecastRunConfig) -> None:
    """Reject a saved model whose immutable geometry differs from its run metadata."""
    horizon, input_size, _ = _window_sizes(config)
    models = getattr(neuralforecast, "models", [])
    if len(models) != 1:
        raise ValueError("saved evaluation bundles must contain exactly one NeuralForecast model")
    model = models[0]
    if getattr(model, "h", None) != horizon or getattr(model, "input_size", None) != input_size:
        raise ValueError(
            "saved NeuralForecast bundle geometry does not match run_config.json; "
            "input size and horizon cannot be changed during evaluation"
        )


def _as_polars(frame: Any) -> pl.DataFrame:
    """Convert third-party output only when it is not already a Polars frame."""
    return frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)


def _resolve_selected_models(
    config: NeuralForecastRunConfig,
    suites: ModelSuiteConfig,
    profile: str,
) -> tuple[ModelDefinition, ...]:
    suite_models = {
        name: suite.models
        for name, suite in suites.suites.items()
        if profile in suite.profiles
    }
    if config.models in suites.suites and config.models not in suite_models:
        raise ValueError(f"model suite {config.models!r} is not compatible with {profile!r}")
    return resolve_models(config.models, suite_models=suite_models)


def _window_sizes(config: NeuralForecastRunConfig) -> tuple[int, int, int]:
    step_minutes = frequency_minutes(config.freq)
    if config.h_minutes % step_minutes:
        raise ValueError(f"h_minutes={config.h_minutes} is not divisible by freq={config.freq}")
    return (
        config.h_minutes // step_minutes,
        round(config.input_hours * 60 / step_minutes),
        round(config.train_tail_val_hours * 60 / step_minutes),
    )


def _select_groups(splits: PreparedSplits, config: NeuralForecastRunConfig) -> tuple[str, ...]:
    if config.global_model:
        return ("__ALL__",)
    all_groups = tuple(sorted(splits.train["study_group"].unique().cast(pl.String).to_list()))
    selected = set(config.study_groups)
    groups = tuple(group for group in all_groups if not selected or group in selected)
    if not groups:
        raise ValueError("no study groups selected after filtering")
    return groups


def _for_group(splits: PreparedSplits, group: str) -> PreparedSplits:
    if group == "__ALL__":
        return splits
    return PreparedSplits(
        profile=splits.profile,
        train=splits.train.filter(pl.col("study_group") == group),
        validation=splits.validation.filter(pl.col("study_group") == group),
        test=splits.test.filter(pl.col("study_group") == group),
    )
