"""Rolling cross-validation NeuralForecast evaluation."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from eliot import start_action

from glucose_forecasting.backends.neuralforecast.adapter import prepare_splits
from glucose_forecasting.backends.neuralforecast.benchmark import calculate_metrics
from glucose_forecasting.backends.neuralforecast.catalog import create_model
from glucose_forecasting.backends.neuralforecast.config import (
    ModelSuiteConfig,
    NeuralForecastRunConfig,
)
from glucose_forecasting.backends.neuralforecast.evaluations.holdout import (
    _resolve_selected_models,
    _window_sizes,
    resolve_device,
)
from glucose_forecasting.backends.neuralforecast.plotting import (
    write_comparison_dashboard,
    write_prediction_charts,
)
from glucose_forecasting.backends.neuralforecast.telemetry import (
    EpochProgressCallback,
    announce,
    configure_run_logs,
)
from glucose_forecasting.common.data_loading import limit_series


def run_cross_val(
    config: NeuralForecastRunConfig,
    *,
    suites: ModelSuiteConfig,
    suites_yaml: str,
) -> list[Path]:
    """Run rolling cross-validation over the selected YAML model suite."""
    if config.evaluation != "cross-val":
        raise ValueError("run_cross_val requires evaluation='cross-val'")
    from neuralforecast import NeuralForecast

    splits = prepare_splits(
        config.csv,
        profile_name=config.profile,
        unique_id_choice=config.unique_id,
        split_scheme=config.split_scheme,
        drop_interpolated=config.drop_interpolated,
        max_train_series=config.max_train_series,
        max_points_per_series=config.max_points_per_series,
    )
    frame = limit_series(
        pl.concat((splits.train, splits.validation, splits.test)).sort(["unique_id", "ds"]),
        config.max_train_series,
    )
    if frame.is_empty():
        raise ValueError("cross-validation requires non-empty labeled data")
    horizon, input_size, _ = _window_sizes(config)
    models = _resolve_selected_models(config, suites, splits.profile.name)
    device = resolve_device(config.device)
    run_dir = (
        config.out_dir
        / "nf_cross_val"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = configure_run_logs(run_dir)
    announce(
        f"Starting rolling cross-validation with {len(models)} model(s) on {device}.",
        evaluation="cross-val",
        device=device,
        model_count=len(models),
        log_directory=str(logs_dir),
    )
    results: list[pl.DataFrame] = []
    metrics_rows: list[dict[str, float | str]] = []
    predictions_by_model: dict[str, pl.DataFrame] = {}
    for index, definition in enumerate(models, start=1):
        announce(
            f"[{index}/{len(models)}] Training {definition.name.value}.",
            model=definition.name.value,
            model_index=index,
            model_count=len(models),
            max_steps=config.max_steps,
        )
        model = create_model(
            definition,
            horizon=horizon,
            input_size=input_size,
            max_steps=config.max_steps,
            historical_exogenous=splits.profile.historical_exogenous,
            trainer_kwargs={
                "accelerator": device,
                "devices": 1,
                "default_root_dir": str(run_dir / "lightning" / definition.name.value),
                "logger": False,
                "enable_progress_bar": True,
                "callbacks": [EpochProgressCallback(model_name=definition.name.value)],
            },
            learning_rate=config.learning_rate,
            val_check_steps=config.val_check_steps,
            batch_size=config.batch_size,
            valid_batch_size=config.valid_batch_size,
            windows_batch_size=config.windows_batch_size,
            inference_windows_batch_size=config.inference_windows_batch_size,
            step_size=config.step_size,
        )
        nf = NeuralForecast(models=[model], freq=config.freq)
        with start_action(
            action_type="neuralforecast_cross_validation",
            model=definition.name.value,
            max_steps=config.max_steps,
            n_windows=config.n_windows,
        ):
            result = nf.cross_validation(
                df=frame.select(
                    ["unique_id", "ds", "y", *splits.profile.historical_exogenous]
                ).to_pandas(),
                n_windows=config.n_windows,
                step_size=horizon,
            )
        result_frame = pl.from_pandas(result).with_columns(
            pl.lit(definition.name.value).alias("model")
        )
        model_predictions = result_frame.join(
            frame.select(["unique_id", "ds", "study_group"]),
            on=["unique_id", "ds"],
            how="left",
        )
        metrics = calculate_metrics(
            model_predictions.rename({definition.name.value: "yhat"})
        )
        announce(
            (
                f"{definition.name.value} metrics: MAE={metrics.overall.mae:.3f}, "
                f"RMSE={metrics.overall.rmse:.3f}, MARD={metrics.overall.mard:.3f}%."
            ),
            model=definition.name.value,
            mae=metrics.overall.mae,
            rmse=metrics.overall.rmse,
            mard=metrics.overall.mard,
        )
        metrics_rows.append(
            {
                "model": definition.name.value,
                "mae": metrics.overall.mae,
                "rmse": metrics.overall.rmse,
                "mard": metrics.overall.mard,
            }
        )
        result_frame.write_csv(run_dir / f"cross_val_{definition.name.value}.csv")
        nf.save(
            str(run_dir / "neuralforecast" / definition.name.value),
            overwrite=True,
        )
        pl.DataFrame(
            [
                {
                    "mae": metrics.overall.mae,
                    "rmse": metrics.overall.rmse,
                    "mard": metrics.overall.mard,
                }
            ]
        ).write_csv(run_dir / f"cross_val_metrics_{definition.name.value}.csv")
        metrics.by_study_group.write_csv(
            run_dir / f"cross_val_metrics_by_study_group_{definition.name.value}.csv"
        )
        predictions_by_model[definition.name.value] = result_frame
        if config.plot:
            announce(
                f"Writing charts for {definition.name.value}.",
                model=definition.name.value,
            )
            write_prediction_charts(
                frame,
                result_frame,
                model_name=definition.name.value,
                output_dir=run_dir,
                prediction_column=definition.name.value,
                max_sequences=config.max_plot_series,
                title_prefix="Rolling CV — ",
            )
        results.append(result_frame)
    pl.concat(results).write_csv(run_dir / "cross_val_predictions.csv")
    pl.DataFrame(metrics_rows).sort("mae").write_csv(run_dir / "cross_val_metrics_summary.csv")
    if config.plot:
        announce("Writing cross-validation comparison dashboard.")
        write_comparison_dashboard(
            frame,
            predictions_by_model,
            output_dir=run_dir,
            max_sequences=config.max_plot_series,
        )
    (run_dir / "run_config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / "model_suites.yaml").write_text(suites_yaml, encoding="utf-8")
    announce(f"Cross-validation complete; artifacts: {run_dir}", run_directory=str(run_dir))
    return [run_dir]
