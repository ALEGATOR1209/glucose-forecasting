"""Top-level command line interface for glucose forecasting."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from importlib.metadata import version
import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from glucose_forecasting.backends.neuralforecast.config import (
    NeuralForecastRunConfig,
    load_model_suites,
)
from glucose_forecasting.backends.neuralforecast.evaluations.cross_val import run_cross_val
from glucose_forecasting.backends.neuralforecast.evaluations.holdout import (
    run_holdout,
    run_loaded_holdout,
    summarize_holdout_runs,
)
from glucose_forecasting.config import (
    DatasetSpec,
    ModelSelection,
    load_evaluation_config,
)
from glucose_forecasting.evaluation import evaluate_and_compare, evaluate_run_dir, run_evaluation
from glucose_forecasting.models.registry import (
    ModelArtifact,
    ModelRegistry,
    ModelResolutionError,
    load_registry,
    resolve_data_path,
)
from glucose_forecasting.release import (
    download_inference_bundle,
    publish_inference_bundle,
    validate_inference_bundle,
)

app = typer.Typer(
    add_completion=False,
    help="Train, evaluate, and publish glucose forecasting models.",
    no_args_is_help=True,
)
models_app = typer.Typer(help="Inspect and resolve registered model artifacts.")
config_app = typer.Typer(help="Validate modern workflow configuration files.")
data_app = typer.Typer(help="Resolve project data paths.")
release_app = typer.Typer(help="Validate, publish, and retrieve inference releases.")
neuralforecast_app = typer.Typer(help="Train and evaluate saved NeuralForecast bundles.")

_DEFAULT_REGISTRY = Path("data/catalog/model-registry.json")
_PROFILE_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "loop": ("loop-v1", ("basal", "bolus", "carbohydrates")),
    "ai-readi": ("ai-readi-v1", ("heart_rate", "steps")),
}

app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(release_app, name="release")
app.add_typer(neuralforecast_app, name="neuralforecast")


@app.callback()
def main() -> None:
    """Train, evaluate, and publish glucose forecasting models."""


@app.command()
def info() -> None:
    """Print the installed glucose-forecasting version."""
    typer.echo(version("glucose-forecasting"))


@app.command("train")
def train(
    backend: Annotated[str, typer.Option(help="Training backend; currently neuralforecast.")],
    data: Annotated[Path, typer.Option(help="Labeled CSV data file.")],
    evaluation: Annotated[
        str,
        typer.Option(
            "--eval",
            help=(
                "Evaluation protocol: holdout gives comparable fixed-split cohort metrics; "
                "cross-val provides rolling model screening and is not comparable to cohort reports."
            ),
        ),
    ] = "holdout",
    holdout_protocol: Annotated[
        str,
        typer.Option(
            help=(
                "Holdout scoring: sugarone-compatible uses dense stride-1 128/12 windows; "
                "dense supports native experimental geometry; tail is the legacy final-horizon protocol."
            )
        ),
    ] = "sugarone-compatible",
    profile: Annotated[
        str, typer.Option(help="Data profile: auto, ai-readi, or loop.")
    ] = "auto",
    models: Annotated[
        str, typer.Option(help="YAML suite or comma-separated model names; defaults to auto.")
    ] = "auto",
    model_config: Annotated[
        Path | None, typer.Option(help="Optional replacement model-suites YAML.")
    ] = None,
    device: Annotated[
        str, typer.Option(help="auto prefers CUDA, then MPS, then CPU.")
    ] = "auto",
    out_dir: Annotated[Path, typer.Option(help="Directory for run artifacts.")] = Path(
        "data/output/runs"
    ),
    unique_id: Annotated[
        str, typer.Option(help="Series identifier: sequence_id or user_id.")
    ] = "sequence_id",
    split_scheme: Annotated[
        str, typer.Option(help="classic or trainval_test_as_val.")
    ] = "classic",
    global_model: Annotated[
        bool, typer.Option(help="Train one model using all study groups.")
    ] = False,
    study_group: Annotated[
        list[str], typer.Option(help="Study group(s) to train; repeat this option.")
    ] = [],
    max_steps: Annotated[int, typer.Option(help="Maximum optimization steps.")] = 2000,
    h_minutes: Annotated[int, typer.Option(help="Forecast horizon in minutes.")] = 60,
    freq: Annotated[str, typer.Option(help="Sampling frequency, e.g. 5min.")] = "5min",
    input_hours: Annotated[
        float,
        typer.Option(help="Input context in hours; 10.6667 gives SugarOne's 128 steps at 5min."),
    ] = 128 * 5 / 60,
    train_tail_val_hours: Annotated[
        float, typer.Option(help="Internal train-tail validation length in hours.")
    ] = 24.0,
    val_check_steps: Annotated[int, typer.Option(help="Validation check interval.")] = 400,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 8,
    valid_batch_size: Annotated[int, typer.Option(help="Validation batch size.")] = 8,
    windows_batch_size: Annotated[int, typer.Option(help="Training window batch size.")] = 256,
    inference_windows_batch_size: Annotated[
        int, typer.Option(help="Inference window batch size.")
    ] = 256,
    step_size: Annotated[
        int,
        typer.Option(help="Sliding training-window step; 1 matches SugarOne."),
    ] = 1,
    learning_rate: Annotated[float, typer.Option(help="Learning rate.")] = 1e-3,
    max_train_series: Annotated[
        int, typer.Option(help="Cap training series; useful for real-data development runs.")
    ] = 0,
    max_eval_series: Annotated[int, typer.Option(help="Cap evaluation series.")] = 0,
    max_points_per_series: Annotated[
        int, typer.Option(help="Keep the latest N points in each series; zero keeps all.")
    ] = 0,
    n_windows: Annotated[
        int, typer.Option(help="Rolling windows for --eval cross-val.")
    ] = 3,
    drop_interpolated: Annotated[
        bool, typer.Option(help="Remove interpolated rows from every split.")
    ] = False,
    mask_interpolated_targets: Annotated[
        bool, typer.Option(help="Keep interpolated history but exclude target rows from metrics.")
    ] = False,
    save_predictions: Annotated[
        bool, typer.Option(help="Write split prediction CSV files for holdout evaluation.")
    ] = False,
    plot: Annotated[
        bool, typer.Option("--plot/--no-plot", help="Write interactive HTML and PNG prediction charts.")
    ] = True,
    max_plot_series: Annotated[
        int, typer.Option(help="Representative series to chart per model.")
    ] = 3,
    list_models: Annotated[
        bool, typer.Option(help="Show resolved YAML suites and dependency availability, then exit.")
    ] = False,
) -> None:
    """Train NeuralForecast models using held-out or rolling-CV evaluation."""
    if backend != "neuralforecast":
        raise typer.BadParameter("only backend=neuralforecast is currently supported", param_hint="--backend")
    try:
        suites, suite_yaml = load_model_suites(model_config)
        if list_models:
            for name, suite in suites.suites.items():
                typer.echo(
                    f"{name}: profiles={','.join(suite.profiles)} models="
                    f"{','.join(model.value for model in suite.models)}"
                )
            return
        run_config = NeuralForecastRunConfig(
            csv=data,
            profile=profile,
            models=models,
            model_config_path=model_config,
            evaluation=evaluation,
            holdout_protocol=holdout_protocol,
            device=device,
            out_dir=out_dir,
            unique_id=unique_id,
            split_scheme=split_scheme,
            global_model=global_model,
            study_groups=tuple(study_group),
            max_steps=max_steps,
            h_minutes=h_minutes,
            freq=freq,
            input_hours=input_hours,
            train_tail_val_hours=train_tail_val_hours,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            step_size=step_size,
            learning_rate=learning_rate,
            max_train_series=max_train_series,
            max_eval_series=max_eval_series,
            max_points_per_series=max_points_per_series,
            n_windows=n_windows,
            drop_interpolated=drop_interpolated,
            mask_interpolated_targets=mask_interpolated_targets,
            save_predictions=save_predictions,
            plot=plot,
            max_plot_series=max_plot_series,
        )
        run_dirs = (
            run_holdout(run_config, suites=suites, suites_yaml=suite_yaml)
            if run_config.evaluation == "holdout"
            else run_cross_val(run_config, suites=suites, suites_yaml=suite_yaml)
        )
    except (OSError, ValueError, RuntimeError, ValidationError) as error:
        raise typer.BadParameter(str(error)) from error
    for run_dir in run_dirs:
        typer.echo(f"NeuralForecast run written to {run_dir}")


@neuralforecast_app.command("evaluate")
def evaluate_neuralforecast(
    run_dir: Annotated[Path, typer.Option(help="Completed NeuralForecast model run directory.")],
    data: Annotated[Path, typer.Option(help="Labeled CSV to evaluate using the source split settings.")],
    out: Annotated[
        Path | None,
        typer.Option(help="New output directory; defaults to RUN_DIR/evaluations/<UTC timestamp>."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(help="Optional inference device override: auto, cpu, cuda, or mps."),
    ] = None,
    max_eval_series: Annotated[
        int | None,
        typer.Option(help="Optional cap on evaluated series for a diagnostic run."),
    ] = None,
) -> None:
    """Re-evaluate a saved NeuralForecast bundle without fitting."""
    config_path = run_dir / "run_config.json"
    bundle_dir = run_dir / "neuralforecast"
    if not config_path.is_file():
        raise typer.BadParameter(f"run config not found: {config_path}", param_hint="--run-dir")
    if not bundle_dir.is_dir():
        raise typer.BadParameter(f"saved NeuralForecast bundle not found: {bundle_dir}", param_hint="--run-dir")
    if not data.is_file():
        raise typer.BadParameter(f"CSV data file not found: {data}", param_hint="--data")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        source_protocol = payload.get("holdout_protocol")
        if source_protocol is None:
            payload["holdout_protocol"] = "dense"
        payload["csv"] = str(data)
        if device is not None:
            payload["device"] = device
        if max_eval_series is not None:
            payload["max_eval_series"] = max_eval_series
        config = NeuralForecastRunConfig.model_validate(payload)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = out if out is not None else run_dir / "evaluations" / timestamp
        written_dir = run_loaded_holdout(config, bundle_dir=bundle_dir, run_dir=output_dir)
    except (OSError, ValueError, RuntimeError, ValidationError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"NeuralForecast evaluation written to {written_dir}")


@neuralforecast_app.command("summarize-holdout")
def summarize_neuralforecast_holdout(
    run_dir: Annotated[
        list[Path],
        typer.Option("--run-dir", help="Completed per-model holdout run; repeat for each model."),
    ],
    out: Annotated[
        Path | None,
        typer.Option(help="Summary directory; defaults below the shared group directory."),
    ] = None,
    plot: Annotated[
        bool,
        typer.Option("--plot/--no-plot", help="Write val/test comparison dashboards."),
    ] = True,
) -> None:
    """Combine compatible per-model holdout runs without retraining."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = out if out is not None else run_dir[0].parent / "summaries" / timestamp
    try:
        written_dir = summarize_holdout_runs(run_dir, output_dir=output_dir, plot=plot)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"NeuralForecast holdout summary written to {written_dir}")


@app.command("evaluate")
def evaluate(
    run_dir: Annotated[
        list[Path] | None,
        typer.Option("--run-dir", help="Model run directory; repeat for multi-model comparison."),
    ] = None,
    data: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "CSV data file(s) for live inference.  When omitted, precomputed "
                "metrics are read from each run directory."
            ),
        ),
    ] = None,
    train_data: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Training CSV for scaler fitting (custom PyTorch models).  "
                "When omitted, resolved from run directory metadata or --data."
            ),
        ),
    ] = None,
    label: Annotated[
        list[str] | None,
        typer.Option(help="Model label for the corresponding --run-dir; repeat to match."),
    ] = None,
    device: Annotated[str, typer.Option(help="Inference device: auto, cpu, cuda, mps.")] = "auto",
    out: Annotated[
        Path | None,
        typer.Option(help="Output directory for the comparison report."),
    ] = None,
    plot: Annotated[
        bool,
        typer.Option("--plot/--no-plot", help="Write interactive Plotly comparison charts."),
    ] = True,
    models: Annotated[
        list[str] | None,
        typer.Option(help="Model selector(s) for registry-based evaluation (requires --registry)."),
    ] = None,
    registry: Annotated[
        Path | None,
        typer.Option(help="Registry JSON path. Defaults to data/catalog/model-registry.json."),
    ] = None,
) -> None:
    """Evaluate model run directories and produce a comparison report.

    Run-directory mode (primary)::

        glucose evaluate --run-dir DIR1 --run-dir DIR2 --out REPORT_DIR

    Registry-based mode (requires a populated model registry)::

        glucose evaluate --data DATA.csv --models NAME1,NAME2 --registry REG.json
    """
    if run_dir:
        data_path = Path(data[0]) if data and len(data) == 1 else None
        if data and len(data) > 1:
            raise typer.BadParameter(
                "pass at most one --data CSV with --run-dir; each run directory "
                "is evaluated against the same dataset",
                param_hint="--data",
            )
        try:
            if len(run_dir) == 1:
                result = evaluate_run_dir(
                    run_dir[0], data=data_path, train_data=train_data,
                    label=(label[0] if label else None), device=device,
                )
                for split_name, split_metrics in sorted(result.split_results.items()):
                    typer.echo(
                        f"{result.model_name} {split_name}: "
                        f"MAE={split_metrics.overall.mae:.3f}  "
                        f"RMSE={split_metrics.overall.rmse:.3f}  "
                        f"MARD={split_metrics.overall.mard:.3f}%"
                    )
            else:
                report_dir = evaluate_and_compare(
                    run_dir,
                    data=data_path,
                    train_data=train_data,
                    labels=label or [],
                    output_dir=out,
                    device=device,
                    plot=plot,
                )
                typer.echo(f"Comparison report written to {report_dir}")
        except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
            raise typer.BadParameter(str(error)) from error
        return

    if models is not None or registry is not None:
        if data is None:
            raise typer.BadParameter("--data is required for registry-based evaluation", param_hint="--data")
        registry_path = _registry_path(registry)
        model_registry = _load_registry_or_error(registry)
        try:
            evaluation_run = run_evaluation(
                data=data,
                models=models or [],
                registry=model_registry,
                registry_path=registry_path,
                project_root=Path.cwd(),
                output_dir=out,
            )
        except (FileNotFoundError, FileExistsError, ValueError) as error:
            raise typer.BadParameter(str(error)) from error
        typer.echo(f"Evaluation records written to {evaluation_run.output_dir}")
        return

    raise typer.BadParameter(
        "provide --run-dir for run-directory evaluation or --models/--registry for registry mode",
        param_hint="--run-dir",
    )


def _registry_path(registry: Path | None) -> Path:
    """Return the supplied registry or the documented project-local default."""
    return registry if registry is not None else Path.cwd() / _DEFAULT_REGISTRY


def _load_registry_or_error(registry: Path | None) -> ModelRegistry:
    """Load a registry while providing command-line actionable failures."""
    registry_path = _registry_path(registry)
    if not registry_path.is_file():
        raise typer.BadParameter(
            f"registry not found at {registry_path}; pass --registry FILE or add "
            "data/catalog/model-registry.json",
            param_hint="--registry",
        )
    try:
        return load_registry(registry_path)
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as error:
        raise typer.BadParameter(
            f"registry at {registry_path} is invalid: {error}",
            param_hint="--registry",
        ) from error


def _artifact_json(artifact: ModelArtifact) -> str:
    """Render an artifact in a stable, machine-readable form."""
    return json.dumps(artifact.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True)


def _infer_dataset_spec(data_path: Path, profile: str | None) -> DatasetSpec:
    """Build a dataset contract from a CSV header or an explicit known profile."""
    if not data_path.is_file():
        raise typer.BadParameter(f"data file not found at {data_path}", param_hint="--data")

    normalized_profile = profile.lower() if profile is not None else None
    if normalized_profile is not None:
        profile_spec = _PROFILE_SPECS.get(normalized_profile)
        if profile_spec is None:
            known_profiles = ", ".join(sorted(_PROFILE_SPECS))
            raise typer.BadParameter(
                f"unknown profile {profile!r}; choose one of: {known_profiles}",
                param_hint="--profile",
            )
        schema, covariates = profile_spec
    else:
        with data_path.open(encoding="utf-8", newline="") as data_file:
            headers = set(csv.reader(data_file).__next__())
        if {"Basal Rate (U/h)", "Bolus Insulin (U)", "Carbohydrates (g)"} & headers:
            schema, covariates = _PROFILE_SPECS["loop"]
        elif {"Heart Rate", "Step Count"} & headers:
            schema, covariates = _PROFILE_SPECS["ai-readi"]
        else:
            raise typer.BadParameter(
                "could not infer a dataset profile from CSV columns; pass "
                "--profile loop or --profile ai-readi",
                param_hint="--profile",
            )

    return DatasetSpec(
        name=data_path.stem,
        path=data_path,
        data_schema=schema,
        covariates=covariates,
        cadence_minutes=5,
        horizon_steps=12,
    )


@models_app.command("list")
def list_models(
    registry: Annotated[
        Path | None,
        typer.Option(help="Registry JSON path. Defaults to data/catalog/model-registry.json."),
    ] = None,
) -> None:
    """List all registered artifacts."""
    model_registry = _load_registry_or_error(registry)
    artifacts = sorted(model_registry.models, key=lambda artifact: (artifact.name, artifact.version))
    typer.echo(
        json.dumps(
            [artifact.model_dump(mode="json", by_alias=True) for artifact in artifacts],
            indent=2,
            sort_keys=True,
        )
    )


@models_app.command("show")
def show_model(
    name: str,
    registry: Annotated[
        Path | None,
        typer.Option(help="Registry JSON path. Defaults to data/catalog/model-registry.json."),
    ] = None,
) -> None:
    """Show every registered version of NAME."""
    model_registry = _load_registry_or_error(registry)
    artifacts = sorted(
        (artifact for artifact in model_registry.models if artifact.name == name),
        key=lambda artifact: artifact.version,
    )
    if not artifacts:
        raise typer.BadParameter(f"model {name!r} is not registered", param_hint="NAME")
    typer.echo(
        json.dumps(
            [artifact.model_dump(mode="json", by_alias=True) for artifact in artifacts],
            indent=2,
            sort_keys=True,
        )
    )


@models_app.command("resolve")
def resolve_model(
    data: Annotated[Path, typer.Option(help="CSV data file to match against the registry.")],
    model: Annotated[str | None, typer.Option(help="Optional model name.")] = None,
    profile: Annotated[
        str | None,
        typer.Option(help="Dataset profile: loop or ai-readi. Inferred from CSV when omitted."),
    ] = None,
    registry: Annotated[
        Path | None,
        typer.Option(help="Registry JSON path. Defaults to data/catalog/model-registry.json."),
    ] = None,
) -> None:
    """Resolve the best compatible registered artifact for a CSV."""
    model_registry = _load_registry_or_error(registry)
    data_path = resolve_data_path(data, Path.cwd())
    dataset = _infer_dataset_spec(data_path, profile)
    try:
        artifact = model_registry.resolve(dataset, ModelSelection(name=model))
    except ModelResolutionError as error:
        raise typer.BadParameter(str(error), param_hint="--model") from error
    typer.echo(_artifact_json(artifact))


@config_app.command("check")
def check_config(
    config: Annotated[Path, typer.Option(help="Versioned evaluation configuration YAML file.")],
) -> None:
    """Validate an evaluation configuration file."""
    if not config.is_file():
        raise typer.BadParameter(f"config file not found at {config}", param_hint="--config")
    try:
        evaluation_config = load_evaluation_config(config)
    except (OSError, ValidationError, ValueError) as error:
        raise typer.BadParameter(
            f"config at {config} is invalid: {error}",
            param_hint="--config",
        ) from error
    typer.echo(f"Configuration valid: {evaluation_config.dataset.name}")


@data_app.command("path")
def data_path(name: str) -> None:
    """Show NAME resolved relative to the project's data/input directory."""
    typer.echo(resolve_data_path(name, Path.cwd()))


@release_app.command("check")
def check_release(
    bundle: Annotated[Path, typer.Argument(help="Local inference bundle directory.")],
) -> None:
    """Validate a local inference bundle."""
    manifest = validate_inference_bundle(bundle)
    typer.echo(f"Release bundle valid: {manifest.release_id}")


@release_app.command("publish")
def publish_release(
    bundle: Annotated[Path, typer.Argument(help="Local inference bundle directory.")],
    repo: Annotated[str, typer.Option(help="Hugging Face model repository (ORG/NAME).")],
    private: Annotated[bool, typer.Option(help="Create or keep the repository private.")] = False,
) -> None:
    """Publish a local inference bundle to the Hugging Face Hub."""
    manifest = publish_inference_bundle(bundle, repo_id=repo, private=private)
    typer.echo(f"Published inference release {manifest.release_id} to {repo}")


@release_app.command("pull")
def pull_release(
    repo: Annotated[str, typer.Option(help="Hugging Face model repository (ORG/NAME).")],
    out: Annotated[Path, typer.Option(help="Local directory for the downloaded bundle.")],
    revision: Annotated[
        str,
        typer.Option(help="Hub revision to download. Defaults to main."),
    ] = "main",
) -> None:
    """Download and validate an inference bundle from the Hugging Face Hub."""
    manifest = download_inference_bundle(repo, revision=revision, target_dir=out)
    typer.echo(f"Downloaded inference release {manifest.release_id} to {out}")
