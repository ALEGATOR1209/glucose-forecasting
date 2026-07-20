"""Aggregate visual reports for comparable NeuralForecast holdout runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl


_METRICS = ("mae", "rmse", "mard")
_COLORS = ("#4C78A8", "#E45756", "#59A14F", "#F28E2B", "#B279A2", "#76B7B2")


def write_holdout_report_visuals(
    output_dir: Path,
    *,
    run_dirs: list[Path],
    models: list[str],
) -> None:
    """Write aggregate metric, cohort, and training-history report artifacts."""
    val_metrics = pl.read_csv(output_dir / "val_metrics_summary.csv")
    test_metrics = pl.read_csv(output_dir / "test_metrics_summary.csv")
    write_metrics_figure(output_dir, val_metrics=val_metrics, test_metrics=test_metrics)
    study_group_metrics = collect_study_group_metrics(run_dirs, models)
    study_group_metrics.write_csv(output_dir / "study_group_metrics.csv")
    write_study_group_figure(output_dir, study_group_metrics)
    training_history = collect_training_history(run_dirs, models)
    training_history.write_csv(output_dir / "training_history.csv")
    write_training_figure(output_dir, training_history)


def collect_study_group_metrics(run_dirs: list[Path], models: list[str]) -> pl.DataFrame:
    """Combine every model's cohort-level scores in long form."""
    parts = [
        pl.read_csv(run_dir / f"{split_name}_metrics_by_study_group.csv").with_columns(
            pl.lit(model).alias("model"),
            pl.lit(split_name).alias("split"),
        )
        for model, run_dir in zip(models, run_dirs, strict=True)
        for split_name in ("val", "test")
    ]
    return pl.concat(parts, how="vertical").select(
        ["split", "model", "study_group", "n_points", *_METRICS]
    )


def collect_training_history(run_dirs: list[Path], models: list[str]) -> pl.DataFrame:
    """Parse per-epoch losses from structured logs without filling missing validation."""
    rows: list[dict[str, Any]] = []
    for model, run_dir in zip(models, run_dirs, strict=True):
        events = _read_model_events(run_dir / "logs" / "training.json", model)
        validation_by_epoch: dict[int, float] = {}
        last_fallback_validation: float | None = None
        for event in events:
            epoch = event.get("epoch")
            if not isinstance(epoch, int):
                continue
            if event.get("message_type") == "validation_epoch_completed":
                validation = _validation_loss(event)
                if validation is not None:
                    validation_by_epoch[epoch] = validation
                continue
            if event.get("message_type") != "train_epoch_completed":
                continue
            fallback_validation = _validation_loss(event)
            if (
                fallback_validation is not None
                and fallback_validation != last_fallback_validation
            ):
                validation_by_epoch.setdefault(epoch, fallback_validation)
                last_fallback_validation = fallback_validation
        for event in events:
            if event.get("message_type") != "train_epoch_completed":
                continue
            epoch = event.get("epoch")
            if not isinstance(epoch, int):
                continue
            train_loss = _train_loss(event)
            rows.append(
                {
                    "model": model,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": validation_by_epoch.get(epoch),
                    "source_run_dir": str(run_dir),
                }
            )
    schema = {
        "model": pl.String,
        "epoch": pl.Int64,
        "train_loss": pl.Float64,
        "valid_loss": pl.Float64,
        "source_run_dir": pl.String,
    }
    return pl.DataFrame(rows, schema=schema).sort(["model", "epoch"])


def write_metrics_figure(
    output_dir: Path,
    *,
    val_metrics: pl.DataFrame,
    test_metrics: pl.DataFrame,
    config: dict[str, Any] | None = None,
) -> Path:
    """Visualize all-model validation/test metrics alongside the run configuration.

    When *config* is ``None`` the function reads ``run_config.json`` from
    *output_dir* for backward compatibility.  Pass an explicit dict (even
    empty) to skip the file read — useful for cross-backend comparisons
    where no single NeuralForecast config applies.
    """
    ordered_models = test_metrics.sort("mae")["model"].to_list()
    if config is None:
        config_path = output_dir / "run_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
    has_config = bool(config)
    n_rows = 4 if has_config else 3
    specs: list[list[dict[str, str]]] = [[{"type": "bar"}]] * 3
    titles = ["MAE", "RMSE", "MARD (%)"]
    row_heights = [0.3, 0.3, 0.3]
    if has_config:
        specs.append([{"type": "table"}])
        titles.append("Run configuration")
        row_heights = [0.2, 0.2, 0.2, 0.4]
    figure = make_subplots(
        rows=n_rows,
        cols=1,
        specs=specs,
        subplot_titles=titles,
        vertical_spacing=0.08,
        row_heights=row_heights,
    )
    for row, metric in enumerate(_METRICS, start=1):
        for split_name, metrics, color in (
            ("Validation", val_metrics, "#4C78A8"),
            ("Test", test_metrics, "#E45756"),
        ):
            values = (
                metrics.join(pl.DataFrame({"model": ordered_models}), on="model", how="right")
                .sort("model")
                .select(metric)
                .to_series()
                .to_list()
            )
            figure.add_trace(
                go.Bar(name=split_name, x=ordered_models, y=values, marker_color=color),
                row=row,
                col=1,
            )
    if has_config:
        fields = [
            "csv",
            "profile",
            "holdout_protocol",
            "split_scheme",
            "global_model",
            "h_minutes",
            "freq",
            "input_hours",
            "step_size",
            "max_steps",
            "batch_size",
            "learning_rate",
            "selected_models",
        ]
        labels = [field for field in fields if field in config]
        values = [json.dumps(config[field]) if isinstance(config[field], (list, dict)) else str(config[field]) for field in labels]
        figure.add_trace(
            go.Table(
                header={"values": ["Setting", "Value"]},
                cells={"values": [labels, values]},
            ),
            row=4,
            col=1,
        )
    figure.update_layout(
        title="Aggregate holdout metrics — all evaluated models",
        barmode="group",
        height=1400 if has_config else 1000,
        width=1500,
        template="plotly_white",
    )
    return _write_figure(figure, output_dir / "plots" / "metrics")


def write_study_group_figure(output_dir: Path, metrics: pl.DataFrame) -> Path:
    """Write cohort-level MAE heatmaps with scored-point counts in hover text."""
    splits = ("val", "test")
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Validation MAE by study group", "Test MAE by study group"],
    )
    for column, split_name in enumerate(splits, start=1):
        subset = metrics.filter(pl.col("split") == split_name)
        models = subset.group_by("model").agg(pl.col("mae").mean()).sort("mae")["model"].to_list()
        groups = sorted(subset["study_group"].unique().to_list())
        matrix = [
            [
                _metric_value(subset, model=model, group=group, column="mae")
                for group in groups
            ]
            for model in models
        ]
        point_counts = [
            [
                _metric_value(subset, model=model, group=group, column="n_points")
                for group in groups
            ]
            for model in models
        ]
        figure.add_trace(
            go.Heatmap(
                x=groups,
                y=models,
                z=matrix,
                customdata=point_counts,
                colorbar={"title": "MAE"} if column == 2 else None,
                hovertemplate="model=%{y}<br>study group=%{x}<br>MAE=%{z:.3f}<br>n=%{customdata}<extra></extra>",
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        title="Cohort-level holdout MAE — all scored points are retained in hover details",
        height=700,
        width=1500,
        template="plotly_white",
    )
    return _write_figure(figure, output_dir / "plots" / "study_group_metrics")


def write_training_figure(output_dir: Path, history: pl.DataFrame) -> Path:
    """Write training and observed validation losses without interpolating gaps."""
    figure = go.Figure()
    for index, model in enumerate(history["model"].unique(maintain_order=True).to_list()):
        subset = history.filter(pl.col("model") == model).sort("epoch")
        color = _COLORS[index % len(_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=subset["epoch"].to_list(),
                y=subset["train_loss"].to_list(),
                mode="lines",
                name=f"{model} train",
                line={"color": color},
            )
        )
        validation = subset.drop_nulls("valid_loss")
        figure.add_trace(
            go.Scatter(
                x=validation["epoch"].to_list(),
                y=validation["valid_loss"].to_list(),
                mode="lines+markers",
                name=f"{model} validation",
                line={"color": color, "dash": "dash"},
                marker={"size": 7},
                connectgaps=False,
            )
        )
    figure.update_layout(
        title="Training history — validation markers appear only at observed checks",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=750,
        width=1500,
        hovermode="x unified",
        template="plotly_white",
    )
    return _write_figure(figure, output_dir / "plots" / "training")


def _read_model_events(path: Path, model: str) -> list[dict[str, Any]]:
    """Return events emitted for one model from potentially shared Eliot logs."""
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("model") == model:
            events.append(event)
    return events


def _train_loss(event: dict[str, Any]) -> float | None:
    """Return the epoch-level train loss when available."""
    value = event.get("train_loss_epoch", event.get("train_loss"))
    return float(value) if isinstance(value, int | float) else None


def _validation_loss(event: dict[str, Any]) -> float | None:
    """Return the validation loss under either Lightning metric spelling."""
    value = event.get("valid_loss", event.get("val_loss"))
    return float(value) if isinstance(value, int | float) else None


def _metric_value(
    frame: pl.DataFrame,
    *,
    model: str,
    group: str,
    column: str,
) -> float | int | None:
    values = frame.filter(
        (pl.col("model") == model) & (pl.col("study_group") == group)
    ).select(column)
    if values.is_empty():
        return None
    return values.item(0, 0)


def _write_figure(figure: go.Figure, path_without_suffix: Path) -> Path:
    """Persist HTML plus best-effort PNG without making reports depend on Chrome."""
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    html_path = path_without_suffix.with_suffix(".html")
    figure.write_html(html_path, include_plotlyjs="cdn")
    try:
        figure.write_image(path_without_suffix.with_suffix(".png"), width=1500, height=900)
    except (OSError, RuntimeError, ValueError):
        pass
    return html_path
