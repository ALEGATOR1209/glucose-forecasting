"""Interactive actual-versus-prediction charts for NeuralForecast runs."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl


_MODEL_COLORS = (
    "#E45756",
    "#4C78A8",
    "#59A14F",
    "#F28E2B",
    "#B279A2",
    "#76B7B2",
)


def write_prediction_charts(
    actual: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    model_name: str,
    output_dir: Path,
    prediction_column: str = "yhat",
    max_sequences: int = 3,
    title_prefix: str = "",
    split_name: str | None = None,
) -> list[Path]:
    """Write deterministically selected per-sequence diagnostic charts."""
    _require_columns(actual, ("unique_id", "ds", "y"))
    _require_columns(predictions, ("unique_id", "ds", "y", prediction_column))
    plot_dir = output_dir / "plots" / "diagnostic_examples" / model_name
    if split_name is not None:
        plot_dir /= split_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    sequence_ids = predictions["unique_id"].unique(maintain_order=True).head(max_sequences)
    written: list[Path] = []
    for sequence_id in sequence_ids:
        actual_series = actual.filter(pl.col("unique_id") == sequence_id).sort("ds")
        predicted_series = predictions.filter(pl.col("unique_id") == sequence_id).sort("ds")
        figure = _prediction_figure(
            actual_series,
            predicted_series,
            model_name=model_name,
            prediction_column=prediction_column,
            title=(
                f"{title_prefix}{model_name}: diagnostic example for {sequence_id} "
                "(first encountered sequence ID; not a random or performance-selected result)"
            ),
        )
        stem = f"sequence_{_safe_filename(str(sequence_id))}"
        html_path = plot_dir / f"{stem}.html"
        figure.write_html(html_path, include_plotlyjs="cdn")
        _write_png(figure, plot_dir / f"{stem}.png")
        written.append(html_path)
    return written


def write_comparison_dashboard(
    actual: pl.DataFrame,
    predictions_by_model: Mapping[str, pl.DataFrame],
    *,
    output_dir: Path,
    max_sequences: int = 3,
    dashboard_subdir: str = "dashboard",
) -> Path:
    """Write one deterministically selected diagnostic comparison dashboard."""
    _require_columns(actual, ("unique_id", "ds", "y"))
    dashboard_dir = output_dir / "plots" / dashboard_subdir
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    sequence_ids = actual["unique_id"].unique(maintain_order=True).head(max_sequences).to_list()
    figure = make_subplots(
        rows=len(sequence_ids),
        cols=1,
        shared_xaxes=False,
        subplot_titles=[
            f"Diagnostic example: {sequence_id} (first encountered; not performance-selected)"
            for sequence_id in sequence_ids
        ],
        vertical_spacing=0.08,
    )
    for row, sequence_id in enumerate(sequence_ids, start=1):
        series = actual.filter(pl.col("unique_id") == sequence_id).sort("ds")
        figure.add_trace(
            go.Scatter(
                x=series["ds"].to_list(),
                y=series["y"].to_list(),
                mode="lines",
                name="Actual",
                line={"color": "#202020", "width": 3},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        for index, (model_name, predictions) in enumerate(predictions_by_model.items()):
            if model_name not in predictions.columns:
                continue
            predicted_series = predictions.filter(pl.col("unique_id") == sequence_id).sort("ds")
            figure.add_trace(
                go.Scatter(
                    x=predicted_series["ds"].to_list(),
                    y=predicted_series[model_name].to_list(),
                    mode="lines+markers",
                    name=model_name,
                    line={"color": _MODEL_COLORS[index % len(_MODEL_COLORS)], "dash": "dash"},
                    marker={"size": 4},
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
    figure.update_layout(
        title=(
            "NeuralForecast model comparison — diagnostic sequence examples; "
            "first encountered IDs, not aggregate results"
        ),
        height=max(450, 350 * len(sequence_ids)),
        width=1500,
        hovermode="x unified",
        template="plotly_white",
    )
    path = dashboard_dir / "model_comparison.html"
    figure.write_html(path, include_plotlyjs="cdn")
    _write_png(figure, dashboard_dir / "model_comparison.png")
    return path


def _prediction_figure(
    actual: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    model_name: str,
    prediction_column: str,
    title: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual["ds"].to_list(),
            y=actual["y"].to_list(),
            mode="lines",
            name="Actual glucose",
            line={"color": "#202020", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=predictions["ds"].to_list(),
            y=predictions[prediction_column].to_list(),
            mode="lines+markers",
            name=f"{model_name} forecast",
            line={"color": _MODEL_COLORS[0], "width": 2, "dash": "dash"},
            marker={"size": 6, "symbol": "diamond"},
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        hovermode="x unified",
        template="plotly_white",
        width=1500,
        height=600,
    )
    return figure


def _write_png(figure: go.Figure, path: Path) -> None:
    """Attempt static export without making the training run depend on Chrome."""
    try:
        figure.write_image(path, width=1500, height=600)
    except (OSError, RuntimeError, ValueError):
        return


def _require_columns(frame: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing chart columns: {', '.join(sorted(missing))}")


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)
