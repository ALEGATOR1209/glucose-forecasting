"""Terminal progress and persistent structured logs for NeuralForecast runs."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eliot import Message
from pycomfort.logging import to_nice_file
from pytorch_lightning.callbacks import Callback

from glucose_forecasting.common.console import safe_echo


def configure_run_logs(run_dir: Path) -> Path:
    """Write Eliot JSON and rendered logs alongside one training run."""
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    to_nice_file(
        output_file=logs_dir / "training.json",
        rendered_file=logs_dir / "training.log",
    )
    return logs_dir


def announce(message: str, **fields: Any) -> None:
    """Print a concise status line and record the matching Eliot event."""
    safe_echo(message)
    Message.log(
        message_type="status",
        message=message,
        **fields,
    )


class EpochProgressCallback(Callback):
    """Report and persist train/validation losses after every epoch."""

    def __init__(self, *, model_name: str) -> None:
        self.model_name = model_name

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = _scalar_metrics(trainer.callback_metrics)
        epoch = trainer.current_epoch + 1
        fields = {"model": self.model_name, "epoch": epoch, **metrics}
        Message.log(
            message_type="train_epoch_completed",
            **fields,
        )
        safe_echo(
            _epoch_message(
                self.model_name,
                epoch,
                trainer.global_step,
                trainer.max_steps,
                metrics,
            )
        )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Record validation loss only when Lightning actually runs validation."""
        if trainer.sanity_checking:
            return
        metrics = _scalar_metrics(trainer.callback_metrics)
        validation = {
            name: value
            for name, value in metrics.items()
            if name in {"valid_loss", "val_loss", "ptl/val_loss"}
        }
        if validation:
            Message.log(
                message_type="validation_epoch_completed",
                model=self.model_name,
                epoch=trainer.current_epoch + 1,
                **validation,
            )


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(value.detach().cpu().item())
        for name, value in metrics.items()
        if hasattr(value, "detach") and getattr(value, "numel", lambda: 0)() == 1
    }


def _epoch_message(
    model_name: str,
    epoch: int,
    step: int,
    max_steps: int,
    metrics: Mapping[str, float],
) -> str:
    values = " ".join(
        f"{name}={value:.5f}"
        for name, value in metrics.items()
        if name in {"train_loss", "valid_loss", "val_loss"}
    )
    return (
        f"  Epoch {epoch} | step {step}/{max_steps} | {model_name} | "
        f"{values or 'loss unavailable'}"
    )
