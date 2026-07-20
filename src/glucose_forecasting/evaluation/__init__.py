"""Repeatable multi-model evaluation planning and reporting."""

from glucose_forecasting.evaluation.runner import (
    EvaluationRun,
    evaluate_and_compare,
    evaluate_run_dir,
    run_evaluation,
)
from glucose_forecasting.evaluation.types import (
    RunDirKind,
    SingleModelResult,
    SplitMetrics,
)

__all__ = [
    "EvaluationRun",
    "RunDirKind",
    "SingleModelResult",
    "SplitMetrics",
    "evaluate_and_compare",
    "evaluate_run_dir",
    "run_evaluation",
]


def __getattr__(name: str):
    if name == "evaluate_pytorch_run_dir":
        from glucose_forecasting.evaluation.pytorch_adapter import evaluate_pytorch_run_dir
        return evaluate_pytorch_run_dir
    if name == "evaluate_nf_run_dir":
        from glucose_forecasting.evaluation.nf_adapter import evaluate_nf_run_dir
        return evaluate_nf_run_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
