from __future__ import annotations

import torch

from glucose_forecasting.release import (
    EvaluationProtocol,
    ImputationSpec,
    InferenceConfig,
    MetricsSpec,
    PreprocessorSpec,
    ProvenanceSpec,
    ReleaseManifest,
    ScalerSpec,
    SelectionMetric,
    WindowSpec,
)


class TinyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


def release_manifest() -> ReleaseManifest:
    return ReleaseManifest(
        release_id="sugar-one-2026-07",
        config=InferenceConfig(
            model_id="sugar-one",
            model_type="sugar_one",
            architecture={"d_model": 128, "n_heads": 8},
            feature_order=("glucose", "basal_rate", "bolus", "carbohydrates"),
            horizon=12,
            cadence=5,
        ),
        preprocessor=PreprocessorSpec(
            scalers={
                "glucose": ScalerSpec(
                    kind="standard",
                    parameters={"mean": 120.0},
                )
            },
            aliases={"Glucose (mg/dL)": "glucose"},
            imputation={"bolus": ImputationSpec(method="zero")},
            window=WindowSpec(input_steps=72),
            units={"glucose": "mg/dL"},
        ),
        metrics=MetricsSpec(
            selection_metric=SelectionMetric(name="mae", direction="minimize"),
            validation={"mae": 18.2, "rmse": 24.1},
            test={"mae": 19.1, "rmse": 25.3},
            protocol=EvaluationProtocol(name="held-out evaluation", split="test"),
        ),
        provenance=ProvenanceSpec(
            git_sha="abc1234",
            lock_hash="def5678",
            env={"python": "3.12"},
            dataset_fingerprint="sha256:dataset",
            seed=42,
        ),
    )
