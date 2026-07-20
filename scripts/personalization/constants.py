"""Shared constants for Milestone 8 personalization experiments."""
from __future__ import annotations

from typing import Final

# Default global SugarOne checkpoint for personalization experiments.
DEFAULT_BASE_RUN_DIR: Final[str] = "test_model_sugar_one"

# Quality Loop users present in loop.csv but excluded from loop_ai_ready_joined2.csv
# (passed basal/bolus/carb completeness; not selected by row-balance builder).
LOOP_HOLDOUT_QUALITY_USERS: Final[tuple[str, ...]] = (
    "154",
    "556",
    "730",
    "1017",
    "1029",
    "1082",
)

# Default chronological split fractions (of the person's full timeline).
DEFAULT_TEST_FRACTION: Final[float] = 0.25
DEFAULT_VAL_FRACTION_OF_REMAINDER: Final[float] = 0.15

# Step 3: data-size sweep grid (personal train days). "all" = full train split.
DEFAULT_DATA_SIZE_DAYS: Final[tuple[int | str, ...]] = (1, 3, 7, 14, 30, 60, "all")

# Step 2: LwF starting point from GluMind continual tuning (reports/glumind/).
# AI_READY_PLUS_TYPE_1 best continual: lwf_lambda=0.3 (glumind_continual_h12_20260226_011733).
# AI_READY best continual: lwf_lambda=0.2 — secondary reference for non-T1DM cohorts.
GLUMIND_BEST_LWF_TYPE1: Final[float] = 0.3
GLUMIND_BEST_LWF_AI_READY: Final[float] = 0.2

# LwF grid centered on GluMind type-1 best (0.3): steps below and above starting point.
DEFAULT_LWF_LAMBDAS: Final[tuple[float, ...]] = (0.2, 0.25, 0.3, 0.35)

# Step 2: LR multipliers relative to base model ``tuning_meta.json`` lr (test_model_sugar_one: 4e-4).
DEFAULT_LR_MULTIPLIERS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)

# Step 2: weight_decay tuning (base = 3e-5 from test_model_sugar_one / SugarOne training).
DEFAULT_WEIGHT_DECAY: Final[float] = 3e-5
DEFAULT_WEIGHT_DECAY_MULTIPLIERS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)

# Personalization fine-tune defaults (Step 2+).
DEFAULT_FT_PATIENCE: Final[int] = 3
DEFAULT_VAL_EVERY_N_EPOCHS: Final[int] = 2
DEFAULT_PROGRESS_LOG_INTERVAL_S: Final[float] = 10.0

# Sliding-window stride for train windows (5-min CGM steps).
DENSE_WINDOW_STRIDE: Final[int] = 1
SPARSE_WINDOW_STRIDE: Final[int] = 6  # 6×5 min = 30 min between window starts
DEFAULT_TRAIN_WINDOW_STRIDE: Final[int] = SPARSE_WINDOW_STRIDE

DEFAULT_SEED: Final[int] = 43
DEFAULT_HORIZON: Final[int] = 12
DEFAULT_STUDY_GROUP: Final[str] = "T1DM"

# Loop / SugarOne CSV column names.
COL_SEQ: Final[str] = "sequence_id"
COL_USER: Final[str] = "User ID"
COL_TS: Final[str] = "Timestamp"
COL_SPLIT: Final[str] = "Recommended Split"
COL_GROUP: Final[str] = "Study Group"
COL_EVENT: Final[str] = "Event Type"
COL_GLU: Final[str] = "Glucose (mg/dL)"
COL_BASAL: Final[str] = "Basal Rate (U/h)"
COL_BOLUS: Final[str] = "Bolus Insulin (U)"
COL_CARB: Final[str] = "Carbohydrates (g)"

TS_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S"

SUGAR_ONE_VALUE_COLUMNS: Final[dict[str, str]] = {
    "glucose": COL_GLU,
    "basal": COL_BASAL,
    "bolus": COL_BOLUS,
    "carbs": COL_CARB,
}
