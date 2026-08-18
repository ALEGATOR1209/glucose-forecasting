#!/usr/bin/env python3
"""Shared CSV / canonical column names used across experiments."""
from __future__ import annotations

# Structural columns (AI-READI / Loop style)
COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_TS = "Timestamp (YYYY-MM-DDThh:mm:ss)"
COL_TS_SHORT = "Timestamp"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_EVENT = "Event Type"

# Feature CSV headers
COL_GLU_VALUE = "Glucose Value (mg/dL)"
COL_GLU = "Glucose (mg/dL)"
COL_HR = "Heart Rate"
COL_STEPS = "Step Count"
COL_BASAL = "Basal Rate (U/h)"
COL_BOLUS = "Bolus Insulin (U)"
COL_CARB = "Carbohydrates (g)"
COL_USER = "User ID"

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Canonical internal feature names
FEAT_GLUCOSE = "glucose"
FEAT_HR = "hr"
FEAT_STEPS = "steps"
FEAT_BASAL = "basal"
FEAT_BOLUS = "bolus"
FEAT_CARBS = "carbs"
FEAT_GLUCOSE_JEPA = "glucose_jepa"

GLUMIND_CHANNELS: tuple[str, ...] = (FEAT_GLUCOSE, FEAT_HR, FEAT_STEPS)
SUGAR_ONE_CHANNELS: tuple[str, ...] = (FEAT_GLUCOSE, FEAT_BASAL, FEAT_BOLUS, FEAT_CARBS)
GLUMIND_UNI_CHANNELS: tuple[str, ...] = (FEAT_GLUCOSE,)
TARGET_CHANNEL = FEAT_GLUCOSE
