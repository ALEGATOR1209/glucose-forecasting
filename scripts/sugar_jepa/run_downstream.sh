#!/usr/bin/env bash
# Train the SugarJepa forecaster on the pretrained CGM encoder.
# Encoder weights already extracted to pretrained/xcgm_jepa/cgm_encoder.pt.
# Usage:  bash scripts/sugar_jepa/run_downstream.sh [CSV]
set -euo pipefail

ENCODER="scripts/sugar_jepa/pretrained/xcgm_jepa/cgm_encoder.pt"
CSV="${1:-data/input/loop_ai_ready_joined2.csv}"           # optional arg, defaults to full CSV

uv run python scripts/sugar_jepa/train_sugar_jepa.py \
  --csv "$CSV" \
  --jepa-init "$ENCODER"
