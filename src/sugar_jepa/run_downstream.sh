#!/usr/bin/env bash
# Train the SugarJepa forecaster on the pretrained CGM encoder.
# Encoder weights already extracted to pretrained/xcgm_jepa/cgm_encoder.pt.
# Usage:  bash src/sugar_jepa/run_downstream.sh [CSV]
#
# NOTE: --jepa-init / SugarJepaModel2 is not yet wired into train_sugar_jepa.py
# (see "Second architecture track" in README.md). This script is a placeholder
# until that integration lands.
set -euo pipefail

ENCODER="src/sugar_jepa/pretrained/xcgm_jepa/cgm_encoder.pt"
CSV="${1:-data/input/loop_and_ai_ready/loop_ai_ready_joined2.csv}"           # optional arg, defaults to full CSV

uv run python src/sugar_jepa/train_sugar_jepa.py \
  --csv "$CSV" \
  --jepa-init "$ENCODER"
