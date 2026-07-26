#!/usr/bin/env bash
set -euo pipefail

python scripts/run_protocol_c_sequential_replay.py \
  --dinov2-features results/features/dinov2_vit_b14_rad.npz \
  --clip-features results/features/clip_vit_b16_rad.npz \
  --clip-config configs/clip_vit_b16.yaml \
  --prompt-config configs/prompt_sets.yaml \
  --ks 1 2 3 5 \
  --theta-low-quantile 0.50 \
  --theta-high-quantile 0.95 \
  --summary-out results/tables/protocol_c_sequential_replay_summary.csv \
  --episodes-out results/tables/protocol_c_sequential_replay_episodes.csv \
  --thresholds-out results/tables/protocol_c_sequential_replay_thresholds.csv
