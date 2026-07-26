#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/extract_clip_features.py --config configs/clip_vit_b16.yaml
python scripts/run_robovlm_ad.py --features results/features/clip_vit_b16_rad.npz

python scripts/extract_timm_features.py --config configs/dinov2_vit_b14.yaml
python scripts/run_visual_multiview.py --features results/features/dinov2_vit_b14_rad.npz --prefix dinov2_vit_b14_visual

python scripts/make_paper_tables.py
python scripts/make_additional_analysis_tables.py
