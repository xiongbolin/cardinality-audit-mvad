#!/usr/bin/env bash
set -euo pipefail

# Prepare the HuggingFace timm snapshots used by the official Anomalib PatchCore
# sanity check. This is useful when the GPU host cannot reach HuggingFace
directly: run this on a machine with Hub access, then copy `.cache/huggingface`
# to the remote project root.

HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME HF_HUB_CACHE

hf download timm/wide_resnet50_2.racm_in1k --cache-dir "${HF_HUB_CACHE}"
hf download timm/wide_resnet50_2.tv2_in1k --cache-dir "${HF_HUB_CACHE}"
