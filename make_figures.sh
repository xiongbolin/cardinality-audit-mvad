#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/make_paper_figures.py --only all
