#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/run_robustness_severity_curve.py --device auto --resume
python scripts/make_paper_tables.py
python scripts/make_paper_figures.py --only figure2
