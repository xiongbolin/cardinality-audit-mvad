#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/make_additional_analysis_tables.py
python scripts/make_paper_tables.py
