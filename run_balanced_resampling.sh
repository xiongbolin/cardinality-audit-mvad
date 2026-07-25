#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/run_protocol_a_balanced_resampling.py
python scripts/make_paper_tables.py
