#!/usr/bin/env bash
set -euo pipefail

python scripts/audit_rad_metadata_identity.py \
  --data-root data/RAD \
  --out-summary results/tables/rad_metadata_identity_audit_summary.csv \
  --out-by-condition results/tables/rad_metadata_identity_audit_by_condition.csv \
  --out-md results/tables/rad_metadata_identity_audit.md \
  --online-resource-dir online_resource_1
