# Current-generation official PatchCore baseline

This directory contains the lightweight, Git-compatible metadata portion of the 24 July 2026 official Anomalib PatchCore 2.4.0 rerun on the same 2,985-image RAD no-mask test split used by the current DINOv2/CLIP evidence generation.

Primary reporting rule: use category-macro AUROC because the 18 object categories were fitted with separate PatchCore memory banks. The pooled raw-score AUROC is descriptive only because raw scores are not guaranteed to be calibrated across category-specific models.

Key verified values:

- category-macro image AUROC: 0.827871332521187
- pooled image AUROC (secondary): 0.8294520757621477
- Equal-1 category-macro AUROC: 0.8187222222222221
- Equal-4 category-macro AUROC: 0.8674212962962963
- Equal-4 minus Equal-1: 0.0486990740740741

The 18 `model.ckpt` files are each 128,228,847 bytes and exceed GitHub's ordinary 100 MB per-file limit. Their exact paths, sizes, and SHA-256 hashes are recorded in `checkpoint_asset_register.csv`. The six source ZIP archives and detailed score tables belong in immutable GitHub Release assets rather than ordinary Git history.
