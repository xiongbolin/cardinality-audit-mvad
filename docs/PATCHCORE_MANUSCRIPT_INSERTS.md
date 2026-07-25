# Manuscript reporting text for the official PatchCore rerun

## Recommended result statement

A same-generation official Anomalib PatchCore baseline, trained separately on normal images from each of 18 released object categories, achieved category-macro image AUROC 0.8279. Its pooled raw-score AUROC was 0.8295 and is treated as secondary because category-specific models are not score-calibrated across objects. Under the fixed released-condition observation-budget analysis, PatchCore category-macro AUROC increased from 0.8187 at Equal-1 to 0.8674 at Equal-4.

## Required reporting constraint

Because PatchCore uses 18 separately fitted category memory banks, report the category-macro AUROC as the primary cross-method comparator. Do not use pooled raw-score AUROC as the sole headline PatchCore result.
