# Reproducibility status

## Verified release layer

- Apache-2.0 license and PatchCore attribution files are present.
- Official Anomalib PatchCore 2.4.0 summary metrics and per-category recomputation tables are frozen.
- Equal-1 through Equal-4 category-macro summaries are frozen.
- The 18 checkpoint paths, sizes, and SHA-256 hashes are registered.
- The six external ZIP-part sizes and SHA-256 hashes are registered.
- A standard-library metadata verifier and a pytest wrapper are included.

## Distribution boundary

Raw RAD/Real-IAD datasets, third-party pretrained weights, and PatchCore checkpoint binaries are not committed to ordinary Git history. Large immutable assets must be attached separately to a release and retain their upstream licensing boundaries.

## Remaining release condition

The branch remains a draft release candidate until the remote CI gate passes and an immutable release tag is created.
