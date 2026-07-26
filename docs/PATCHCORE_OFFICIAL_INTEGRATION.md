# Official Anomalib PatchCore integration

## Scope

This release adds the completed same-split official Anomalib PatchCore 2.4.0 run used as an established current-generation baseline for the RAD no-mask repeated-observation study.

## Frozen execution evidence

- Implementation: `anomalib.models.Patchcore`
- Anomalib: 2.4.0
- Backbone: wide-ResNet50-2
- Feature layers: layer2 and layer3
- Coreset ratio: 0.1
- Nearest neighbors: 9
- Resize / center crop: 256 x 256 / 224 x 224
- Device: NVIDIA GeForce RTX 5070
- PyTorch: 2.11.0+cu128
- Categories completed: 18/18
- Normal training images: 1,062
- Test images: 2,985, including 162 normal and 2,823 anomalous
- Run duration: 431.42 seconds

## Primary results

PatchCore uses a separately fitted memory bank for each released object category. Raw anomaly-score scales are therefore not guaranteed to be calibrated across category models. The primary cross-category result is the mean of the 18 within-category AUROCs.

| Result | AUROC |
|---|---:|
| Protocol A category-macro | 0.827871 |
| Protocol A pooled raw score, secondary | 0.829452 |
| Equal-1 category-macro | 0.818722 |
| Equal-2 category-macro | 0.857356 |
| Equal-3 category-macro | 0.875019 |
| Equal-4 category-macro | 0.867421 |
| Equal-4 minus Equal-1 | +0.048699 |
| Full released-condition pooled, descriptive | 0.940972 |

The category-macro metrics were independently recomputed from the frozen per-image scores with 1,000 fixed-budget resampling seeds.

## Repository boundary

The repository stores lightweight summary metrics, metadata, audits and exact hashes. The eighteen approximately 128 MB checkpoint files and detailed per-image or replicate tables are not committed to ordinary Git history. Their hashes are registered in the release metadata; the six original ZIP parts should be attached to an immutable GitHub Release.
