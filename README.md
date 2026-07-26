# Cardinality Audit for Multi-View Industrial Anomaly Detection

This repository contains the code, frozen protocol, derived score tables, verified result summaries, and figure-generation materials for the manuscript:

> **Cardinality Confounding in Multi-View Industrial Anomaly Detection: Null-Model Diagnostics and Fixed-Order Evaluation**
>
> Bailin Xiong, Qiguo Hu, Kai Du, Jinhui Tang, and Renxiang Wang

## What this repository evaluates

Multi-view anomaly systems often aggregate several image-level anomaly scores into one sample-level score. If the number of views differs systematically by label, extreme-value aggregators can discriminate from view count alone. This repository implements:

1. Cardinality-preserving size-only null controls.
2. Equal-view evaluation controls.
3. Fixed-order subset aggregation controls.

This repository is an evaluation and audit package. It does not claim detector-performance leadership.

## PatchCore compatibility boundary

This release documents PatchCore-related evaluation compatibility only.

- PatchCore reference implementation: *PatchCore: Towards Total Recall in Industrial Anomaly Detection*.
- Upstream license: Apache License 2.0.
- Attribution details: see `PATCHCORE_ATTRIBUTION.md` and `NOTICE`.

No PatchCore source code, pretrained weights, or benchmark datasets are redistributed in this repository.

## Repository layout

- `code/cardinality_audit/` - reusable aggregation, control, IO, metric, and view utilities.
- `scripts/` - audit and figure generation tools.
- `configs/` - frozen evaluation configurations.
- `protocols/` - experiment protocol definitions.
- `data/` - derived score exports only; no raw benchmark images.
- `tests/` - regression tests.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run tests

```bash
pytest -q
```

## Data and licensing boundaries

This repository does not redistribute raw RAD or Real-IAD images. Dataset access remains governed by original providers. Included files are derived model-score exports, audit summaries, and figure-generation tables.

Source code in this release branch is provided under the Apache License 2.0. Third-party materials retain their original licenses.

## Complete release archive

The full release archive can be reconstructed and verified using the release manifest and checksum files included with the release package.

## Contact

Corresponding author: **Qiguo Hu**  
School of Mechatronics and Vehicle Engineering, Chongqing Jiaotong University, Chongqing, China
