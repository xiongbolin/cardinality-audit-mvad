# Cardinality Audit for Multi-View Industrial Anomaly Detection

This repository contains the code, frozen protocol, derived score tables, verified result summaries, and figure-generation materials for the manuscript:

> **Cardinality Confounding in Multi-View Industrial Anomaly Detection: Null-Model Diagnostics and Fixed-Order Evaluation**
>
> Bailin Xiong, Qiguo Hu, Kai Du, Jinhui Tang, and Renxiang Wang

## What this repository evaluates

Multi-view anomaly systems often aggregate several image-level anomaly scores into one sample-level score. If the number of views differs systematically by label, extreme-value aggregators can discriminate from view count alone. This repository implements three controls:

1. **Cardinality-preserving size-only nulls** that retain labels and observed group sizes but replace image scores with IID noise.
2. **Equal-view evaluation** that samples the same number of views from every group.
3. **Fixed-order subset aggregation** that averages a constant-order kernel across subsets of a larger group.

The repository is an evaluation and audit package, not a claim of detector-performance leadership.

## Repository layout

- `code/cardinality_audit/` - reusable aggregation, control, IO, metric, and Real-IAD view utilities.
- `scripts/run_cardinality_audit.py` - registered end-to-end audit runner.
- `scripts/generate_scheme_a_figures.py` - paper figure generator.
- `scripts/verify_scheme_a_return.py` - independent returned-output verifier.
- `configs/` - frozen Scheme A configuration.
- `protocols/` - aggregator/scorer registries, statistical plan, and experiment matrix.
- `data/rad_image_scores/` - derived RAD per-image score exports; no raw benchmark images.
- `data/verified_results/` - independently verified summary tables used by the manuscript.
- `figure_data/` and `figures/` - plotted data and exported paper figures.
- `tests/` - regression tests for aggregation, controls, metrics, IO, CLI, and view handling.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run tests

```bash
pytest -q
```

## Reproduce the RAD audit

The included RAD derived score exports are sufficient to rerun the registered RAD aggregation audit. The full protocol uses 1,000 Equal-4 repetitions, 1,000 null repetitions, and 2,000 bootstrap repetitions.

```bash
python code/run_cardinality_audit.py run \
  --config configs/cardinality_audit_scheme_a.yaml \
  --rad-score-dir data/rad_image_scores \
  --output-dir outputs/rad_audit
```

Use `python code/run_cardinality_audit.py --help` for all options. Runtime depends on repetition counts and available hardware.

## Complete release archive

The full clean release is also stored under `release_parts/` as numbered Base64 chunks to avoid browser and connector upload limits. Reconstruct it with:

```bash
python assemble_release.py
```

This creates `CARDINALITY_AUDIT_FULL_RELEASE.tar.xz`. Verify its SHA-256 against `release-manifest.txt`, then extract it with a standard archive tool.

## Data and licensing boundaries

This repository does **not** redistribute raw RAD or Real-IAD images. Dataset access remains governed by the original providers. Included CSV files are derived model-score and analysis tables released for reproducibility. See `DATA_NOTICE.md`.

Code is released under the MIT License. Derived tables and figures are released under CC BY 4.0 unless a third-party source imposes a narrower condition.

## Main verified observations

- In the RAD size-only null, unnormalized log-sum-exp achieved mean AUROC 0.9625 and top-3 pooling achieved 0.9322 despite IID image scores.
- Fixed-order controls remained near chance (approximately 0.504) under the registered null.
- Equal-4 matching changed 74 of 132 registered scorer-aggregator ranks.
- Real-IAD controlled stress tests remained near chance when synthetic cardinality was label-independent and became predictive when cardinality was synthetically associated with label.

These findings are bounded to the registered datasets, derived score inventories, and diagnostic simulations.

## Contact

Corresponding author: **Qiguo Hu**  
School of Mechatronics and Vehicle Engineering, Chongqing Jiaotong University, Chongqing, China  
Email: swpihqg@cqjtu.edu.cn
