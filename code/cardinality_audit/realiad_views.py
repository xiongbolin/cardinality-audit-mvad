from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

_EPS = 1e-12


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), _EPS)


def _subspace_fit(values: np.ndarray, variance_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    x = _l2_normalize(values)
    center = x.mean(axis=0)
    centered = x - center
    pca = PCA(n_components=float(variance_ratio), svd_solver="full")
    pca.fit(centered)
    basis = pca.components_.T
    return center, basis


def _subspace_scores(values: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = _l2_normalize(values) - center
    projected = (centered @ basis) @ basis.T
    return np.linalg.norm(centered - projected, axis=1)


def _knn_scores(values: np.ndarray, fit: np.ndarray, k: int) -> np.ndarray:
    query = _l2_normalize(values)
    reference = _l2_normalize(fit)
    distances = pairwise_distances(query, reference, metric="cosine")
    k_eff = max(1, min(int(k), len(reference)))
    return np.partition(distances, kth=k_eff - 1, axis=1)[:, :k_eff].mean(axis=1)


def _upper_p_values(test_scores: np.ndarray, calibration_scores: np.ndarray) -> np.ndarray:
    test = np.asarray(test_scores, dtype=float).reshape(-1)
    cal = np.asarray(calibration_scores, dtype=float).reshape(-1)
    if len(cal) == 0:
        raise ValueError("calibration score vector is empty")
    return (1.0 + (cal[None, :] >= test[:, None]).sum(axis=1)) / (len(cal) + 1.0)


def _uniform_cvm(p_values: np.ndarray) -> float:
    values = np.sort(np.asarray(p_values, dtype=float))
    expected = (np.arange(len(values), dtype=float) + 0.5) / len(values)
    return float(np.mean((values - expected) ** 2))


def _row_index(feature_rows: pd.DataFrame) -> dict[str, int]:
    if "record_id" not in feature_rows.columns:
        raise ValueError("feature row table lacks record_id")
    if feature_rows["record_id"].duplicated().any():
        raise ValueError("feature row table contains duplicate record_id")
    return {str(value): index for index, value in enumerate(feature_rows["record_id"].tolist())}


def _canonical_view_id(source: pd.Series) -> str:
    """Return canonical Real-IAD camera position C1-C5 when encoded in metadata."""
    candidates = []
    for column in ("view_name", "image_path", "relative_path", "record_id"):
        if column in source.index and pd.notna(source[column]):
            candidates.append(str(source[column]))
    for value in candidates:
        match = re.search(r"(?:^|[_\\/.-])(C[1-5])(?:[_\\/.-]|$)", value, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return candidates[0] if candidates else "unknown"


def score_realiad_views(
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    feature_rows: pd.DataFrame,
    features: np.ndarray,
    *,
    seed: int,
    variance_ratio: float = 0.95,
    gamma: float = 0.5,
    knn_candidates: Sequence[int] = (3, 5, 7),
) -> tuple[pd.DataFrame, dict[str, object]]:
    if "label" in test_manifest.columns or "anomaly_class" in test_manifest.columns:
        raise ValueError("test manifest must be label-free during per-view scoring")
    required_train = {"record_id", "category", "sample_id", "partition"}
    required_test = {"record_id", "category", "sample_id"}
    if required_train - set(train_manifest.columns):
        raise ValueError("training split manifest is missing required columns")
    if required_test - set(test_manifest.columns):
        raise ValueError("test manifest is missing required columns")
    index = _row_index(feature_rows)
    train_indices = np.asarray([index.get(str(value), -1) for value in train_manifest["record_id"]], dtype=int)
    test_indices = np.asarray([index.get(str(value), -1) for value in test_manifest["record_id"]], dtype=int)
    if (train_indices < 0).any() or (test_indices < 0).any():
        raise ValueError("manifest record identities are not covered by the feature row table")
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or max(train_indices.max(initial=-1), test_indices.max(initial=-1)) >= len(x):
        raise ValueError("feature matrix does not cover manifest rows")

    categories = sorted(train_manifest["category"].astype(str).unique())
    if set(test_manifest["category"].astype(str).unique()) - set(categories):
        raise ValueError("test categories are missing from normal training data")

    models: dict[str, dict[str, object]] = {}
    validation_p: dict[int, list[np.ndarray]] = {int(k): [] for k in knn_candidates}
    for category in categories:
        local = train_manifest["category"].astype(str).to_numpy() == category
        partitions = train_manifest["partition"].astype(str).to_numpy()
        fit_pos = np.flatnonzero(local & (partitions == "fit"))
        cal_pos = np.flatnonzero(local & (partitions == "calibration"))
        val_pos = np.flatnonzero(local & (partitions == "validation"))
        if min(len(fit_pos), len(cal_pos), len(val_pos)) < 1:
            raise ValueError(f"category {category} has an empty fit/calibration/validation partition")
        fit_features = x[train_indices[fit_pos]]
        cal_features = x[train_indices[cal_pos]]
        val_features = x[train_indices[val_pos]]
        center, basis = _subspace_fit(fit_features, variance_ratio)
        cal_subspace = _subspace_scores(cal_features, center, basis)
        cal_knn: dict[int, np.ndarray] = {}
        for candidate in knn_candidates:
            k = int(candidate)
            cal_score = _knn_scores(cal_features, fit_features, k)
            val_score = _knn_scores(val_features, fit_features, k)
            cal_knn[k] = cal_score
            validation_p[k].append(_upper_p_values(val_score, cal_score))
        models[category] = {
            "center": center,
            "basis": basis,
            "fit": fit_features,
            "cal_subspace": cal_subspace,
            "cal_knn": cal_knn,
        }

    preference = {5: 0, 3: 1, 7: 2}
    quality = {k: _uniform_cvm(np.concatenate(parts)) for k, parts in validation_p.items()}
    selected_k = min(quality, key=lambda k: (quality[k], preference.get(k, 99), k))

    rows: list[dict[str, object]] = []
    for category in categories:
        positions = np.flatnonzero(test_manifest["category"].astype(str).to_numpy() == category)
        if len(positions) == 0:
            continue
        model = models[category]
        category_features = x[test_indices[positions]]
        sub_raw = _subspace_scores(category_features, model["center"], model["basis"])
        sub_p = _upper_p_values(sub_raw, model["cal_subspace"])
        knn_raw = _knn_scores(category_features, model["fit"], selected_k)
        knn_p = _upper_p_values(knn_raw, model["cal_knn"][selected_k])
        sub_loge = np.log(max(1.0 - gamma, _EPS)) - gamma * np.log(np.maximum(sub_p, _EPS))
        knn_loge = np.log(max(1.0 - gamma, _EPS)) - gamma * np.log(np.maximum(knn_p, _EPS))
        for local_position, sr, se, kr, ke in zip(positions, sub_raw, sub_loge, knn_raw, knn_loge):
            source = test_manifest.iloc[int(local_position)]
            for scorer_id, score in (
                ("subspace_raw", sr),
                ("subspace_loge", se),
                ("knn_raw", kr),
                ("knn_loge", ke),
            ):
                rows.append(
                    {
                        "seed": int(seed),
                        "category": str(source["category"]),
                        "sample_id": str(source["sample_id"]),
                        "record_id": str(source["record_id"]),
                        "view_id": _canonical_view_id(source),
                        "scorer_id": scorer_id,
                        "score": float(score),
                    }
                )
    output = pd.DataFrame(rows).sort_values(
        ["seed", "category", "sample_id", "record_id", "scorer_id"], kind="mergesort"
    ).reset_index(drop=True)
    audit = {
        "seed": int(seed),
        "variance_ratio": float(variance_ratio),
        "gamma": float(gamma),
        "knn_candidates": [int(value) for value in knn_candidates],
        "selected_knn_k": int(selected_k),
        "knn_validation_uniform_cvm": {str(k): float(value) for k, value in sorted(quality.items())},
        "categories": len(categories),
        "train_views": int(len(train_manifest)),
        "test_views": int(len(test_manifest)),
        "score_rows": int(len(output)),
        "labels_accessed": False,
    }
    return output, audit


def freeze_view_scores(scores: pd.DataFrame, audit: dict[str, object], output_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / "realiad_per_view_scores.csv"
    scores.to_csv(score_path, index=False, lineterminator="\n")
    record = {
        "file": score_path.name,
        "rows": int(len(scores)),
        "sha256": sha256_file(score_path),
        "audit": audit,
    }
    freeze_path = root / "realiad_per_view_scores_freeze.json"
    freeze_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def score_realiad_views_from_run(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (42, 43, 44),
) -> dict[str, object]:
    root = Path(run_dir)
    train_base = pd.read_csv(root / "manifests" / "train_manifest.csv")
    test = pd.read_csv(root / "manifests" / "test_manifest_unlabeled.csv")
    feature_rows = pd.read_csv(root / "features" / "feature_rows.csv")
    features = np.load(root / "features" / "dinov2_features.npy", mmap_mode="r")
    pieces = []
    audits = []
    for seed in seeds:
        split_path = root / "splits" / f"train_split_seed_{int(seed)}.csv"
        train = pd.read_csv(split_path) if split_path.exists() else train_base
        if "partition" not in train.columns:
            raise FileNotFoundError(f"missing partitioned training split for seed {seed}: {split_path}")
        scored, audit = score_realiad_views(train, test, feature_rows, features, seed=int(seed))
        pieces.append(scored)
        audits.append(audit)
    combined = pd.concat(pieces, ignore_index=True)
    return freeze_view_scores(combined, {"seeds": audits, "source_run": str(root.resolve())}, output_dir)
