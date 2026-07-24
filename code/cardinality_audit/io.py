from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NORMALIZED_COLUMNS = [
    "dataset",
    "scorer_id",
    "seed",
    "category",
    "group_id",
    "view_id",
    "label",
    "defect",
    "score",
]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _view_id(path_value: object) -> str:
    value = str(path_value).replace("\\", "/")
    parts = [part for part in value.split("/") if part]
    for index, part in enumerate(parts):
        if part.lower() == "test" and index >= 1 and index + 2 < len(parts):
            return "/".join(parts[index - 1 :])
    return value


def _standardize(
    frame: pd.DataFrame,
    *,
    dataset: str,
    scorer_id: str,
    score_column: str,
    seed: int = 0,
    path_column: str = "path",
    group_column: str = "condition_group",
    defect_column: str = "defect",
) -> pd.DataFrame:
    _require_columns(
        frame,
        [path_column, "category", group_column, "label", defect_column, score_column],
        scorer_id,
    )
    out = pd.DataFrame(
        {
            "dataset": dataset,
            "scorer_id": scorer_id,
            "seed": int(seed),
            "category": frame["category"].astype(str),
            "group_id": frame[group_column].astype(str),
            "view_id": frame[path_column].map(_view_id),
            "label": pd.to_numeric(frame["label"], errors="raise").astype(int),
            "defect": frame[defect_column].astype(str),
            "score": pd.to_numeric(frame[score_column], errors="raise").astype(float),
        }
    )
    return out


def load_rad_scores(input_dir: str | Path) -> pd.DataFrame:
    root = Path(input_dir)
    pieces: list[pd.DataFrame] = []

    dino_path = root / "dinov2_vit_b14_visual_image_scores.csv"
    if dino_path.exists():
        dino = pd.read_csv(dino_path)
        _require_columns(dino, ["method", "score_z"], dino_path.name)
        for method, scorer_id in {
            "knn": "dinov2_knn",
            "isolation_forest": "dinov2_if",
            "one_class_svm": "dinov2_ocsvm",
        }.items():
            selected = dino.loc[dino["method"].astype(str) == method]
            if not selected.empty:
                pieces.append(_standardize(selected, dataset="RAD", scorer_id=scorer_id, score_column="score_z"))

    clip_path = root / "clip_vit_b16_robovlm_image_scores.csv"
    if clip_path.exists():
        clip = pd.read_csv(clip_path)
        for score_column, scorer_id in {
            "visual_knn_zscore": "clip_visual",
            "clip_zero_shot_generic": "clip_prompt_generic",
            "robovlm_fusion_knn": "clip_fusion",
        }.items():
            if score_column in clip.columns:
                pieces.append(_standardize(clip, dataset="RAD", scorer_id=scorer_id, score_column=score_column))

    baseline_specs = [
        ("strong_baseline_image_scores.csv", {
            "PatchCore-style": "patchcore_r18",
            "PaDiM-style": "padim_r18",
            "WinCLIP-style": "winclip_style",
            "AnomalyCLIP-style prompts": "anomalyclip_style",
        }),
        ("strong_baseline_wide_resnet50_2_image_scores.csv", {
            "PatchCore-style": "patchcore_wrn50",
        }),
        ("efficientad_style_wide_resnet50_2_image_scores.csv", {
            "EfficientAD-style teacher-student": "efficientad_style",
        }),
    ]
    for filename, method_map in baseline_specs:
        path = root / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        _require_columns(frame, ["method", "score"], filename)
        for method, scorer_id in method_map.items():
            selected = frame.loc[frame["method"].astype(str) == method]
            if not selected.empty:
                pieces.append(_standardize(selected, dataset="RAD", scorer_id=scorer_id, score_column="score"))

    if not pieces:
        raise FileNotFoundError(f"No registered RAD score files found in {root}")
    out = pd.concat(pieces, ignore_index=True)
    return validate_normalized_scores(out)


def load_realiad_view_scores(score_path: str | Path, label_path: str | Path) -> pd.DataFrame:
    scores = pd.read_csv(score_path)
    labels = pd.read_csv(label_path)
    _require_columns(scores, ["seed", "category", "sample_id", "record_id", "scorer_id", "score"], "Real-IAD view scores")
    _require_columns(labels, ["category", "sample_id", "label", "anomaly_class"], "Real-IAD sealed labels")
    if "label" in scores.columns or "anomaly_class" in scores.columns:
        raise ValueError("Real-IAD score table must remain label-free before the merge")
    if labels.duplicated(["category", "sample_id"]).any():
        raise ValueError("Real-IAD sealed labels contain duplicate sample identities")
    merged = scores.merge(
        labels[["category", "sample_id", "label", "anomaly_class"]],
        on=["category", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["label"].isna().any():
        raise ValueError("Real-IAD score table contains samples missing sealed labels")
    out = pd.DataFrame(
        {
            "dataset": "Real-IAD",
            "scorer_id": merged["scorer_id"].astype(str),
            "seed": pd.to_numeric(merged["seed"], errors="raise").astype(int),
            "category": merged["category"].astype(str),
            "group_id": merged["sample_id"].astype(str),
            "view_id": merged["view_id"].astype(str) if "view_id" in merged.columns else merged["record_id"].astype(str),
            "label": pd.to_numeric(merged["label"], errors="raise").astype(int),
            "defect": merged["anomaly_class"].astype(str),
            "score": pd.to_numeric(merged["score"], errors="raise").astype(float),
        }
    )
    return validate_normalized_scores(out)


def validate_normalized_scores(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, NORMALIZED_COLUMNS, "normalized score table")
    out = frame[NORMALIZED_COLUMNS].copy()
    if out[NORMALIZED_COLUMNS].isna().any().any():
        raise ValueError("normalized score table contains missing values")
    if not np.isfinite(out["score"].to_numpy(dtype=float)).all():
        raise ValueError("normalized score table contains non-finite scores")
    if not set(out["label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("labels must be binary 0/1")
    identity = ["dataset", "scorer_id", "seed", "group_id", "view_id"]
    if out.duplicated(identity).any():
        raise ValueError("duplicate view identity in normalized score table")
    label_counts = out.groupby(["dataset", "seed", "group_id"], sort=False)["label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("inconsistent label within group")
    category_counts = out.groupby(["dataset", "seed", "group_id"], sort=False)["category"].nunique()
    if (category_counts > 1).any():
        raise ValueError("inconsistent category within group")
    out["seed"] = out["seed"].astype(int)
    out["label"] = out["label"].astype(int)
    out["score"] = out["score"].astype(float)
    return out.sort_values(["scorer_id", "view_id"], kind="mergesort").reset_index(drop=True)
