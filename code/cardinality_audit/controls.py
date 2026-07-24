from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .aggregators import aggregate
from .metrics import binary_metrics

GROUP_KEYS = ["dataset", "scorer_id", "seed", "category", "group_id", "label", "defect"]
SAMPLE_KEYS = ["dataset", "seed", "category", "group_id", "label", "defect"]


def _stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return (int(base_seed) + zlib.crc32(text)) % (2**32 - 1)


def aggregate_groups(
    view_scores: pd.DataFrame,
    aggregator_ids: list[str] | tuple[str, ...],
    *,
    seed: int = 42,
    subset_cap: int = 128,
    tau: float = 1.0,
) -> pd.DataFrame:
    missing = set(GROUP_KEYS + ["view_id", "score"]) - set(view_scores.columns)
    if missing:
        raise ValueError(f"view score table missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for key, group in view_scores.groupby(GROUP_KEYS, sort=True, dropna=False):
        meta = dict(zip(GROUP_KEYS, key))
        scores = group.sort_values("view_id", kind="mergesort")["score"].to_numpy(dtype=float)
        for aggregator_id in aggregator_ids:
            value = aggregate(
                scores,
                str(aggregator_id),
                seed=_stable_seed(seed, *key, aggregator_id),
                subset_cap=subset_cap,
                tau=tau,
            )
            if np.isnan(value):
                continue
            rows.append({**meta, "views": int(len(scores)), "aggregator_id": str(aggregator_id), "score": float(value)})
    return pd.DataFrame(rows)


def _metrics_by_cell(group_scores: pd.DataFrame, extra: dict[str, object] | None = None) -> pd.DataFrame:
    rows = []
    for key, group in group_scores.groupby(["dataset", "scorer_id", "seed", "aggregator_id"], sort=True):
        metrics = binary_metrics(group["label"], group["score"])
        rows.append(
            {
                "dataset": key[0],
                "scorer_id": key[1],
                "seed": int(key[2]),
                "aggregator_id": key[3],
                "groups": int(len(group)),
                **metrics,
                **(extra or {}),
            }
        )
    return pd.DataFrame(rows)


def sample_equal_views(
    view_scores: pd.DataFrame,
    *,
    budget: int,
    seed: int,
) -> pd.DataFrame:
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive")
    sampled_parts = []
    for key, group in view_scores.groupby(SAMPLE_KEYS, sort=True, dropna=False):
        scorer_sets = [set(part["view_id"].astype(str)) for _, part in group.groupby("scorer_id", sort=True)]
        if not scorer_sets:
            continue
        reference = scorer_sets[0]
        if any(current != reference for current in scorer_sets[1:]):
            raise ValueError(f"scorers do not share identical view identities for group {key}")
        ordered = np.asarray(sorted(reference), dtype=object)
        if len(ordered) < budget:
            raise ValueError("at least one group has fewer views than the equal-view budget")
        rng = np.random.default_rng(_stable_seed(seed, "equal_sample", *key))
        selected = set(rng.choice(ordered, size=budget, replace=False).tolist())
        sampled_parts.append(group.loc[group["view_id"].astype(str).isin(selected)])
    return pd.concat(sampled_parts, ignore_index=True) if sampled_parts else view_scores.iloc[0:0].copy()


def equal_view_repeats(
    view_scores: pd.DataFrame,
    aggregator_ids: list[str] | tuple[str, ...],
    *,
    budget: int,
    repeats: int = 1000,
    seed: int = 42,
    subset_cap: int = 128,
    tau: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for repeat in range(int(repeats)):
        sampled = sample_equal_views(
            view_scores, budget=int(budget), seed=_stable_seed(seed, "equal", repeat)
        )
        scores = aggregate_groups(
            sampled, aggregator_ids, seed=_stable_seed(seed, repeat), subset_cap=subset_cap, tau=tau
        )
        metrics = _metrics_by_cell(scores, {"repeat": repeat, "budget": int(budget)})
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def size_only_null(
    view_scores: pd.DataFrame,
    aggregator_ids: list[str] | tuple[str, ...],
    *,
    repeats: int = 1000,
    seed: int = 42,
    subset_cap: int = 128,
    tau: float = 1.0,
) -> pd.DataFrame:
    group_meta = (
        view_scores.groupby(GROUP_KEYS, sort=True, dropna=False)
        .size()
        .rename("views")
        .reset_index()
    )
    rows: list[pd.DataFrame] = []
    for repeat in range(int(repeats)):
        rng = np.random.default_rng(_stable_seed(seed, "size_null", repeat))
        group_rows = []
        for item in group_meta.itertuples(index=False):
            scores = rng.normal(size=int(item.views))
            for aggregator_id in aggregator_ids:
                value = aggregate(
                    scores,
                    str(aggregator_id),
                    seed=_stable_seed(seed, repeat, item.group_id, aggregator_id),
                    subset_cap=subset_cap,
                    tau=tau,
                )
                if np.isnan(value):
                    continue
                group_rows.append(
                    {
                        "dataset": item.dataset,
                        "scorer_id": item.scorer_id,
                        "seed": int(item.seed),
                        "category": item.category,
                        "group_id": item.group_id,
                        "label": int(item.label),
                        "defect": item.defect,
                        "views": int(item.views),
                        "aggregator_id": str(aggregator_id),
                        "score": float(value),
                    }
                )
        metrics = _metrics_by_cell(pd.DataFrame(group_rows), {"repeat": repeat, "control_id": "size_only_iid"})
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def permutation_null(
    view_scores: pd.DataFrame,
    aggregator_ids: list[str] | tuple[str, ...],
    *,
    repeats: int = 1000,
    seed: int = 42,
    subset_cap: int = 128,
    tau: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for repeat in range(int(repeats)):
        permuted_parts = []
        for key, frame in view_scores.groupby(["dataset", "scorer_id", "seed"], sort=True):
            copy = frame.copy()
            rng = np.random.default_rng(_stable_seed(seed, "perm", repeat, *key))
            copy["score"] = rng.permutation(copy["score"].to_numpy(dtype=float))
            permuted_parts.append(copy)
        permuted = pd.concat(permuted_parts, ignore_index=True)
        scores = aggregate_groups(permuted, aggregator_ids, seed=_stable_seed(seed, "permagg", repeat), subset_cap=subset_cap, tau=tau)
        rows.append(_metrics_by_cell(scores, {"repeat": repeat, "control_id": "score_permutation"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cardinality_assignment_repeats(
    view_scores: pd.DataFrame,
    aggregator_ids: list[str] | tuple[str, ...],
    *,
    mode: str,
    repeats: int = 1000,
    seed: int = 42,
    subset_cap: int = 128,
    tau: float = 1.0,
) -> pd.DataFrame:
    grouped = list(view_scores.groupby(SAMPLE_KEYS, sort=True, dropna=False))
    allowed = {"independent_1to5", "correlated_1to5", "correlated_4to5"}
    if mode not in allowed:
        raise ValueError(f"unknown cardinality assignment mode: {mode}")
    for key, group in grouped:
        scorer_sets = [set(part["view_id"].astype(str)) for _, part in group.groupby("scorer_id", sort=True)]
        if not scorer_sets or any(current != scorer_sets[0] for current in scorer_sets[1:]):
            raise ValueError(f"scorers do not share identical view identities for group {key}")
        if len(scorer_sets[0]) < 5:
            raise ValueError("cardinality assignment controls require at least five source views per group")
    rows = []
    for repeat in range(int(repeats)):
        sampled_parts = []
        for key, group in grouped:
            label = int(key[4])
            rng = np.random.default_rng(_stable_seed(seed, mode, repeat, *key))
            if mode == "independent_1to5":
                budget = int(rng.integers(1, 6))
            elif mode == "correlated_1to5":
                budget = int(rng.integers(1, 3)) if label == 0 else int(rng.integers(4, 6))
            else:
                budget = 4 if label == 0 else 5
            ordered = np.asarray(sorted(set(group["view_id"].astype(str))), dtype=object)
            selected = set(rng.choice(ordered, size=budget, replace=False).tolist())
            sampled_parts.append(group.loc[group["view_id"].astype(str).isin(selected)])
        sampled = pd.concat(sampled_parts, ignore_index=True)
        scores = aggregate_groups(
            sampled, aggregator_ids, seed=_stable_seed(seed, mode, repeat, "aggregate"),
            subset_cap=subset_cap, tau=tau
        )
        metrics = _metrics_by_cell(scores, {"repeat": repeat, "control_id": mode})
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
