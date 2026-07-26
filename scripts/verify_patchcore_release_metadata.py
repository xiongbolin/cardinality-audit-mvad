from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "results" / "current" / "patchcore_official"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tol: float = 1e-12) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol), f"{actual} != {expected}")


def main() -> int:
    metrics = rows(PC / "tables" / "patchcore_protocol_a_metrics.csv")
    require(len(metrics) == 1, "expected one Protocol A metrics row")
    close(float(metrics[0]["auroc"]), 0.8294520757621477)
    close(float(metrics[0]["auprc"]), 0.9889990554024957)

    image = rows(PC / "validation" / "patchcore_image_summary_recomputed.csv")
    require(len(image) == 1, "expected one recomputed image summary row")
    close(float(image[0]["category_macro_auroc"]), 0.827871332521187)
    require(int(image[0]["category_count"]) == 18, "expected 18 categories")
    require(int(image[0]["test_images"]) == 2985, "expected 2,985 test images")

    equal = {int(row["budget"]): row for row in rows(PC / "validation" / "patchcore_equal_budget_macro_summary.csv")}
    require(set(equal) == {1, 2, 3, 4}, "expected Equal-1 through Equal-4")
    close(float(equal[1]["category_macro_auroc_mean"]), 0.8187222222222221)
    close(float(equal[4]["category_macro_auroc_mean"]), 0.8674212962962963)

    categories = rows(PC / "validation" / "patchcore_category_metrics_recomputed.csv")
    require(len(categories) == 18, "expected 18 category rows")

    checkpoints = rows(PC / "checkpoint_asset_register.csv")
    require(len(checkpoints) == 18, "expected 18 checkpoint records")
    require(all(int(row["size_bytes"]) == 128228847 for row in checkpoints), "unexpected checkpoint size")
    require(all(len(row["sha256"]) == 64 for row in checkpoints), "invalid checkpoint hash")
    require(not list(ROOT.rglob("model.ckpt")), "checkpoint binaries must not be committed to ordinary Git history")

    parts = rows(PC / "source_part_register.csv")
    require(len(parts) == 6, "expected six release-asset ZIP records")
    require(all(len(row["sha256"]) == 64 for row in parts), "invalid release-asset hash")

    print("PatchCore release metadata verification: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PatchCore release metadata verification: FAIL: {exc}", file=sys.stderr)
        raise
