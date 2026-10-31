#!/usr/bin/env python3

import csv
import json
from collections import Counter
from pathlib import Path


MANIFEST_PATH = Path("analysis/expanded100_manifest.json")
OUTPUT_PATH = Path("analysis/expanded100_execution_audit.csv")


def classify_model(record):
    status = record.get("status")
    stages = record.get("stages", {})

    if status == "test_complete":
        return "completed", ""

    resource_failures = []

    for stage_name, stage in stages.items():
        if stage.get("status") != "resource_censored":
            continue

        failure_kind = stage.get("failure_kind")

        if failure_kind == "timeout_exceeded":
            resource_failures.append(
                ("timeout_censored", stage_name)
            )
        elif failure_kind == "memory_limit_exceeded":
            resource_failures.append(
                ("memory_censored", stage_name)
            )
        else:
            raise RuntimeError(
                f"{record['model_id']}: unknown resource failure "
                f"{failure_kind!r} in stage {stage_name!r}"
            )

    if len(resource_failures) > 1:
        raise RuntimeError(
            f"{record['model_id']}: multiple resource-censored stages: "
            f"{resource_failures}"
        )

    if resource_failures:
        return resource_failures[0]

    candidates = stages.get("candidates", {})
    pretest = stages.get("pretest", {})

    if (
        status == "failed"
        and candidates.get("status") == "running"
        and pretest.get("status") == "failed"
        and pretest.get("failure_kind") == "worker_nonzero_exit"
    ):
        return "unresolved_worker_exit", "candidates"

    raise RuntimeError(
        f"{record['model_id']}: unclassified execution outcome"
    )


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    models = manifest["models"]

    if len(models) != 100:
        raise RuntimeError(
            f"Expected 100 models, found {len(models)}"
        )

    rows = []

    for model_id in sorted(models):
        record = models[model_id]

        if record.get("model_id") != model_id:
            raise RuntimeError(
                f"Manifest key/model_id mismatch for {model_id}"
            )

        audit_category, stage = classify_model(record)

        rows.append(
            {
                "model_id": model_id,
                "manifest_status": record.get("status", ""),
                "audit_category": audit_category,
                "stage": stage,
            }
        )

    counts = Counter(row["audit_category"] for row in rows)

    expected_counts = {
        "completed": 75,
        "timeout_censored": 12,
        "memory_censored": 2,
        "unresolved_worker_exit": 11,
    }

    if dict(counts) != expected_counts:
        raise RuntimeError(
            f"Unexpected audit counts: {dict(counts)}"
        )

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=[
                "model_id",
                "manifest_status",
                "audit_category",
                "stage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")

    for category, count in expected_counts.items():
        print(f"{category}: {count}")


if __name__ == "__main__":
    main()
