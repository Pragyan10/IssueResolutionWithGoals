import csv
import json
import random
import subprocess
from pathlib import Path

from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"

TARGET_COUNT = 20
RUNS_PER_INSTANCE = 3

# Fixed seed so your sampling order can be reproduced.
RANDOM_SEED = 42

# SWE-bench CLI timeout per instance, in seconds.
TIMEOUT = 1800

OUTPUT_DIR = Path("verified_gold_selection")
OUTPUT_DIR.mkdir(exist_ok=True)

ORDER_FILE = OUTPUT_DIR / "randomized_order.json"
RUN_LOG_FILE = OUTPUT_DIR / "gold_run_results.csv"
SELECTED_FILE = OUTPUT_DIR / "selected_20.json"
SUMMARY_FILE = OUTPUT_DIR / "selection_summary.json"

SWE_LOG_DIR = Path("logs/evaluation")


# ============================================================
# HELPERS
# ============================================================

def load_results(run_id):
    """
    Read SWE-bench's results.json for one run.
    """
    result_file = SWE_LOG_DIR / run_id / "results.json"

    if not result_file.exists():
        return None

    with open(result_file, "r") as f:
        return json.load(f)


def is_instance_resolved(results, instance_id):
    """
    Determine whether this instance was resolved.

    Handles a few likely result-field names defensively.
    """

    if results is None:
        return False

    # Current/older SWE-bench result formats may use
    # slightly different keys.
    possible_keys = [
        "resolved_ids",
        "resolved_instances",
    ]

    for key in possible_keys:
        value = results.get(key)

        if isinstance(value, list):
            if instance_id in value:
                return True

        # Some summaries may store a numeric count instead.
        if isinstance(value, int):
            if value == 1:
                return True

    return False


def classify_result(results, instance_id):
    """
    Produce a readable status for our own experiment log.
    """

    if results is None:
        return "missing_results"

    if is_instance_resolved(results, instance_id):
        return "resolved"

    categories = {
        "unresolved_ids": "unresolved",
        "unresolved_instances": "unresolved",
        "infra_failure_ids": "infrastructure_failure",
        "infrastructure_failure_ids": "infrastructure_failure",
        "ambiguous_failure_ids": "ambiguous_failure",
        "error_ids": "error",
        "errors": "error",
        "empty_patch_ids": "empty_patch",
    }

    for key, label in categories.items():
        value = results.get(key)

        if isinstance(value, list) and instance_id in value:
            return label

    return "not_resolved"


def run_gold(instance_id, attempt):
    """
    Run one independent gold-patch evaluation.
    """

    # Unique run ID prevents SWE-bench from reusing cached results.
    safe_id = instance_id.replace("/", "__")

    run_id = (
        f"verified-gold-selection-"
        f"{safe_id}-"
        f"attempt-{attempt}"
    )

    command = [
        "swebench",
        "eval",
        "verified",
        "--gold",
        "-i",
        instance_id,
        "-j",
        "1",
        "--timeout",
        str(TIMEOUT),
        "--run-id",
        run_id,
    ]

    print()
    print("=" * 80)
    print(f"Instance: {instance_id}")
    print(f"Attempt:  {attempt}/{RUNS_PER_INSTANCE}")
    print(f"Run ID:   {run_id}")
    print("=" * 80)

    process = subprocess.run(command)

    if process.returncode != 0:
        print(
            f"WARNING: SWE-bench exited with "
            f"code {process.returncode}"
        )

    results = load_results(run_id)

    status = classify_result(
        results,
        instance_id,
    )

    resolved = status == "resolved"

    print(
        f"\nResult: {status}"
    )

    return {
        "run_id": run_id,
        "status": status,
        "resolved": resolved,
        "return_code": process.returncode,
    }


def append_run_log(row):
    """
    Append one attempt to CSV immediately so results survive
    interruptions.
    """

    fieldnames = [
        "candidate_order",
        "instance_id",
        "repo",
        "attempt",
        "run_id",
        "status",
        "resolved",
        "return_code",
    ]

    exists = RUN_LOG_FILE.exists()

    with open(
        RUN_LOG_FILE,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def save_selected(selected):
    """
    Persist selected tasks after every accepted instance.
    """

    data = {
        "dataset": DATASET_NAME,
        "random_seed": RANDOM_SEED,
        "runs_per_instance": RUNS_PER_INSTANCE,
        "inclusion_rule": (
            "Gold patch must resolve successfully "
            "in all 3 independent SWE-bench evaluations."
        ),
        "selected_count": len(selected),
        "selected": selected,
    }

    with open(SELECTED_FILE, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SWE-BENCH VERIFIED GOLD-PATCH SELECTION")
    print("=" * 80)

    print("\nLoading SWE-bench Verified...")

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
    )

    instances = list(dataset)

    print(
        f"Loaded {len(instances)} instances."
    )

    # --------------------------------------------------------
    # Create the random ordering once.
    # --------------------------------------------------------

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(instances)

    randomized_order = [
        {
            "order": i,
            "instance_id": instance["instance_id"],
            "repo": instance["repo"],
        }
        for i, instance in enumerate(
            instances,
            start=1,
        )
    ]

    with open(ORDER_FILE, "w") as f:
        json.dump(
            {
                "dataset": DATASET_NAME,
                "seed": RANDOM_SEED,
                "order": randomized_order,
            },
            f,
            indent=2,
        )

    print(
        f"Randomized order saved to {ORDER_FILE}"
    )

    selected = []
    candidates_examined = 0

    # --------------------------------------------------------
    # Walk through candidates until 20 pass 3/3.
    # --------------------------------------------------------

    for candidate_order, instance in enumerate(
        instances,
        start=1,
    ):

        if len(selected) >= TARGET_COUNT:
            break

        candidates_examined += 1

        instance_id = instance["instance_id"]
        repo = instance["repo"]

        print()
        print("#" * 80)
        print(
            f"CANDIDATE {candidate_order}: "
            f"{instance_id}"
        )
        print(
            f"Currently selected: "
            f"{len(selected)}/{TARGET_COUNT}"
        )
        print("#" * 80)

        attempts = []

        # ----------------------------------------------------
        # Three independent gold runs
        # ----------------------------------------------------

        for attempt in range(
            1,
            RUNS_PER_INSTANCE + 1,
        ):

            result = run_gold(
                instance_id,
                attempt,
            )

            attempts.append(result)

            append_run_log(
                {
                    "candidate_order":
                        candidate_order,
                    "instance_id":
                        instance_id,
                    "repo":
                        repo,
                    "attempt":
                        attempt,
                    "run_id":
                        result["run_id"],
                    "status":
                        result["status"],
                    "resolved":
                        result["resolved"],
                    "return_code":
                        result["return_code"],
                }
            )

        resolved_count = sum(
            result["resolved"]
            for result in attempts
        )

        include = (
            resolved_count
            == RUNS_PER_INSTANCE
        )

        # ----------------------------------------------------
        # Include only 3/3
        # ----------------------------------------------------

        if include:

            selected_number = len(selected) + 1

            selected_instance = {
                "selection_number":
                    selected_number,
                "candidate_order":
                    candidate_order,
                "instance_id":
                    instance_id,
                "repo":
                    repo,
                "base_commit":
                    instance.get("base_commit"),
                "problem_statement":
                    instance.get(
                        "problem_statement"
                    ),
                "gold_runs_resolved":
                    resolved_count,
                "gold_runs_total":
                    RUNS_PER_INSTANCE,
                "statuses": [
                    r["status"]
                    for r in attempts
                ],
            }

            selected.append(
                selected_instance
            )

            save_selected(selected)

            print()
            print(
                f"✓ INCLUDED: {instance_id}"
            )
            print(
                f"Gold result: "
                f"{resolved_count}/"
                f"{RUNS_PER_INSTANCE}"
            )
            print(
                f"Selected: "
                f"{len(selected)}/"
                f"{TARGET_COUNT}"
            )

        else:

            print()
            print(
                f"✗ EXCLUDED: {instance_id}"
            )
            print(
                f"Gold result: "
                f"{resolved_count}/"
                f"{RUNS_PER_INSTANCE}"
            )
            print(
                "Statuses:",
                [
                    r["status"]
                    for r in attempts
                ],
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary = {
        "dataset": DATASET_NAME,
        "seed": RANDOM_SEED,
        "target_count": TARGET_COUNT,
        "runs_per_instance":
            RUNS_PER_INSTANCE,
        "candidates_examined":
            candidates_examined,
        "selected_count":
            len(selected),
        "selected_instance_ids": [
            x["instance_id"]
            for x in selected
        ],
    }

    with open(
        SUMMARY_FILE,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("SELECTION FINISHED")
    print("=" * 80)

    print(
        f"\nCandidates examined: "
        f"{candidates_examined}"
    )

    print(
        f"Instances selected: "
        f"{len(selected)}"
    )

    print("\nSelected instances:")

    for item in selected:

        print(
            f"{item['selection_number']:02d}. "
            f"{item['instance_id']}"
        )

    print("\nFiles:")
    print(f"  {ORDER_FILE}")
    print(f"  {RUN_LOG_FILE}")
    print(f"  {SELECTED_FILE}")
    print(f"  {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
