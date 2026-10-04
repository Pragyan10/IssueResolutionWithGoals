"""
Select 20 SWE-bench-Live/MultiLang instances for the goal-impact experiment.

Inclusion criteria (applied in this order to a single random permutation):
  1. created_at >= 2025-09-01T00:00:00Z  (after the model's knowledge cutoff)
  2. the issue text is in English          (NEW; see is_english_issue below)
  3. the gold patch resolves in all 3 independent evaluations

Procedure:
  * All date-eligible instances from every language split are shuffled once
    with random seed 42, and that order is saved to randomized_order.json
    BEFORE any evaluation.
  * Candidates are taken in that order. A candidate whose issue is not in
    English is skipped without spending gold runs. The rest get 3 gold runs;
    the first 20 that pass 3/3 are selected.

Changes from the original select_20_multilang.py:
  * English-language criterion (2), checked before gold evaluation.
  * If randomized_order.json already exists, that saved order is reused
    instead of reshuffling. SWE-bench-Live grows over time, so reloading the
    dataset later can change the pool and therefore the shuffle. Delete the
    file (or pass --reshuffle) only if you deliberately want a new order.
  * Gold results already recorded in gold_run_results.csv are reused, so
    previously evaluated candidates are not run again (pass --fresh-gold to
    re-run them).
  * Non-English candidates are recorded in excluded_by_language.json, and the
    language rule is stated in selected_20.json and selection_summary.json.

Applying criterion (2) after the shuffle gives exactly the same sample as the
original run for every English candidate, so previously selected English
instances (and their extracted goals) stay valid; only replacements are new.

Run from the root of a SWE-bench-Live checkout (it calls evaluation.evaluation).
"""

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "SWE-bench-Live/MultiLang"
TARGET_COUNT = 20
RUNS_PER_INSTANCE = 3
RANDOM_SEED = 42
CUTOFF_DATE = datetime(2025, 9, 1, tzinfo=timezone.utc)   # knowledge cutoff 2025-08-31

# English criterion: an issue is non-English if at least this share of the letters
# in its problem_statement prose (code blocks, inline code, URLs and HTML removed)
# come from a script other than Latin or Greek (Greek is allowed as maths notation).
NON_ENGLISH_THRESHOLD = 0.10

OUTPUT_DIR = Path("multilang_gold_selection")
RUNS_DIR = OUTPUT_DIR / "runs"
ORDER_FILE = OUTPUT_DIR / "randomized_order.json"
RUN_LOG_FILE = OUTPUT_DIR / "gold_run_results.csv"
SELECTED_FILE = OUTPUT_DIR / "selected_20.json"
SUMMARY_FILE = OUTPUT_DIR / "selection_summary.json"
DATE_EXCLUDED_FILE = OUTPUT_DIR / "excluded_by_date.json"
LANGUAGE_EXCLUDED_FILE = OUTPUT_DIR / "excluded_by_language.json"

LANGUAGE_RULE = (
    "Issue text must be in English: fewer than 10% of the letters in the "
    "problem_statement prose (code blocks, inline code, URLs and HTML removed) "
    "may come from scripts other than Latin or Greek."
)

# ============================================================
# HELPERS
# ============================================================


def parse_created_at(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        value = str(value).strip()
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def passes_date_filter(instance):
    created_dt = parse_created_at(instance.get("created_at"))
    return created_dt is not None and created_dt >= CUTOFF_DATE


_PROSE_CLEANUP = [
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"`[^`]*`"), " "),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"<[^>]+>"), " "),
]


def non_english_share(text):
    text = str(text or "")
    for pattern, repl in _PROSE_CLEANUP:
        text = pattern.sub(repl, text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    foreign = [c for c in letters
               if not any(s in unicodedata.name(c, "") for s in ("LATIN", "GREEK"))]
    return len(foreign) / len(letters)


def is_english_issue(instance):
    return non_english_share(instance.get("problem_statement")) < NON_ENGLISH_THRESHOLD


def iso(dt):
    return dt.isoformat() if dt else None


# ============================================================
# LOAD POOL AND ORDER
# ============================================================


def load_eligible_instances():
    print("=" * 80)
    print("SWE-BENCH LIVE MULTILANG SAMPLE SELECTION")
    print("=" * 80)
    print(f"\nDataset: {DATASET_NAME}\nCutoff:  {CUTOFF_DATE.isoformat()}\n\nLoading dataset...")
    dataset = load_dataset(DATASET_NAME)

    print("\nAvailable language splits:")
    for split_name in dataset.keys():
        print(f"  {split_name}: {len(dataset[split_name])}")

    eligible, excluded, seen_ids = [], [], set()
    for split_name in dataset.keys():
        for row in dataset[split_name]:
            instance = dict(row)
            instance_id = instance["instance_id"]
            if instance_id in seen_ids:
                continue
            seen_ids.add(instance_id)
            instance["_language"] = split_name
            if not passes_date_filter(instance):
                created_dt = parse_created_at(instance.get("created_at"))
                excluded.append({
                    "instance_id": instance_id, "language": split_name, "repo": instance.get("repo"),
                    "created_at": iso(created_dt) or str(instance.get("created_at")),
                    "reason": "Does not satisfy created_at >= 2025-09-01T00:00:00Z",
                })
                continue
            eligible.append(instance)

    with open(DATE_EXCLUDED_FILE, "w") as f:
        json.dump(excluded, f, indent=2, default=str)

    print(f"\nTotal unique tasks: {len(seen_ids)}")
    print(f"Excluded by date: {len(excluded)}")
    print(f"Eligible post-cutoff: {len(eligible)}")
    if len(eligible) < TARGET_COUNT:
        raise RuntimeError(f"Only {len(eligible)} post-cutoff instances exist, but {TARGET_COUNT} are required.")
    return eligible


def ordered_candidates(eligible, reshuffle):
    """Return eligible instances in the saved random order (or a new one)."""
    by_id = {inst["instance_id"]: inst for inst in eligible}

    if ORDER_FILE.exists() and not reshuffle:
        saved = json.loads(ORDER_FILE.read_text())
        if saved.get("random_seed") != RANDOM_SEED:
            raise RuntimeError(f"{ORDER_FILE} was made with seed {saved.get('random_seed')}, not {RANDOM_SEED}.")
        order_ids = [row["instance_id"] for row in saved["randomized_order"]]
        missing = [i for i in order_ids if i not in by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} instance(s) in the saved order are no longer in the dataset "
                f"(e.g. {missing[:3]}). Pin the dataset revision or pass --reshuffle.")
        new = sorted(set(by_id) - set(order_ids))
        print(f"\nReusing saved random order from {ORDER_FILE} ({len(order_ids)} candidates).")
        if new:
            print(f"Note: {len(new)} instance(s) added to the dataset since then are NOT in the "
                  f"saved order and will not be considered.")
        return [by_id[i] for i in order_ids]

    instances = list(eligible)
    random.Random(RANDOM_SEED).shuffle(instances)
    randomized_order = [{
        "order": i, "instance_id": inst["instance_id"], "language": inst["_language"],
        "repo": inst.get("repo"), "created_at": iso(parse_created_at(inst.get("created_at"))),
    } for i, inst in enumerate(instances, start=1)]
    with open(ORDER_FILE, "w") as f:
        json.dump({"dataset": DATASET_NAME, "random_seed": RANDOM_SEED, "cutoff": CUTOFF_DATE.isoformat(),
                   "eligible_count": len(instances), "randomized_order": randomized_order}, f, indent=2)
    print(f"\nNew randomized order saved to: {ORDER_FILE}")
    return instances


# ============================================================
# GOLD EVALUATION
# ============================================================


def read_gold_result(output_dir, instance_id):
    result_file = Path(output_dir) / "gold_patch_evaluated_instances.jsonl"
    if not result_file.exists():
        return {"resolved": False, "status": "missing_result_file"}
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("instance_id") == instance_id:
                return {"resolved": True, "status": "resolved"}
    return {"resolved": False, "status": "not_resolved"}


def run_gold(instance, attempt):
    instance_id = instance["instance_id"]
    output_dir = RUNS_DIR / instance_id.replace("/", "__") / f"attempt_{attempt}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = ["python", "-m", "evaluation.evaluation",
               "--dataset", DATASET_NAME, "--instance_ids", instance_id, "--platform", "linux",
               "--patch_dir", "gold", "--output_dir", str(output_dir),
               "--workers", "1", "--overwrite", "1"]
    print(f"\n{'=' * 80}\nInstance:  {instance_id}\nLanguage:  {instance['_language']}\n"
          f"Created:   {instance.get('created_at')}\nAttempt:   {attempt}/{RUNS_PER_INSTANCE}\n"
          f"Output:    {output_dir}\n{'=' * 80}")
    process = subprocess.run(command)
    if process.returncode != 0:
        print(f"\nEvaluator returned exit code {process.returncode}")
        return {"resolved": False, "status": "evaluation_process_error",
                "return_code": process.returncode, "output_dir": str(output_dir)}
    result = read_gold_result(output_dir, instance_id)
    print(f"\nResult: {result['status']}")
    return {**result, "return_code": process.returncode, "output_dir": str(output_dir)}


RUN_LOG_COLUMNS = ["candidate_order", "instance_id", "language", "repo", "created_at",
                   "attempt", "status", "resolved", "return_code", "output_dir"]


def load_previous_gold_results():
    """{instance_id: {attempt: row}} from an earlier gold_run_results.csv (last row wins)."""
    previous = {}
    if not RUN_LOG_FILE.exists():
        return previous
    with open(RUN_LOG_FILE, newline="") as f:
        for row in csv.DictReader(f):
            previous.setdefault(row["instance_id"], {})[int(row["attempt"])] = row
    return previous


def append_run_log(row):
    file_exists = RUN_LOG_FILE.exists()
    with open(RUN_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# OUTPUT
# ============================================================


def criteria_block():
    return {
        "model_knowledge_cutoff": "2025-08-31",
        "date_filter": {"field": "created_at", "rule": "created_at >= 2025-09-01T00:00:00Z"},
        "language_filter": {"field": "problem_statement", "rule": LANGUAGE_RULE,
                            "threshold": NON_ENGLISH_THRESHOLD},
        "sampling_rule": ("All date-eligible language splits are combined and shuffled together using "
                          "random seed 42; candidates are taken in that order."),
        "gold_inclusion_rule": "Gold patch must resolve successfully in all three independent evaluations.",
    }


def save_selected(selected):
    with open(SELECTED_FILE, "w") as f:
        json.dump({"dataset": DATASET_NAME, "random_seed": RANDOM_SEED, "runs_per_instance": RUNS_PER_INSTANCE,
                   **criteria_block(), "selected_count": len(selected), "selected": selected},
                  f, indent=2, default=str, ensure_ascii=False)


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reshuffle", action="store_true",
                        help="Ignore the saved randomized_order.json and shuffle again.")
    parser.add_argument("--fresh-gold", action="store_true",
                        help="Re-run gold evaluations even if results are already logged.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    instances = ordered_candidates(load_eligible_instances(), args.reshuffle)
    previous = {} if args.fresh_gold else load_previous_gold_results()

    selected, language_excluded = [], []
    candidates_examined = gold_runs_reused = gold_runs_new = 0

    for candidate_order, instance in enumerate(instances, start=1):
        if len(selected) >= TARGET_COUNT:
            break
        candidates_examined += 1
        instance_id, language, repo = instance["instance_id"], instance["_language"], instance.get("repo", "")
        created_dt = parse_created_at(instance.get("created_at"))
        print(f"\n{'#' * 80}\nCANDIDATE {candidate_order}: {instance_id}\nLanguage: {language}\n"
              f"Created:  {created_dt}\nSelected: {len(selected)}/{TARGET_COUNT}\n{'#' * 80}")

        # ---- English criterion (no gold runs spent on non-English issues) ----
        share = non_english_share(instance.get("problem_statement"))
        if share >= NON_ENGLISH_THRESHOLD:
            language_excluded.append({"candidate_order": candidate_order, "instance_id": instance_id,
                                      "language": language, "repo": repo,
                                      "non_english_share": round(share, 3), "reason": LANGUAGE_RULE})
            with open(LANGUAGE_EXCLUDED_FILE, "w") as f:
                json.dump(language_excluded, f, indent=2, ensure_ascii=False)
            print(f"\n✗ EXCLUDED (not English, non-English share {share:.3f}): {instance_id}")
            continue

        # ---- Gold evaluation, reusing earlier results where available ----
        attempts = []
        for attempt in range(1, RUNS_PER_INSTANCE + 1):
            prev = previous.get(instance_id, {}).get(attempt)
            if prev is not None:
                result = {"resolved": prev["resolved"] == "True", "status": prev["status"],
                          "return_code": prev["return_code"], "output_dir": prev["output_dir"]}
                gold_runs_reused += 1
                print(f"  attempt {attempt}: reusing logged result ({result['status']})")
            else:
                result = run_gold(instance, attempt)
                gold_runs_new += 1
                append_run_log({"candidate_order": candidate_order, "instance_id": instance_id,
                                "language": language, "repo": repo, "created_at": iso(created_dt) or "",
                                "attempt": attempt, "status": result["status"],
                                "resolved": result["resolved"], "return_code": result["return_code"],
                                "output_dir": result["output_dir"]})
            attempts.append(result)

        resolved_count = sum(1 for r in attempts if r["resolved"])
        if resolved_count == RUNS_PER_INSTANCE:
            selected.append({
                "selection_number": len(selected) + 1, "candidate_order": candidate_order,
                "instance_id": instance_id, "language": language, "repo": repo,
                "created_at": iso(created_dt), "base_commit": instance.get("base_commit"),
                "problem_statement": instance.get("problem_statement"),
                "gold_runs_resolved": resolved_count, "gold_runs_total": RUNS_PER_INSTANCE,
                "statuses": [r["status"] for r in attempts],
            })
            save_selected(selected)
            print(f"\n✓ INCLUDED: {instance_id}\nGold: {resolved_count}/{RUNS_PER_INSTANCE}\n"
                  f"Selected: {len(selected)}/{TARGET_COUNT}")
        else:
            print(f"\n✗ EXCLUDED (gold {resolved_count}/{RUNS_PER_INSTANCE}): {instance_id}\n"
                  f"Statuses: {[r['status'] for r in attempts]}")

    save_selected(selected)
    lang_counts = {}
    for s in selected:
        lang_counts[s["language"]] = lang_counts.get(s["language"], 0) + 1
    summary = {
        "dataset": DATASET_NAME, "random_seed": RANDOM_SEED, **criteria_block(),
        "minimum_created_at": CUTOFF_DATE.isoformat(),
        "eligible_instances_after_date_filter": len(instances),
        "runs_per_instance": RUNS_PER_INSTANCE, "required_gold_passes": RUNS_PER_INSTANCE,
        "target_count": TARGET_COUNT, "candidates_examined": candidates_examined,
        "excluded_by_language": [e["instance_id"] for e in language_excluded],
        "gold_runs_new": gold_runs_new, "gold_runs_reused": gold_runs_reused,
        "selected_count": len(selected), "selected_language_counts": lang_counts,
        "selected_instance_ids": [s["instance_id"] for s in selected],
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}\nSELECTION COMPLETE\n{'=' * 80}")
    print(f"\nEligible post-cutoff: {len(instances)}\nCandidates examined: {candidates_examined}")
    print(f"Excluded as non-English: {len(language_excluded)}")
    print(f"Gold runs: {gold_runs_new} new, {gold_runs_reused} reused from {RUN_LOG_FILE}")
    print(f"Selected: {len(selected)}\n\nSELECTED INSTANCES\n{'-' * 80}")
    for s in selected:
        print(f"{s['selection_number']:02d}. {s['instance_id']} | {s['language']} | "
              f"{s['created_at']} | {s['gold_runs_resolved']}/3 | candidate #{s['candidate_order']}")
    if len(selected) < TARGET_COUNT:
        print(f"\nWARNING: only {len(selected)} instances met all criteria.")


if __name__ == "__main__":
    main()
