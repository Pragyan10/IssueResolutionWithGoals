#!/usr/bin/env python3
"""
Generate patches with mini-swe-agent for the goal-impact experiment.

Conditions (choose with --conditions):
  baseline   Task + Issue only.
  goal       Task + Patch Procedure + Issue + Extracted Goals (goal-guided).
  procedure  Task + Patch Procedure + Issue, without goals (ablation: separates the
             effect of the goals from the effect of the step-by-step procedure).

Inputs: the goal files produced by extract_goals_claude_v3.py, e.g.
  swe_verified_selected_20_with_goals.json   (SWE-bench Verified)
  swe_live_selected_20_with_goals.json       (SWE-bench-Live)

All conditions use exactly the same issue title and description (title = first
line of problem_statement, duplicated text collapsed), so prompts differ only in
the goal / procedure sections.

Outputs (under --output-dir, default ./patch_runs):
  <dataset>/<condition>/rep_<n>/<instance_id>/prompt.txt
  <dataset>/<condition>/rep_<n>/<instance_id>/patch.diff
  <dataset>/<condition>/rep_<n>/<instance_id>/<instance_id>.traj.json
  <dataset>/<condition>/rep_<n>/preds.json        (SWE-bench prediction format)
  run_manifest.json                               (one entry per run)

Usage:
  python run_patches.py FILES... --check-images              # pull all Docker images first
  python run_patches.py FILES... --dry-run                   # write prompts only, no model calls
  python run_patches.py FILES... --conditions baseline goal --reps 3 --workers 2
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# ============================================================
# PROMPTS
# ============================================================

TASK_BASELINE = """### Task
You are fixing a reported issue in this repository. The Issue section describes what is wrong and what behavior is expected.
Your objective is a patch that resolves the issue."""

TASK_GOAL = """### Task
You are fixing a reported issue in this repository. The sections below are:
1. Patch Procedure: how to use the goals to find what must change and to implement the patch.
2. Issue: what is wrong and what behavior is expected.
3. Extracted Goals: the stakeholder intent behind the issue, stated as high-level goals and formalized in KAOS notation. They describe the full set of cases in which the expected behavior must hold.
Your objective is a patch that resolves the issue in every case the goals cover, without changing behavior that neither the issue nor the goals require."""

TASK_PROCEDURE = """### Task
You are fixing a reported issue in this repository. The sections below are:
1. Patch Procedure: how to find what must change and to implement the patch.
2. Issue: what is wrong and what behavior is expected.
Your objective is a patch that resolves the issue in every case it covers, without changing behavior that the issue does not require."""

PROCEDURE_GOAL = """### Patch Procedure
Step 1 - Goals to behaviors: For each goal, state the observable behavior it requires and the cases it covers, including cases beyond the example in the issue.
Step 2 - Behaviors to files: Find every component and file involved in producing those behaviors. Look beyond the location the issue names: search for other places where the same cause can occur (the same code pattern, sibling classes, other callers or code paths).
Step 3 - Files to code locations: In those files, identify the specific functions and lines that must change. For each location you inspected but decided not to change, state why.
Step 4 - Patch: Implement the changes at every location identified in Step 3.
Step 5 - Verify: Reproduce the issue, then check the fix against the cases each goal covers, not only the issue's example.

Rules:
- Address every applicable goal. If goals offer alternatives, choose the one best supported by the issue and the codebase.
- Do not make changes that neither the issue nor the goals require.
- The formalized goals describe required behavior, not implementation steps.
- Preserve existing behavior unless the issue or goals require otherwise.
- The issue is authoritative about what is wrong and what behavior is expected. The goals define the cases in which that behavior must hold."""

PROCEDURE_ONLY = """### Patch Procedure
Step 1 - Issue to behaviors: State the observable behavior the issue expects and the cases it covers, including cases beyond the example in the issue.
Step 2 - Behaviors to files: Find every component and file involved in producing those behaviors. Look beyond the location the issue names: search for other places where the same cause can occur (the same code pattern, sibling classes, other callers or code paths).
Step 3 - Files to code locations: In those files, identify the specific functions and lines that must change. For each location you inspected but decided not to change, state why.
Step 4 - Patch: Implement the changes at every location identified in Step 3.
Step 5 - Verify: Reproduce the issue, then check the fix against the cases the issue covers, not only its example.

Rules:
- Do not make changes that the issue does not require.
- Preserve existing behavior unless the issue requires otherwise."""


def issue_section(title: str, description: str) -> str:
    return f"### Issue\nTitle: {title}\nDescription:\n{description}"


def goals_section(goals: list[dict[str, str]]) -> str:
    parts = ["### Extracted Goals"]
    for i, g in enumerate(goals, start=1):
        parts.append(
            f"Goal {i}:\n"
            f"Goal type: {g['goal_type']}\n"
            f"High-level goal: {g['high_level_goal']}\n"
            f"Formalized goal: {g['complete_formalized_goal']}"
        )
    return "\n\n".join(parts)


def build_prompt(condition: str, title: str, description: str, goals: list[dict]) -> str:
    if condition == "baseline":
        sections = [TASK_BASELINE, issue_section(title, description)]
    elif condition == "goal":
        if not goals:
            raise ValueError("goal condition requires extracted_goals")
        sections = [TASK_GOAL, PROCEDURE_GOAL, issue_section(title, description), goals_section(goals)]
    elif condition == "procedure":
        sections = [TASK_PROCEDURE, PROCEDURE_ONLY, issue_section(title, description)]
    else:
        raise ValueError(f"unknown condition {condition}")
    return "\n\n".join(sections)


# ============================================================
# ISSUE TEXT (identical rules to the goal extractor)
# ============================================================

def dedupe_repeated_text(text: str) -> str:
    text = text.strip()
    half = len(text) // 2
    for cut in range(max(1, half - 3), half + 4):
        first, second = text[:cut].strip(), text[cut:].strip()
        if first and first == second:
            return first
    return text


def issue_title_and_description(instance: dict[str, Any]) -> tuple[str, str]:
    statement = dedupe_repeated_text(str(instance.get("problem_statement") or ""))
    first_line, _, rest = statement.partition("\n")
    title, description = first_line.strip(), rest.strip()
    used = (instance.get("goal_extraction_trace") or {}).get("issue_title_used")
    if used and used != title:
        print(f"  note: {instance['instance_id']}: title differs from extractor's; using first line",
              file=sys.stderr)
    return title, description or title


# ============================================================
# DATASETS AND IMAGES
# ============================================================

def dataset_key(data: dict[str, Any], path: Path) -> str:
    name = str(data.get("dataset", "")).lower()
    if "live" in name or "live" in path.name.lower():
        return "live"
    if "verified" in name or "verified" in path.name.lower():
        return "verified"
    raise ValueError(f"Cannot tell whether {path} is SWE-bench Verified or Live")


def image_name(dataset: str, instance_id: str, overrides: dict[str, str]) -> str:
    if instance_id in overrides:
        return overrides[instance_id]
    iid = instance_id.replace("__", "_1776_").lower()
    if dataset == "verified":
        return f"docker.io/swebench/sweb.eval.x86_64.{iid}:latest"
    return f"starryzhang/sweb.eval.x86_64.{iid}"


def load_instances(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        ds = dataset_key(data, p)
        for inst in data["selected"]:
            inst = dict(inst)
            inst["_dataset"] = ds
            out.append(inst)
    return out


def check_images(instances: list[dict], overrides: dict[str, str]) -> int:
    missing = []
    for inst in instances:
        img = image_name(inst["_dataset"], inst["instance_id"], overrides)
        have = subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode == 0
        if not have:
            print(f"pulling {img} ...", file=sys.stderr)
            have = subprocess.run(["docker", "pull", img], capture_output=True, text=True).returncode == 0
        print(f"{'OK     ' if have else 'MISSING'} {inst['instance_id']}  ->  {img}", file=sys.stderr)
        if not have:
            missing.append(inst["instance_id"])
    if missing:
        print(f"\n{len(missing)} image(s) missing. Find the right names and pass them with "
              f"--image-map images.json  ({{\"instance_id\": \"image\"}}).", file=sys.stderr)
        return 1
    print("\nAll images available.", file=sys.stderr)
    return 0


# ============================================================
# AGENT CONFIG (matches the earlier experiment)
# ============================================================

def build_config(args: argparse.Namespace) -> dict[str, Any]:
    from minisweagent.config import get_config_from_spec
    config = get_config_from_spec("swebench.yaml")  # default SWE-bench templates
    config["agent"]["step_limit"] = args.step_limit
    config["agent"]["cost_limit"] = args.cost_limit
    config["model"]["model_name"] = args.model
    config["model"]["model_class"] = args.model_class
    kwargs = dict(config["model"].get("model_kwargs", {}))
    if args.reasoning_effort:
        kwargs["reasoning"] = {"effort": args.reasoning_effort}
        kwargs["text"] = {"verbosity": "medium"}
    config["model"]["model_kwargs"] = kwargs
    env = config.setdefault("environment", {})
    env.update({"environment_class": "docker", "cwd": "/testbed", "timeout": 60,
                "interpreter": ["bash", "-c"], "pull_timeout": 600})
    return config


# ============================================================
# RUNNING
# ============================================================

_LOCK = threading.Lock()


def update_json(path: Path, update) -> None:
    with _LOCK:
        data = json.loads(path.read_text()) if path.exists() else None
        data = update(data)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(path)


def run_one(job: dict, config: dict, args: argparse.Namespace, overrides: dict[str, str]) -> str:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    inst, cond, rep = job["instance"], job["condition"], job["rep"]
    iid, ds = inst["instance_id"], inst["_dataset"]
    run_dir = args.output_dir / ds / cond / f"rep_{rep}"
    inst_dir = run_dir / iid
    inst_dir.mkdir(parents=True, exist_ok=True)
    prompt = job["prompt"]
    (inst_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    traj_path = inst_dir / f"{iid}.traj.json"

    start = time.time()
    agent = env = None
    exit_status, submission, extra = None, "", {}
    try:
        env_cfg = dict(config["environment"])
        env_cfg["image"] = image_name(ds, iid, overrides)
        env = get_environment(env_cfg)
        model = get_model(config=dict(config["model"]))
        agent = DefaultAgent(model, env, **config["agent"])
        info = agent.run(prompt)
        exit_status, submission = info.get("exit_status"), info.get("submission") or ""
    except Exception as e:
        exit_status, submission = type(e).__name__, ""
        extra = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            agent.save(traj_path, {"info": {"exit_status": exit_status, "submission": submission, **extra},
                                   "instance_id": iid, "condition": cond, "repetition": rep})
        if env is not None and hasattr(env, "cleanup"):
            env.cleanup()

    # Any exception (Docker, network, API) is an infrastructure error: no patch.diff is written,
    # so the run is retried on the next invocation. LimitsExceeded etc. are normal outcomes.
    infra_error = bool(extra)
    if not infra_error:
        # patch.diff marks the run as finished (also for empty patches / limits exceeded)
        (inst_dir / "patch.diff").write_text(submission, encoding="utf-8")
        update_json(run_dir / "preds.json", lambda d: {**(d or {}), iid: {
            "instance_id": iid, "model_name_or_path": f"{args.model}_{cond}_rep{rep}",
            "model_patch": submission}})

    entry = {
        "dataset": ds, "condition": cond, "repetition": rep, "instance_id": iid,
        "repo": inst.get("repo"), "base_commit": inst.get("base_commit"), "model": args.model,
        "image": image_name(ds, iid, overrides),
        "prompt_path": str((inst_dir / "prompt.txt").relative_to(args.output_dir)),
        "patch_path": str((inst_dir / "patch.diff").relative_to(args.output_dir)),
        "trajectory_path": str(traj_path.relative_to(args.output_dir)),
        "patch_nonempty": bool(submission.strip()), "patch_chars": len(submission),
        "elapsed_seconds": round(time.time() - start, 2),
        "instance_cost": agent.cost if agent else 0.0, "api_calls": agent.n_calls if agent else 0,
        "exit_status": exit_status, "infra_error": infra_error,
        "error": extra.get("exception_str", ""),
    }

    def add_entry(d):
        d = d or {"experiment": {}, "runs": []}
        d["runs"] = [r for r in d["runs"] if not (r["dataset"] == ds and r["condition"] == cond
                                                   and r["repetition"] == rep and r["instance_id"] == iid)]
        d["runs"].append(entry)
        return d
    update_json(args.output_dir / "run_manifest.json", add_entry)
    status = f"{exit_status}{' (infra error, will retry next run)' if infra_error else ''}"
    print(f"[done] {ds}/{cond}/rep_{rep}/{iid}: {status}, {entry['api_calls']} calls, "
          f"${entry['instance_cost']:.2f}, patch {entry['patch_chars']} chars", file=sys.stderr)
    return status


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path, help="*_with_goals.json files")
    p.add_argument("--conditions", nargs="+", default=["baseline", "goal"],
                   choices=["baseline", "goal", "procedure"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--instance-ids", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=1, help="Parallel runs (each starts a Docker container).")
    p.add_argument("--output-dir", type=Path, default=Path("patch_runs"))
    p.add_argument("--model", default="openai/gpt-5.2-2025-12-11")
    p.add_argument("--model-class", default="litellm_response")
    p.add_argument("--reasoning-effort", default="medium", help="'' to disable (for non-OpenAI models).")
    p.add_argument("--cost-limit", type=float, default=2.5)
    p.add_argument("--step-limit", type=int, default=250)
    p.add_argument("--image-map", type=Path, default=None, help="JSON {instance_id: docker image}")
    p.add_argument("--check-images", action="store_true", help="Pull/verify all Docker images and exit.")
    p.add_argument("--dry-run", action="store_true", help="Write prompt.txt files only; no model calls.")
    p.add_argument("--redo", action="store_true", help="Re-run runs that already have a patch.diff.")
    args = p.parse_args()

    overrides = json.loads(args.image_map.read_text()) if args.image_map else {}
    instances = load_instances(args.files)
    if args.instance_ids:
        wanted = set(args.instance_ids)
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            print(f"Unknown instance ids: {sorted(missing)}", file=sys.stderr)
            return 2
    if not instances:
        print("No instances selected.", file=sys.stderr)
        return 2

    if args.check_images:
        return check_images(instances, overrides)

    jobs = []
    for inst in instances:
        title, desc = issue_title_and_description(inst)
        for cond in args.conditions:
            goals = inst.get("extracted_goals") or []
            if cond == "goal" and not goals:
                print(f"ERROR: {inst['instance_id']} has no extracted_goals", file=sys.stderr)
                return 2
            prompt = build_prompt(cond, title, desc, goals)
            for rep in range(1, args.reps + 1):
                inst_dir = args.output_dir / inst["_dataset"] / cond / f"rep_{rep}" / inst["instance_id"]
                if args.dry_run:
                    inst_dir.mkdir(parents=True, exist_ok=True)
                    (inst_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                    continue
                if (inst_dir / "patch.diff").exists() and not args.redo:
                    continue
                jobs.append({"instance": inst, "condition": cond, "rep": rep, "prompt": prompt})

    if args.dry_run:
        print(f"Dry run: prompts written under {args.output_dir.resolve()}", file=sys.stderr)
        return 0
    if not jobs:
        print("Nothing to do: all selected runs already have a patch.diff (use --redo to re-run).",
              file=sys.stderr)
        return 0

    config = build_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def set_experiment(d):
        d = d or {"experiment": {}, "runs": []}
        d["experiment"] = {"model": args.model, "model_class": args.model_class,
                           "reasoning_effort": args.reasoning_effort, "cost_limit_per_run_usd": args.cost_limit,
                           "step_limit": args.step_limit, "repetitions": args.reps,
                           "conditions": args.conditions, "input_files": [f.name for f in args.files],
                           "note": "FAIL_TO_PASS and PASS_TO_PASS are not exposed during generation."}
        return d
    update_json(args.output_dir / "run_manifest.json", set_experiment)

    print(f"Running {len(jobs)} job(s) with {args.workers} worker(s)...", file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one, j, config, args, overrides) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Unexpected error: {e}", file=sys.stderr)
    print(f"Finished. Results in {args.output_dir.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
