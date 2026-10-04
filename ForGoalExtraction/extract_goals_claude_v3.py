#!/usr/bin/env python3
"""
Extract high-level KAOS/GORE goals for each issue in a SWE-bench-Live JSON file.

Version 3 of the extraction prompts. v3 change: goal scope is defined by the
cause the issue identifies (not the location/trigger where it was observed),
and includes other occurrences the issue raises. Earlier changes from v1:
  * Turn 2 analyses the stakeholder CAPABILITY and the SCOPE of the issue's
    reasoning, instead of anchoring on the single component named in the issue.
  * Turn 3 builds an explicit why-chain (L0 symptom -> L1 expected behaviour ->
    L2 stakeholder purpose -> L3 system property) and extracts goals at L2/L3.
  * "Grounded" is defined so that generalising from the issue's example to the
    class its reasoning applies to is allowed (but not beyond).
  * No generic "no regression" goals and no Maintain/Avoid negation pairs.
  * KAOS templates corrected (the v1 Maintain template mixed two forms).
  * Few-shot examples use an unrelated domain and show what NOT to do.
  * Intermediate outputs (Turn 2 analysis, why-chain, scope, rationales) are
    stored in "goal_extraction_trace" for auditing.
  * A local quality check flags goals that restate the issue text, duplicate
    each other, or have a malformed KAOS formalisation.

Input (one or more files, processed independently):
    swe_verified_selected_20.json   (SWE-bench Verified)
    swe_live_selected_20.json       (SWE-bench-Live)
    Each file has a top-level "selected" list of instances with "instance_id"
    and "problem_statement". The issue title is the first line of
    "problem_statement" (an "issues[0].title" field is used if present).
    "tasks" or a bare top-level list are also accepted.

Output (one per input; the input files are never modified):
    swe_verified_selected_20_with_goals.json
    swe_live_selected_20_with_goals.json
    The output is the input file unchanged, plus two new keys on every instance
    and a "goal_extraction" provenance block at the top level.

Each instance receives (same shape as v1, so downstream patch generation is unchanged):
    "extracted_goals": [
        {"goal_type": "Achieve | Maintain | Avoid",
         "high_level_goal": "...",
         "complete_formalized_goal": "..."}
    ]
plus:
    "goal_extraction_trace": {...}       (analysis, why-chain, scope, quality flags)

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python extract_goals_claude_v3.py swe_verified_selected_20.json swe_live_selected_20.json
    # options: --limit N  --overwrite  --effort {low,medium,high}  --model MODEL
    #          --max-tokens N  --instance-ids ID [ID ...]

LLM: Claude via the Anthropic Messages API, with adaptive thinking, the
`effort` setting, and JSON-schema structured outputs (output_config.format).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import anthropic


DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_MAX_TOKENS = 16000
PROMPT_VERSION = "v3-claude-2026-10-02"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert Requirements Engineer applying Axel van Lamsweerde's
Goal-Oriented Requirements Engineering (GORE) and the KAOS methodology.

Your task is to recover the high-level stakeholder goal(s) behind a software
issue. In KAOS, higher-level goals are found by repeatedly asking WHY a
lower-level requirement is wanted. An issue report usually states a symptom
and an expected behaviour; your job is to climb above them to the purpose
they serve.

Grounding rule (applies to every turn):
- Use only the issue title, the issue description, and the prior analysis in
  this conversation. Do not invent APIs, components, stakeholders, constraints,
  or behaviours that the issue does not mention or clearly imply.
- A goal is GROUNDED if it follows from the issue's stated reason, expected
  behaviour, or explanation of what went wrong. It does NOT have to be limited
  to the specific example, component, input, or error message in the issue.
- When the issue's own reasoning applies to a wider class than its example
  (for instance, the same mechanism affects several kinds of data or several
  callers), generalise the goal to that class. Do not generalise beyond the
  point where the issue's reasoning stops applying, and do not add business
  goals the issue gives no basis for.

Write every field of your output in English, even when the issue is written
in another language.

A high-level goal states a capability, a desired system state, or a property
the stakeholder relies on, independently of any implementation. Code changes,
fixes, tests, refactorings, files, and internal mechanisms are never goals.
""".strip()


# ============================================================
# TURN 2: capability and scope analysis
# ============================================================

TURN_2_INSTRUCTION = """
Analyse the issue above. Do not propose or describe a fix.

Answer the following, citing the issue text where possible:

1) Stakeholder: Who is affected (e.g., application developer, operator,
   end user of a CLI)? Use only roles the issue supports.

2) Capability: What does the stakeholder rely on the system to provide that
   this issue shows is not being delivered? Describe it as something the
   stakeholder can observe or do, not as a code location.

3) Purpose: Why does the stakeholder need that capability? What would go
   wrong for them, beyond the immediate error, if it is not delivered?

4) Scope of the issue's reasoning: Separate three things and quote the
   issue text for each:
   a) Where it was observed: the location, component, input, or trigger
      under which the reporter saw the problem.
   b) The cause the issue identifies: the mechanism the issue says produces
      the problem (e.g. a coding pattern, a missing step, a rule that is not
      applied). Write "not stated" if the issue gives no cause.
   c) Other occurrences: does the issue say or suggest that the same cause
      may exist elsewhere (other locations, components, operators, data
      kinds, callers)? Quote it, or write "not raised".
   Then state the class the issue's reasoning covers. It is defined by the
   cause (b), not by where it was observed (a), unless the issue says the
   cause only applies there. It includes any other occurrences raised in (c).
   If the issue gives no cause and raises no other occurrences, the class is
   the observed case.

5) Distinct concerns: Does the issue express more than one genuinely
   different concern (e.g., a capability that is missing AND a separate
   existing property that must not be broken)? List them, or state that
   there is a single concern.
""".strip()


# ============================================================
# TURN 3: why-chain and high-level goal extraction
# ============================================================

TURN_3_INSTRUCTION = """
Extract the high-level stakeholder goal(s) using an explicit why-chain.

STEP A: Build the why-chain for the issue:
  L0 symptom: what the stakeholder observed (error, wrong output, failure).
  L1 expected behaviour: the behaviour the issue says should happen instead.
  L2 stakeholder purpose: why the stakeholder wants L1, i.e. what it lets
     them achieve or rely on.
  L3 system property (optional): the general property of the system that L2
     is an instance of, if the issue's reasoning supports it. Use "" if not
     supported.

STEP B: Record the scope:
  literal_scope: the specific location, example, or trigger where the issue
     observed the problem.
  goal_scope: the class the goal should cover, taken from the Turn 2 scope
     analysis. Define it by the cause the issue identifies, not by the
     location or trigger where the problem was observed, unless the issue
     says the cause is specific to that location or trigger. If the issue
     says or suggests that the same cause may occur elsewhere, goal_scope
     must include those other occurrences. If the issue states no cause and
     raises no other occurrences, goal_scope equals literal_scope.
  scope_justification: the issue text that supports goal_scope (quote the
     stated cause and any mention of other occurrences).

STEP C: Extract goals at level L2 or L3 (choose the highest level that is
still grounded). Rules:
  1) A goal at L0 or L1 is not acceptable. A goal is L1 if it only says the
     expected behaviour from the issue happens (or the error does not occur).
  2) Phrase each goal over goal_scope, not literal_scope, unless they are
     equal.
  3) Prefer ONE goal. Add another only if it describes a genuinely different
     state (from Turn 2, question 5).
  4) Never output a Maintain/Avoid pair where one is the negation of the
     other, and never an Avoid goal that is the negation of an Achieve goal.
     Keep only the one that is most natural.
  5) Do not add a generic "no regressions / existing behaviour preserved"
     goal. Use Maintain only when the issue identifies a specific property
     that must keep holding over time.
  6) Each goal must remain valid under a completely different implementation.
  7) Use observable, verifiable conditions. Do not use vague or comparative
     words ("better", "improved", "correctly", "properly", "faithfully",
     "user-friendly") unless the goal states what they mean observably.
  8) Component or API names may appear only when they identify the
     stakeholder-visible capability.

KAOS goal types:
  Achieve: a target condition must eventually become true.
  Maintain: a good condition must hold at all times (under some condition).
  Avoid: a bad condition must never hold (under some condition).

ILLUSTRATIVE EXAMPLE (different domain; do not copy its content)

Issue: "CSV export: rows whose 'notes' column contains a comma open in
spreadsheet tools with the row split into extra columns. Expected: the
comment stays in one column. The exporter writes field values without
quoting them."

  L0: rows with a comma in 'notes' are split into extra columns when opened.
  L1: the 'notes' value stays in one column.
  L2: users can open an exported file in other tools and get back the same
      table they exported.
  L3: an exported file represents the source table exactly for every cell
      value, including values containing the delimiter or quote characters.
  literal_scope: commas in the 'notes' column.
  goal_scope: any cell value containing characters that have special meaning
      in CSV (the issue's stated cause, missing quoting, applies to all of
      them and to every column).

  GOOD (L3, one goal):
    Achieve: "An exported CSV file, when read by a standard CSV reader, yields
    the same cell values in the same columns as the source table, for any cell
    content."
  BAD (L1, restates the expected behaviour):
    Achieve: "Notes containing commas stay in one column."
  BAD (negation pair, redundant):
    Maintain: "Exported columns stay aligned."  +
    Avoid: "Exported columns become misaligned."
  BAD (generic regression goal):
    Maintain: "Existing export behaviour is preserved."

Return JSON matching the provided schema.
""".strip()


# ============================================================
# STEP 4: KAOS formalisation
# ============================================================

STEP_4_INSTRUCTION = """
Formalise each goal extracted in Turn 3 using the KAOS template for its type.

Templates (square brackets around "if ... then" mean that part is optional):
  Achieve [GoalName]: [if CurrentCondition then] sooner-or-later TargetCondition
  Maintain [GoalName]: [if CurrentCondition then] always GoodCondition
  Avoid [GoalName]: [if CurrentCondition then] always not BadCondition

Rules:
- GoalName is a short UpperCamelCase name for the condition (letters and
  digits only), e.g. ExportedCsvPreservesCellValues.
- Replace every placeholder with a concrete condition. No placeholder words
  (CurrentCondition, TargetCondition, GoodCondition, BadCondition, GoalName)
  may remain.
- complete_formalized_goal must contain the whole goal: the type keyword, the
  bracketed GoalName, the colon, and the full condition.
- Keep the same goal type, the same number of goals, the same order, and the
  same meaning and scope as Turn 3. Do not narrow the goal back to the
  issue's specific example, location, or trigger while formalising; the
  condition must cover the whole goal_scope.
- Include an "if ... then" condition only when the issue supports it.
- No implementation mechanisms. Keep each goal to one sentence.
- high_level_goal is the Turn 3 high_level_goal text (you may lightly edit it
  for clarity, without changing its meaning or scope).

Example (from the illustrative CSV issue):
{
  "goal_type": "Achieve",
  "high_level_goal": "An exported CSV file, when read by a standard CSV reader, yields the same cell values in the same columns as the source table, for any cell content.",
  "complete_formalized_goal": "Achieve [ExportedCsvPreservesCellValues]: if a table is exported to CSV and the file is read by a standard CSV reader then sooner-or-later the reader yields the same cell values in the same columns as the source table, for any cell content."
}

Incorrect (missing type and name before the colon):
  "if a table is exported ... then sooner-or-later the reader yields ..."

Return JSON matching the provided schema.
""".strip()


# ============================================================
# SCHEMAS (Claude structured outputs; minItems is not supported,
# so "at least one goal" is checked in code instead)
# ============================================================

TURN_3_SCHEMA = {
    "type": "object",
    "properties": {
        "why_chain": {
            "type": "object",
            "properties": {
                "L0_symptom": {"type": "string"},
                "L1_expected_behaviour": {"type": "string"},
                "L2_stakeholder_purpose": {"type": "string"},
                "L3_system_property": {"type": "string"},
            },
            "required": [
                "L0_symptom",
                "L1_expected_behaviour",
                "L2_stakeholder_purpose",
                "L3_system_property",
            ],
            "additionalProperties": False,
        },
        "scope": {
            "type": "object",
            "properties": {
                "literal_scope": {"type": "string"},
                "goal_scope": {"type": "string"},
                "scope_justification": {"type": "string"},
            },
            "required": ["literal_scope", "goal_scope", "scope_justification"],
            "additionalProperties": False,
        },
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal_type": {"type": "string", "enum": ["Achieve", "Maintain", "Avoid"]},
                    "level": {"type": "string", "enum": ["L2", "L3"]},
                    "high_level_goal": {"type": "string"},
                    "rationale": {"type": "string"},
                    "issue_evidence": {"type": "string"},
                },
                "required": ["goal_type", "level", "high_level_goal", "rationale", "issue_evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["why_chain", "scope", "goals"],
    "additionalProperties": False,
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal_type": {"type": "string", "enum": ["Achieve", "Maintain", "Avoid"]},
                    "high_level_goal": {"type": "string"},
                    "complete_formalized_goal": {"type": "string"},
                },
                "required": ["goal_type", "high_level_goal", "complete_formalized_goal"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["goals"],
    "additionalProperties": False,
}


# ============================================================
# QUALITY CHECKS (local, no API calls)
# ============================================================

FORMAL_RE = re.compile(r"^(Achieve|Maintain|Avoid) \[([A-Za-z][A-Za-z0-9]*)\]: \S")
PLACEHOLDERS = ("CurrentCondition", "TargetCondition", "GoodCondition",
                "BadCondition", "SomeCondition", "GoalName")
KEYWORD = {"Achieve": "sooner-or-later", "Maintain": "always", "Avoid": "always not"}

STOPWORDS = set("""
a an the and or of to in on for with without by from as at is are be been being
it its this that these those when if then than any all each every can should
must not no do does did so such into via per same their them they which who
""".split())

RESTATEMENT_THRESHOLD = 0.6   # share of goal content words found in issue L0/L1 text
DUPLICATE_THRESHOLD = 0.6     # Jaccard similarity between two goals


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def check_formalization(goal: dict[str, str]) -> list[str]:
    problems = []
    formal = goal["complete_formalized_goal"].strip()
    match = FORMAL_RE.match(formal)
    if not match:
        problems.append("does not start with '<Type> [GoalName]: '")
    elif match.group(1) != goal["goal_type"]:
        problems.append(f"type keyword {match.group(1)} != goal_type {goal['goal_type']}")
    if KEYWORD[goal["goal_type"]] not in formal:
        problems.append(f"missing temporal keyword '{KEYWORD[goal['goal_type']]}'")
    for p in PLACEHOLDERS:
        if re.search(rf"\b{p}\b", formal):
            problems.append(f"unresolved placeholder {p}")
    return problems


def quality_flags(goals: list[dict[str, str]], why_chain: dict[str, str],
                  title: str) -> list[dict[str, Any]]:
    """Heuristic flags; they are recorded, not enforced."""
    low_level_text = content_words(" ".join([
        title, why_chain.get("L0_symptom", ""), why_chain.get("L1_expected_behaviour", "")
    ]))
    flags = []
    for i, g in enumerate(goals):
        words = content_words(g["high_level_goal"])
        overlap = len(words & low_level_text) / len(words) if words else 0.0
        if overlap >= RESTATEMENT_THRESHOLD:
            flags.append({"goal_index": i, "flag": "possible_restatement",
                          "overlap_with_title_L0_L1": round(overlap, 2)})
    for i in range(len(goals)):
        for j in range(i + 1, len(goals)):
            a = content_words(goals[i]["high_level_goal"])
            b = content_words(goals[j]["high_level_goal"])
            jac = len(a & b) / len(a | b) if a | b else 0.0
            if jac >= DUPLICATE_THRESHOLD:
                flags.append({"goal_index": [i, j], "flag": "possible_duplicate",
                              "jaccard": round(jac, 2)})
    return flags


# ============================================================
# API HELPERS
# ============================================================

def issue_context(title: str, description: str) -> str:
    # Many SWE-bench-Live problem statements already start with the title;
    # avoid printing it twice.
    if description.startswith(title):
        description = description[len(title):].lstrip()
    return f"ISSUE TITLE:\n{title}\n\nISSUE DESCRIPTION:\n{description}".strip()


class LLM:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, client: "anthropic.Anthropic", model: str, effort: str, max_tokens: int):
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens

    def create(self, user_content: str, schema: dict | None = None) -> str:
        output_config: dict[str, Any] = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config=output_config,
            messages=[{"role": "user", "content": user_content}],
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Response hit max_tokens; increase --max-tokens.")
        if response.stop_reason == "refusal":
            raise RuntimeError("Model refused the request.")
        # Thinking blocks are skipped; only the final text is used.
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise RuntimeError("The API response contained no text output.")
        return text


def call_with_retry(fn, *, attempts: int = 5):
    delay = 2
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except anthropic.BadRequestError:
            raise  # malformed request: retrying will not help
        except Exception as exc:  # rate limits, overload, network errors, bad JSON
            last_exc = exc
            if attempt == attempts:
                break
            print(f"  call failed ({type(exc).__name__}: {exc}); retrying in {delay}s",
                  file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    assert last_exc is not None
    raise last_exc


# ============================================================
# PIPELINE
# ============================================================

def analyze_issue(llm: LLM, title: str, description: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    context = issue_context(title, description)

    # Turn 2: capability and scope analysis (free text)
    turn2 = call_with_retry(lambda: llm.create(f"{context}\n\n{TURN_2_INSTRUCTION}"))

    # Turn 3: why-chain, scope, high-level goals
    def turn3_call() -> dict[str, Any]:
        data = json.loads(llm.create(
            f"{context}\n\nPRIOR ANALYSIS FROM TURN 2:\n{turn2}\n\n{TURN_3_INSTRUCTION}",
            TURN_3_SCHEMA))
        if not data["goals"]:
            raise ValueError("Turn 3 returned no goals.")
        return data
    turn3 = call_with_retry(turn3_call)
    turn3_goals = turn3["goals"]
    turn3_for_step4 = [{"goal_type": g["goal_type"], "level": g["level"],
                        "high_level_goal": g["high_level_goal"]} for g in turn3_goals]

    # Step 4: KAOS formalisation, with one corrective retry on structural problems
    base_step4 = (
        f"{context}\n\n"
        f"GOAL SCOPE FROM TURN 3:\n{json.dumps(turn3['scope'], indent=2, ensure_ascii=False)}\n\n"
        f"GOALS EXTRACTED IN TURN 3:\n{json.dumps(turn3_for_step4, indent=2, ensure_ascii=False)}\n\n"
        f"{STEP_4_INSTRUCTION}"
    )

    def step4_call(extra: str = "") -> list[dict[str, str]]:
        return json.loads(llm.create(base_step4 + extra, FINAL_SCHEMA))["goals"]

    final_goals = call_with_retry(step4_call)
    problems = structural_problems(final_goals, turn3_goals)
    if problems:
        print(f"  step 4 problems, retrying once: {problems}", file=sys.stderr)
        feedback = ("\n\nYOUR PREVIOUS ANSWER HAD THESE PROBLEMS; FIX THEM:\n- "
                    + "\n- ".join(problems))
        final_goals = call_with_retry(lambda: step4_call(feedback))
        problems = structural_problems(final_goals, turn3_goals)
        if problems:
            raise RuntimeError(f"Step 4 output still invalid: {problems}")

    trace = {
        "prompt_version": PROMPT_VERSION,
        "model": llm.model,
        "turn2_analysis": turn2,
        "why_chain": turn3["why_chain"],
        "scope": turn3["scope"],
        "turn3_goals": turn3_goals,
        "quality_flags": quality_flags(final_goals, turn3["why_chain"], title),
    }
    return final_goals, trace


def structural_problems(final_goals: list[dict[str, str]],
                        turn3_goals: list[dict[str, str]]) -> list[str]:
    problems = []
    if len(final_goals) != len(turn3_goals):
        problems.append(f"expected {len(turn3_goals)} goals, got {len(final_goals)}")
    for i, (f, t) in enumerate(zip(final_goals, turn3_goals)):
        if f["goal_type"] != t["goal_type"]:
            problems.append(f"goal {i}: type changed from {t['goal_type']} to {f['goal_type']}")
        problems.extend(f"goal {i}: {p}" for p in check_formalization(f))
    return problems


# ============================================================
# I/O
# ============================================================

def dedupe_repeated_text(text: str) -> str:
    """Some benchmark problem statements contain the whole issue twice; keep one copy."""
    text = text.strip()
    half = len(text) // 2
    for cut in range(max(1, half - 3), half + 4):
        first, second = text[:cut].strip(), text[cut:].strip()
        if first and first == second:
            return first
    return text


def get_issue_material(task: dict[str, Any]) -> tuple[str, str]:
    statement = dedupe_repeated_text(str(task.get("problem_statement") or ""))
    if not statement:
        raise ValueError("problem_statement is missing.")
    issues = task.get("issues") or []
    if len(issues) == 1 and str(issues[0].get("title") or "").strip():
        title = str(issues[0]["title"]).strip()
        description = statement
    else:
        # SWE-bench / SWE-bench-Live: the first line of problem_statement is the title.
        first_line, _, rest = statement.partition("\n")
        title, description = first_line.strip(), rest.strip()
    if not title:
        raise ValueError("Could not determine the issue title.")
    if not description:
        description = title  # some issues have only a title
    return title, description


def get_instances(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    for key in ("selected", "tasks", "instances"):
        if isinstance(data.get(key), list):
            return data[key]
    raise ValueError("Input JSON must contain a 'selected' (or 'tasks') list.")


def save_json(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def prompt_hash() -> str:
    joined = "\n---\n".join([SYSTEM_PROMPT, TURN_2_INSTRUCTION, TURN_3_INSTRUCTION,
                             STEP_4_INSTRUCTION, json.dumps(TURN_3_SCHEMA),
                             json.dumps(FINAL_SCHEMA)])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def process_file(llm: LLM, input_path: Path, output_path: Path, args: argparse.Namespace) -> int:
    # Resume from an existing output file so earlier work is kept.
    source = output_path if output_path.exists() else input_path
    print(f"\n=== {input_path.name} -> {output_path.name}"
          + (" (resuming)" if source == output_path else ""), file=sys.stderr)
    data = json.loads(source.read_text(encoding="utf-8"))
    instances = get_instances(data)

    if isinstance(data, dict):
        data["goal_extraction"] = {
            "model": args.model,
            "provider": "anthropic",
            "thinking": "adaptive",
            "reasoning_effort": args.effort,
            "max_tokens": args.max_tokens,
            "method": "GORE/KAOS three-stage extraction with why-chain, cause-based scope (v3)",
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256_16": prompt_hash(),
            "input_file": input_path.name,
            "issue_text_source": "problem_statement only (title = first line)",
            "stored_key": "extracted_goals",
            "trace_key": "goal_extraction_trace",
        }

    processed = flagged = 0
    wanted = set(args.instance_ids) if args.instance_ids else None
    total = len(instances)

    for index, task in enumerate(instances, start=1):
        if args.limit is not None and processed >= args.limit:
            break
        instance_id = task.get("instance_id", f"instance-{index}")
        if wanted is not None and instance_id not in wanted:
            continue
        done = (task.get("goal_extraction_trace") or {}).get("prompt_version") == PROMPT_VERSION
        if done and not args.overwrite:
            print(f"[{index}/{total}] {instance_id}: already done; skipping.", file=sys.stderr)
            continue
        try:
            title, description = get_issue_material(task)
            print(f"[{index}/{total}] {instance_id}: extracting goals...", file=sys.stderr)
            goals, trace = analyze_issue(llm, title, description)
            trace["issue_title_used"] = title
            task["extracted_goals"] = goals
            task["goal_extraction_trace"] = trace
            processed += 1
            flagged += bool(trace["quality_flags"])
            save_json(output_path, data)
            levels = ",".join(g["level"] for g in trace["turn3_goals"])
            print(f"  -> {len(goals)} goal(s) [{levels}]"
                  + (f"  FLAGS: {trace['quality_flags']}" if trace["quality_flags"] else ""),
                  file=sys.stderr)
        except Exception as exc:
            print(f"ERROR on {instance_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            save_json(output_path, data)
            return 1

    save_json(output_path, data)
    missing = [t.get("instance_id") for t in instances if "extracted_goals" not in t]
    print(f"Done {input_path.name}: processed {processed}, flagged {flagged}, "
          f"still without goals {len(missing)}. Output: {output_path.resolve()}", file=sys.stderr)
    return 0


def default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_with_goals.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+",
                        help="One or more input JSON files (e.g. swe_verified_selected_20.json).")
    parser.add_argument("--output", default=None,
                        help="Output path (only when a single input is given). "
                             "Default: <input>_with_goals.json next to each input.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract goals for instances that are already done.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N instances per file (for a test run).")
    parser.add_argument("--instance-ids", nargs="*", default=None,
                        help="Only process these instance_ids.")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"],
                        help="Claude effort level (controls adaptive thinking depth).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Claude model ID, e.g. claude-opus-5-5 or claude-sonnet-5-5.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2
    if args.output and len(args.inputs) > 1:
        print("--output can only be used with a single input file.", file=sys.stderr)
        return 2

    inputs = [Path(p) for p in args.inputs]
    for p in inputs:
        if not p.exists():
            print(f"Input file not found: {p}", file=sys.stderr)
            return 2

    llm = LLM(anthropic.Anthropic(), args.model, args.effort, args.max_tokens)
    for input_path in inputs:
        output_path = Path(args.output) if args.output else default_output(input_path)
        if output_path.resolve() == input_path.resolve():
            print("Output path must differ from the input path.", file=sys.stderr)
            return 2
        rc = process_file(llm, input_path, output_path, args)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
