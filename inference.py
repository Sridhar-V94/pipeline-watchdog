"""
inference.py — PipelineWatchdog Environment
============================================
MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM (default provided).
    MODEL_NAME     The model identifier to use for inference (default provided).
    HF_TOKEN       Your Hugging Face API token (mandatory, no default).

STDOUT FORMAT (exactly as required by OpenEnv validator):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rewards are raw step rewards in [-1, 1] — the validator does NOT check these.
Only the score from grade() is checked: must be strictly in (0, 1).
"""

import os
import json
from environment import PipelineWatchdogEnv
from models import WatchdogAction

# ── Credentials ───────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""

BENCHMARK = "pipeline-watchdog"
MAX_STEPS  = 40

# ── LLM client — optional, falls back to rule-based agent if unavailable ──
client = None
try:
    from openai import OpenAI
    if HF_TOKEN:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
except Exception:
    pass


def clamp_score(v):
    """Clamp grade() score strictly to (0.01, 0.99). Only used for scores."""
    try:
        return round(max(0.01, min(0.99, float(v))), 3)
    except Exception:
        return 0.5


# ── Rule-based fallback agent (no LLM required) ───────────────────────────
RULES = {
    "VERSION_MISMATCH":         "rollback_deployment",
    "RELOAD_FAILURE":           "rollback_deployment",
    "ROLLBACK_DONE":            "verify_and_redeploy",
    "REDEPLOY_SUCCESS":         "ignore",
    "SLOW_QUERY":               "analyze_query",
    "FULL_RELOAD_OVERLOAD":     "add_incremental_load",
    "MISSING_INDEX":            "inspect_logs",
    "LOAD_SCRIPT_INEFFICIENCY": "inspect_logs",
    "CONCURRENCY_PRESSURE":     "inspect_logs",
    "DESIGN_DEBT_DATA":         "flag_for_optimization",
    "MEMORY_PRESSURE":          "flag_for_optimization",
    "TASK_CLASH":               "deprioritize_job",
    "NULL_SPIKE_PARTIAL":       "inspect_logs",
}

def rule_based_action(obs):
    """Deterministic fallback — works without any LLM."""
    etype    = obs.event_type or ""
    ev_score = obs.evidence_score or 0.0
    pid      = obs.pipeline_id or 1

    if ev_score >= 0.5 and etype in (
        "MISSING_INDEX", "LOAD_SCRIPT_INEFFICIENCY", "CONCURRENCY_PRESSURE"
    ):
        return {"action": "escalate_with_evidence", "pipeline_id": pid}

    if etype == "NULL_SPIKE_PARTIAL" and ev_score >= 0.25:
        return {"action": "clean_data", "pipeline_id": pid}

    return {"action": RULES.get(etype, "inspect_logs"), "pipeline_id": pid}


# ── LLM-based agent ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior data engineering SRE. Monitor pipelines and respond to alerts.

RULES:
VERSION_MISMATCH/RELOAD_FAILURE → {"action": "rollback_deployment", "pipeline_id": <id>}
ROLLBACK_DONE                   → {"action": "verify_and_redeploy", "pipeline_id": <id>}
REDEPLOY_SUCCESS                → {"action": "ignore", "pipeline_id": <id>}
SLOW_QUERY                      → {"action": "analyze_query", "pipeline_id": <id>}
FULL_RELOAD_OVERLOAD            → {"action": "add_incremental_load", "pipeline_id": <id>}
MISSING_INDEX (no evidence yet) → {"action": "analyze_query", "pipeline_id": <id>}
MISSING_INDEX (evidence >= 0.5) → {"action": "escalate_with_evidence", "pipeline_id": <id>}
LOAD_SCRIPT_INEFFICIENCY        → inspect_logs first, then escalate_with_evidence
CONCURRENCY_PRESSURE            → inspect_logs first, then escalate_with_evidence
DESIGN_DEBT_DATA/MEMORY_PRESSURE → {"action": "flag_for_optimization", "pipeline_id": <id>}
TASK_CLASH                      → {"action": "deprioritize_job", "pipeline_id": <id>}
NULL_SPIKE_PARTIAL              → inspect_logs first, then clean_data
INFO severity                   → {"action": "ignore", "pipeline_id": <id>}

Respond ONLY with valid JSON. No markdown. No extra text."""


def llm_action(obs, history):
    user_msg = (
        f"Event: {obs.event_type} | Severity: {obs.event_severity} | "
        f"Layer: {obs.event_category} | Pipeline: {obs.pipeline_name} (id={obs.pipeline_id}) | "
        f"Evidence: {obs.evidence_score or 0.0:.2f} | Hint: {obs.hint or 'none'}\n"
        f"Action? JSON only."
    )
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-10:]
        + [{"role": "user", "content": user_msg}]
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME, max_tokens=80,
        messages=messages, temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def ask_agent(obs, history):
    """Always returns an action — falls back to rules if LLM fails."""
    if client is not None:
        try:
            return llm_action(obs, history), None
        except Exception as e:
            return rule_based_action(obs), str(e)
    return rule_based_action(obs), None


# ── Task runner ───────────────────────────────────────────────────────────

def run_task(env, task_id):
    rewards = []   # raw step rewards — NOT clamped, validator doesn't check these
    step    = 0
    success = False
    score   = 0.5  # safe default in (0,1)
    error   = None

    # [START] always first — even if everything else fails
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        obs     = env.reset(task_id=task_id)
        history = []

        while not env.state.done and step < MAX_STEPS:
            if obs.event_type is None and not env.state.done:
                break

            step += 1
            action_dict, error = ask_agent(obs, history)

            try:
                action = WatchdogAction(**action_dict)
                obs    = env.step(action)
            except Exception as e:
                error = str(e)
                obs   = env.step(WatchdogAction(
                    action="inspect_logs",
                    pipeline_id=action_dict.get("pipeline_id") or 1
                ))

            # Raw reward — kept as-is for learning signal integrity
            reward = obs.reward if obs.reward is not None else 0.0
            done   = env.state.done
            rewards.append(reward)

            print(
                f"[STEP] step={step} "
                f"action={action_dict.get('action')} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={'null' if error is None else error}",
                flush=True
            )

            history += [
                {"role": "assistant", "content": json.dumps(action_dict)},
                {"role": "user",      "content": obs.message},
            ]

        # Only the score from grade() is checked by validator — clamp only this
        score   = clamp_score(env.grade(trajectory=[]))
        success = score > 0.5

    except Exception as e:
        score   = 0.5  # strictly in (0,1), never 0.0 or 1.0
        success = False
        error   = str(e)

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step} "
            f"score={score:.3f} "
            f"rewards={rewards_str}",
            flush=True
        )

    return score


def main():
    env     = PipelineWatchdogEnv()
    results = {}

    for task_id in ["easy", "medium", "hard"]:
        results[task_id] = run_task(env, task_id)

    avg = clamp_score(sum(results.values()) / len(results))
    print(
        f"\nFinal scores: "
        f"easy={results['easy']:.3f} "
        f"medium={results['medium']:.3f} "
        f"hard={results['hard']:.3f} "
        f"avg={avg:.3f}",
        flush=True
    )


if __name__ == "__main__":
    main()
