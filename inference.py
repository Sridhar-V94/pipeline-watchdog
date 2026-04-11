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
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import json
from openai import OpenAI
from environment import PipelineWatchdogEnv
from models import WatchdogAction

# ── Credentials ───────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "pipeline-watchdog"
MAX_STEPS = 40

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior data engineering SRE. Monitor pipelines and respond to alerts.

RULES:
VERSION_MISMATCH/RELOAD_FAILURE → {"action": "rollback_deployment", "pipeline_id": <id>}
ROLLBACK_DONE                   → {"action": "verify_and_redeploy", "pipeline_id": <id>}
REDEPLOY_SUCCESS                → {"action": "ignore", "pipeline_id": <id>}
SLOW_QUERY                      → {"action": "analyze_query", "pipeline_id": <id>}
FULL_RELOAD_OVERLOAD            → {"action": "add_incremental_load", "pipeline_id": <id>}
MISSING_INDEX (no evidence yet) → {"action": "analyze_query", "pipeline_id": <id>}
MISSING_INDEX (evidence >= 0.5) → {"action": "escalate_with_evidence", "pipeline_id": <id>}
LOAD_SCRIPT_INEFFICIENCY        → inspect_logs first, then classify_layer, then escalate_with_evidence
CONCURRENCY_PRESSURE            → inspect_logs first, then classify_layer, then escalate_with_evidence
DESIGN_DEBT_DATA/MEMORY_PRESSURE → {"action": "flag_for_optimization", "pipeline_id": <id>}
TASK_CLASH                      → {"action": "deprioritize_job", "pipeline_id": <id>}
NULL_SPIKE_PARTIAL              → inspect_logs first, then clean_data
INFO severity                   → {"action": "ignore", "pipeline_id": <id>}

Actions: rollback_deployment, verify_and_redeploy, inspect_logs, classify_layer,
compare_runs, analyze_query, flag_for_optimization, escalate_with_evidence,
deprioritize_job, add_incremental_load, clean_data, ignore

Respond ONLY with valid JSON. No markdown. No extra text."""


def ask_agent(obs, history: list) -> dict:
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
        model=MODEL_NAME,
        max_tokens=80,
        messages=messages,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_task(env: PipelineWatchdogEnv, task_id: str) -> float:
    obs     = env.reset(task_id=task_id)
    history = []
    rewards = []
    step    = 0
    success = False
    error   = None
    score   = 0.0

    # ── [START] — exactly one per episode ────────────────────────────────
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    try:
        while not env.state.done and step < MAX_STEPS:
            if obs.event_type is None and not env.state.done:
                break

            step += 1

            try:
                action_dict = ask_agent(obs, history)
                error = None
            except Exception as e:
                action_dict = {"action": "inspect_logs", "pipeline_id": obs.pipeline_id or 1}
                error = str(e)

            action = WatchdogAction(**action_dict)
            obs    = env.step(action)
            reward = obs.reward or 0.0
            done   = env.state.done
            rewards.append(reward)

            # ── [STEP] — one per step, immediately after env.step() ───────
            print(
                f"[STEP] step={step} "
                f"action={action_dict.get('action')} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={'null' if error is None else error}"
            )

            history += [
                {"role": "assistant", "content": json.dumps(action_dict)},
                {"role": "user",      "content": obs.message},
            ]

        score   = env.grade(trajectory=[])
        success = score >= 0.5

    except Exception as e:
        score   = 0.0
        success = False
        error   = str(e)

    finally:
        # ── [END] — always emitted even on exception ──────────────────────
        # Exact format required: success= steps= rewards= (NO score= field)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step} "
            f"rewards={rewards_str}"
        )

    return score


def main():
    env     = PipelineWatchdogEnv()
    results = {}

    for task_id in ["easy", "medium", "hard"]:
        results[task_id] = run_task(env, task_id)

    avg = sum(results.values()) / len(results)
    print(
        f"\nFinal scores: "
        f"easy={results['easy']:.2f} "
        f"medium={results['medium']:.2f} "
        f"hard={results['hard']:.2f} "
        f"avg={avg:.2f}"
    )


if __name__ == "__main__":
    main()
