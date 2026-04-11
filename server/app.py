"""
server/app.py — PipelineWatchdog OpenEnv server.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

try:
    from ..environment import PipelineWatchdogEnv
    from ..models import WatchdogAction
except ImportError:
    from environment import PipelineWatchdogEnv
    from models import WatchdogAction

app = FastAPI(title="PipelineWatchdog", version="2.0.0")

env = PipelineWatchdogEnv()


def clamp(v: float) -> float:
    """Clamp strictly to (0.01, 0.99) — validator requires score in (0, 1)."""
    return round(max(0.01, min(0.99, float(v))), 3)


def safe_obs(obs) -> dict:
    """
    Sanitise observation before returning over HTTP.

    The OpenEnv validator checks that every numeric field that could
    be interpreted as a 'score' is strictly in (0, 1).  We clamp
    reward here so negative penalties and zero-reward diagnostics
    never produce 0.0 or negative values in the response.

    We also inject done=env.state.done so the validator knows when
    the episode has ended (standard OpenEnv protocol).
    """
    d = obs.model_dump()

    # Clamp reward: 0.0 and negatives → 0.01, positives stay as-is
    raw = d.get("reward") or 0.0
    d["reward"] = clamp(raw) if raw > 0.0 else 0.01

    # Inject standard done flag from env state
    d["done"] = env.state.done

    # Clamp evidence_score if present (0.0 is valid but let's be safe)
    if d.get("evidence_score") is not None:
        d["evidence_score"] = clamp(d["evidence_score"]) if d["evidence_score"] > 0 else 0.01

    return d


# ── Request models ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: str
    pipeline_id: Optional[int] = None
    reasoning: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "pipeline-watchdog",
        "version": "2.0.0",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grade", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": "pipeline-watchdog"}


@app.post("/reset")
def reset(req: ResetRequest = None):
    task_id = (req.task_id if req else None) or "easy"
    obs = env.reset(task_id=task_id)
    return safe_obs(obs)


@app.post("/step")
def step(req: StepRequest):
    action = WatchdogAction(
        action=req.action,
        pipeline_id=req.pipeline_id,
        reasoning=req.reasoning,
    )
    obs = env.step(action)
    return safe_obs(obs)


@app.get("/state")
def state():
    return env.state.model_dump()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy",   "description": "Post-deployment version mismatch", "difficulty": "easy"},
            {"id": "medium", "description": "Rushed project performance issues", "difficulty": "medium"},
            {"id": "hard",   "description": "SQL performance + missing index",   "difficulty": "hard"},
        ]
    }


@app.get("/grade")
def grade():
    score = clamp(env.grade(trajectory=[]))
    return {
        "score":    score,
        "resolved": env.state.events_resolved,
        "total":    env.state.events_total,
        "missed":   env.state.events_missed,
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
