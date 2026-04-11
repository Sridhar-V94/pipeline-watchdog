"""
server/app.py — PipelineWatchdog OpenEnv server.

Hand-rolled FastAPI server exposing standard OpenEnv endpoints:
  GET  /health  → health check
  GET  /        → info
  POST /reset   → start new episode
  POST /step    → take an action
  GET  /state   → current state
  GET  /tasks   → list tasks
  GET  /grade   → score current episode
"""

import os
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
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    action = WatchdogAction(
        action=req.action,
        pipeline_id=req.pipeline_id,
        reasoning=req.reasoning,
    )
    obs = env.step(action)
    return obs.model_dump()


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
    score = round(max(0.01, min(0.99, env.grade(trajectory=[]))), 3)
    return {
        "score": score,
        "resolved": env.state.events_resolved,
        "total": env.state.events_total,
        "missed": env.state.events_missed,
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
