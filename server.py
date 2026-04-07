"""
server.py — OpenEnv FastAPI server for PipelineWatchdog.

Exposes the standard OpenEnv HTTP endpoints:
  POST /reset  → start new episode
  POST /step   → take an action
  GET  /state  → get current state
  GET  /health → health check (required by HF Spaces validator)
  GET  /tasks  → list available tasks
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from environment import PipelineWatchdogEnv
from models import WatchdogAction

app = FastAPI(title="PipelineWatchdog", version="2.0.0")

# ── Single shared environment instance ───────────────────────────────────
env = PipelineWatchdogEnv()


# ── Request models ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class StepRequest(BaseModel):
    action: str
    pipeline_id: Optional[int] = None
    reasoning: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────

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


@app.post("/grade")
def grade():
    score = env.grade(trajectory=[])
    return {
        "score": score,
        "resolved": env.state.events_resolved,
        "total": env.state.events_total,
        "missed": env.state.events_missed,
    }
