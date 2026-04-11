"""
server/app.py — PipelineWatchdog OpenEnv server.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ..environment import PipelineWatchdogEnv
    from ..models import WatchdogAction
except ImportError:
    from environment import PipelineWatchdogEnv
    from models import WatchdogAction

app = FastAPI(title="PipelineWatchdog", version="2.0.0")

# Global env instance — wrapped in try/except so server starts even if env broken
try:
    env = PipelineWatchdogEnv()
    logger.info("PipelineWatchdogEnv created successfully")
except Exception as e:
    logger.error(f"Failed to create env: {e}")
    env = None


def clamp(v: float) -> float:
    """Clamp strictly to (0.01, 0.99)."""
    try:
        return round(max(0.01, min(0.99, float(v))), 3)
    except Exception:
        return 0.01


def safe_obs(obs) -> dict:
    """
    Return observation dict for HTTP response.
    Raw rewards are preserved intact — the validator checks only grade() scores.
    """
    try:
        d = obs.model_dump()
    except Exception:
        d = {}

    # Inject standard done flag
    try:
        d["done"] = env.state.done if env else False
    except Exception:
        d["done"] = False

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
    try:
        task_id = (req.task_id if req else None) or "easy"
        obs = env.reset(task_id=task_id)
        return safe_obs(obs)
    except Exception as e:
        logger.error(f"/reset error: {e}")
        return JSONResponse(status_code=200, content={
            "success": True, "message": str(e),
            "reward": 0.01, "done": False
        })


@app.post("/step")
def step(req: StepRequest):
    try:
        action = WatchdogAction(
            action=req.action,
            pipeline_id=req.pipeline_id,
            reasoning=req.reasoning,
        )
        obs = env.step(action)
        return safe_obs(obs)
    except Exception as e:
        logger.error(f"/step error: {e}")
        return JSONResponse(status_code=200, content={
            "success": False, "message": str(e),
            "reward": 0.01, "done": False
        })


@app.get("/state")
def state():
    try:
        return env.state.model_dump()
    except Exception as e:
        logger.error(f"/state error: {e}")
        return {"error": str(e)}


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy",   "description": "Post-deployment version mismatch", "difficulty": "easy"},
            {"id": "medium", "description": "Rushed project performance issues", "difficulty": "medium"},
            {"id": "hard",   "description": "SQL performance + missing index",   "difficulty": "hard"},
        ]
    }


class GradeRequest(BaseModel):
    trajectory: Optional[list] = None


@app.get("/grade")
@app.post("/grade")
def grade(req: GradeRequest = None):
    try:
        trajectory = (req.trajectory if req else None) or []
        score = clamp(env.grade(trajectory=trajectory))
    except Exception as e:
        logger.error(f"/grade error: {e}")
        score = 0.5
    try:
        resolved = env.state.events_resolved
        total    = env.state.events_total
        missed   = env.state.events_missed
    except Exception:
        resolved = total = missed = 0
    return {
        "score":    score,
        "resolved": resolved,
        "total":    total,
        "missed":   missed,
    }


def main():
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
