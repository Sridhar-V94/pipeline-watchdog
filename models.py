from pydantic import BaseModel
from typing import Optional


class WatchdogAction(BaseModel):
    action: str
    pipeline_id: Optional[int] = None
    target_episode: Optional[str] = None
    reasoning: Optional[str] = None


class WatchdogObservation(BaseModel):
    # Standard OpenEnv fields
    done:    bool  = False          # episode complete flag (standard)
    reward:  float = 0.0            # step reward — clamped by server before return

    # Our fields
    success: bool  = True           # was the action accepted
    message: str   = ""

    # Active event
    event_type:     Optional[str]   = None
    event_severity: Optional[str]   = None
    event_category: Optional[str]   = None
    pipeline_id:    Optional[int]   = None
    pipeline_name:  Optional[str]   = None

    # Context
    recent_events:    Optional[str]   = None
    unresolved_count: Optional[int]   = None
    evidence_score:   Optional[float] = None
    log_summary:      Optional[str]   = None
    hint:             Optional[str]   = None


class WatchdogState(BaseModel):
    episode_id:      str   = ""
    step_count:      int   = 0
    task_id:         str   = ""
    events_total:    int   = 0
    events_resolved: int   = 0
    events_missed:   int   = 0
    evidence_steps:  int   = 0
    done:            bool  = False
    rolled_back:     bool  = False
    evidence_score:  float = 0.0
