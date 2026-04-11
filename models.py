from pydantic import BaseModel
from typing import Optional


class WatchdogAction(BaseModel):
    """
    Action the agent takes. Each action maps to a real data engineering response.

    Deployment actions:
      rollback_deployment   — immediately revert to last stable version
      verify_and_redeploy   — push fix only after retesting load script

    Diagnosis actions (free — build evidence before acting):
      inspect_logs          — read BitLog for this pipeline
      classify_layer        — determine: SQL/script layer or API/network layer?
      compare_runs          — diff two episodes to find what changed
      analyze_query         — run EXPLAIN / query plan to find SQL bottleneck

    Fix actions:
      flag_for_optimization  — mark for next sprint (data model / SQL debt)
      escalate_with_evidence — go to stakeholder/DBA with proof
      deprioritize_job       — kill lower-priority clashing task
      add_incremental_load   — replace full reload with incremental load script
      clean_data             — fix partial null spike after inspection

    Ignore (only valid for INFO):
      ignore
    """
    action: str
    pipeline_id: Optional[int] = None
    target_episode: Optional[str] = None   # for compare_runs
    reasoning: Optional[str] = None        # agent's explanation (bonus reward)


class WatchdogObservation(BaseModel):
    success: bool
    message: str
    # Active event
    event_type:     Optional[str]   = None
    event_severity: Optional[str]   = None
    event_category: Optional[str]   = None
    pipeline_id:    Optional[int]   = None
    pipeline_name:  Optional[str]   = None
    # Context
    recent_events:    Optional[str]   = None
    unresolved_count: Optional[int]   = None
    evidence_score:   Optional[float] = None   # 0.0-1.0: how much proof gathered
    log_summary:      Optional[str]   = None
    hint:             Optional[str]   = None   # shown on wrong action
    reward:           Optional[float] = None


class WatchdogState(BaseModel):
    episode_id:     str   = ""
    step_count:     int   = 0
    task_id:        str   = ""
    events_total:   int   = 0
    events_resolved: int  = 0
    events_missed:  int   = 0
    evidence_steps: int   = 0    # how many inspect/classify/analyze steps taken
    done:           bool  = False
    rolled_back:    bool  = False  # ← your addition: tracks rollback state at top level
    evidence_score: float = 0.0   # ← your addition: running evidence score in state
