"""
environment.py — PipelineWatchdog v2
=====================================
Three tasks derived from real data engineering experience:

  EASY   — Post-deployment version mismatch.
            Rule: rollback immediately, stakeholders can't wait.
            Then fix the load script, retest, and redeploy.

  MEDIUM — Rushed project with unclear requirements and performance issues.
            Rule: inspect logs first, classify the layer (data vs load script),
            then either flag_for_optimization or escalate_with_evidence.
            restart_job is always wrong here.

  HARD   — Slow SQL query + missing index + full reload on huge table.
            Cascading issues: no index on join column, full table reload
            running every hour, concurrent users all hitting the same slow query.
            Agent must gather evidence across layers before escalating.
            No single quick fix — partial rewards for each evidence step.
"""

import uuid
import random

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    try:
        from openenv.core.env_server import Environment
    except ImportError:
        # Fallback: plain Python base class so file never crashes on import
        class Environment:
            def __init__(self): pass
            def reset(self, **kwargs): pass
            def step(self, action, **kwargs): pass
            def grade(self, trajectory=None, **kwargs): return 0.5
            def close(self): pass

try:
    from .models import WatchdogAction, WatchdogObservation, WatchdogState
    from .bitlog import BitLog
except ImportError:
    from models import WatchdogAction, WatchdogObservation, WatchdogState
    from bitlog import BitLog

# ── Pipeline registry ─────────────────────────────────────────────────────

PIPELINES = {
    1: "users_dashboard",
    2: "orders_report",
    3: "payments_analytics",
    4: "inventory_tracker",
    5: "executive_summary",
}

# ── Event definitions ─────────────────────────────────────────────────────

TASK_EVENTS = {

    # ── EASY: Version mismatch after bad deployment ───────────────────────
    # NOT shuffled — order is meaningful for this narrative sequence.
    "easy": [
        ("ERROR",    "SYSTEM",  1, "VERSION_MISMATCH"),
        ("CRITICAL", "SCRIPT",  1, "RELOAD_FAILURE"),
        ("INFO",     "SYSTEM",  1, "ROLLBACK_DONE"),
        ("WARN",     "SYSTEM",  1, "VERSION_MISMATCH"),
        ("INFO",     "SYSTEM",  1, "REDEPLOY_SUCCESS"),
    ],

    # ── MEDIUM: Rushed project, unclear data volume requirements ──────────
    "medium": [
        ("WARN",     "SCRIPT",  2, "LOAD_SCRIPT_INEFFICIENCY"),
        ("ERROR",    "SCRIPT",  2, "DESIGN_DEBT_DATA"),
        ("ERROR",    "SYSTEM",  3, "TASK_CLASH"),
        ("WARN",     "API",     2, "NULL_SPIKE_PARTIAL"),
        ("CRITICAL", "SCRIPT",  2, "FULL_RELOAD_OVERLOAD"),
        ("WARN",     "SYSTEM",  3, "MEMORY_PRESSURE"),
        ("ERROR",    "SCRIPT",  4, "LOAD_SCRIPT_INEFFICIENCY"),
        ("WARN",     "SCRIPT",  2, "DESIGN_DEBT_DATA"),
    ],

    # ── HARD: Slow SQL + missing index + full reload overload ─────────────
    "hard": [
        ("WARN",     "SCRIPT",  5, "SLOW_QUERY"),
        ("ERROR",    "SCRIPT",  5, "MISSING_INDEX"),
        ("CRITICAL", "SCRIPT",  5, "CONCURRENCY_PRESSURE"),
        ("ERROR",    "API",     5, "SLOW_QUERY"),
        ("WARN",     "SCRIPT",  5, "DESIGN_DEBT_DATA"),
        ("CRITICAL", "SCRIPT",  5, "FULL_RELOAD_OVERLOAD"),
        ("ERROR",    "NETWORK", 5, "CONCURRENCY_PRESSURE"),
        ("WARN",     "SCRIPT",  2, "LOAD_SCRIPT_INEFFICIENCY"),
        ("ERROR",    "SCRIPT",  2, "DESIGN_DEBT_DATA"),
        ("CRITICAL", "SCRIPT",  5, "MISSING_INDEX"),
    ],
}

# ── Correct action mapping ────────────────────────────────────────────────

CORRECT_ACTION = {
    "VERSION_MISMATCH":         "rollback_deployment",
    "RELOAD_FAILURE":           "rollback_deployment",
    "ROLLBACK_DONE":            "verify_and_redeploy",
    "REDEPLOY_SUCCESS":         "ignore",
    "DESIGN_DEBT_DATA":         "flag_for_optimization",
    "TASK_CLASH":               "deprioritize_job",
    "MEMORY_PRESSURE":          "flag_for_optimization",
    "NULL_SPIKE_PARTIAL":       "clean_data",
    "LOAD_SCRIPT_INEFFICIENCY": "escalate_with_evidence",
    "FULL_RELOAD_OVERLOAD":     "add_incremental_load",
    "SLOW_QUERY":               "analyze_query",
    "MISSING_INDEX":            "escalate_with_evidence",
    "CONCURRENCY_PRESSURE":     "escalate_with_evidence",
}

REQUIRES_EVIDENCE_FIRST = {
    "LOAD_SCRIPT_INEFFICIENCY",
    "MISSING_INDEX",
    "CONCURRENCY_PRESSURE",
    "NULL_SPIKE_PARTIAL",
}

WRONG_FOR_PERFORMANCE = {"restart_job", "rollback_deployment"}
EVIDENCE_THRESHOLD = 0.5
MAX_STEPS = {"easy": 15, "medium": 30, "hard": 50}


# ── Environment ───────────────────────────────────────────────────────────

class PipelineWatchdogEnv(Environment):
    """
    OpenEnv-compatible RL environment simulating a data pipeline SRE.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._state             = WatchdogState()
        self._bitlog            = BitLog()
        self._event_queue       : list[tuple] = []
        self._event_index       : int = 0
        self._current_event     : dict | None = None
        self._prev_episode      : str = ""
        self._evidence_gathered : dict[int, bool] = {}
        self._rollback_done     : dict[int, bool] = {}
        self._query_analyzed    : dict[int, bool] = {}

    # ── OpenEnv API ───────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy", **kwargs) -> WatchdogObservation:
        """Start a new episode for the given task difficulty."""
        task_id = task_id if task_id in TASK_EVENTS else "easy"

        self._prev_episode      = self._state.episode_id
        self._bitlog            = BitLog()
        self._evidence_gathered = {}
        self._rollback_done     = {}
        self._query_analyzed    = {}

        events = list(TASK_EVENTS[task_id])
        # Easy task has a strict narrative order — do NOT shuffle.
        if task_id != "easy":
            random.shuffle(events)

        self._event_queue   = events
        self._event_index   = 0
        self._current_event = None

        self._state = WatchdogState(
            episode_id   = str(uuid.uuid4())[:8],
            task_id      = task_id,
            events_total = len(self._event_queue),
        )

        return self._next_event_obs("🚀 Watchdog started.")

    def step(self, action: WatchdogAction, **kwargs) -> WatchdogObservation:
        """Process one agent action and return the next observation."""
        if self._state.done:
            return WatchdogObservation(
                success=False,
                message="Episode complete. Call reset() to start a new one.",
                reward=0.0,
            )

        self._state.step_count += 1
        act = action.action.lower().strip()

        max_steps = MAX_STEPS.get(self._state.task_id, 30)
        if self._state.step_count > max_steps:
            self._state.done = True
            return WatchdogObservation(
                success=False,
                message=f"⏰ Step limit ({max_steps}) reached. Episode ended.",
                log_summary=self._bitlog.summary(),
                reward=0.0,
            )

        is_free_diagnostic = act in ("inspect_logs", "classify_layer", "compare_runs")
        is_analyze_as_diagnostic = (
            act == "analyze_query"
            and (
                self._current_event is None
                or self._current_event.get("event_type") != "SLOW_QUERY"
            )
        )

        if is_free_diagnostic or is_analyze_as_diagnostic:
            return self._handle_diagnostic(action)

        obs = self._evaluate_fix(action)
        earned_reward = obs.reward or 0.0

        if self._current_event is None and not self._state.done:
            if self._event_index < len(self._event_queue):
                obs = self._next_event_obs(obs.message)
            else:
                self._state.done = True
                obs.message += f"\n\n✅ All {self._state.events_total} events handled."
                obs.log_summary = self._bitlog.summary()

        obs.reward = earned_reward
        return obs

    def grade(self, trajectory: list = None, **kwargs) -> float:
        """
        Compute final episode score in range (0.01, 0.99).

        Formula:
          base     = resolved / total
          penalty  = missed_events * 0.10
          evidence = min(evidence_steps * 0.05, 0.20)
          score    = clamp(base - penalty + evidence, 0.01, 0.99)
        """
        total    = self._state.events_total
        resolved = self._state.events_resolved
        missed   = self._state.events_missed
        evidence = min(self._state.evidence_steps * 0.05, 0.20)

        if total == 0:
            return 0.99

        base    = resolved / total
        penalty = missed * 0.10
        return round(max(0.01, min(0.99, base - penalty + evidence)), 3)

    def close(self) -> None:
        """Cleanup resources."""
        self._state         = WatchdogState()
        self._bitlog        = BitLog()
        self._current_event = None

    @property
    def state(self) -> WatchdogState:
        return self._state

    # ── Fix action evaluator ──────────────────────────────────────────────

    def _evaluate_fix(self, action: WatchdogAction) -> WatchdogObservation:
        if self._current_event is None:
            return WatchdogObservation(
                success=False,
                message="No active event. Use inspect_logs to check the pipeline.",
                reward=-0.05,
            )

        ev    = self._current_event
        etype = ev["event_type"]
        sev   = ev["severity"]
        cat   = ev["category"]
        pid   = ev["pipeline_id"]
        pname = PIPELINES.get(pid, f"pipeline_{pid}")
        act   = action.action.lower().strip()
        ev_score = self._bitlog.evidence_score(pid)

        # ── Rule 1: VERSION_MISMATCH / RELOAD_FAILURE → rollback first ────
        if etype in ("VERSION_MISMATCH", "RELOAD_FAILURE"):
            if act == "rollback_deployment":
                self._bitlog.append(sev, cat, pid, "ROLLBACK_INITIATED",
                                    self._state.step_count, self._state.episode_id)
                self._rollback_done[pid] = True
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Rolled back {pname} immediately.\n"
                        f"   Stakeholders have the previous stable version.\n"
                        f"   Now fix the load script and retest before redeploying."
                    ),
                    reward=0.40,
                )
            return self._wrong(
                message=(
                    f"❌ Wrong! You chose '{act}' on a {etype}.\n"
                    f"   Stakeholders are BLOCKED — the app is broken.\n"
                    f"   Rule: rollback IMMEDIATELY, fix the load script after."
                ),
                hint="rollback_deployment — always first, no exceptions.",
                reward=-0.40,
            )

        # ── Rule 2: ROLLBACK_DONE → verify_and_redeploy ───────────────────
        if etype == "ROLLBACK_DONE":
            if act == "ignore":
                return self._wrong(
                    message=(
                        f"❌ Don't ignore ROLLBACK_DONE on {pname}.\n"
                        f"   The pipeline is still on the old version.\n"
                        f"   Fix the load script, retest, then redeploy."
                    ),
                    hint="verify_and_redeploy — fix script, retest, then push.",
                    reward=-0.20,
                )
            if act == "verify_and_redeploy":
                if not self._rollback_done.get(pid, False):
                    return self._wrong(
                        message=(
                            f"❌ Can't redeploy — you haven't rolled back {pname} yet.\n"
                            f"   Rollback first, fix load script, retest, then redeploy."
                        ),
                        hint="rollback_deployment must come before verify_and_redeploy.",
                        reward=-0.20,
                    )
                self._bitlog.append("INFO", "SYSTEM", pid, "REDEPLOY_INITIATED",
                                    self._state.step_count, self._state.episode_id)
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Load script fixed, retested, and redeployed {pname}.\n"
                        f"   Corrected version is now live."
                    ),
                    reward=0.30,
                )
            return self._wrong(
                message=(
                    f"❌ '{act}' is not the right response to ROLLBACK_DONE on {pname}.\n"
                    f"   The rollback is done — now fix the load script and redeploy."
                ),
                hint="verify_and_redeploy — fix script, retest, then push.",
                reward=-0.15,
            )

        # ── Rule 2b: REDEPLOY_SUCCESS → ignore ───────────────────────────
        if etype == "REDEPLOY_SUCCESS":
            if act == "ignore":
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=f"✅ Correct. REDEPLOY_SUCCESS on {pname} — all clear.",
                    reward=0.05,
                )
            return self._wrong(
                message=(
                    f"❌ '{act}' on REDEPLOY_SUCCESS is unnecessary.\n"
                    f"   The pipeline redeployed successfully — just acknowledge it."
                ),
                hint="ignore — deployment succeeded, no action needed.",
                reward=-0.10,
            )

        # ── Rule 3: Performance events — restart_job / rollback always wrong
        if act in WRONG_FOR_PERFORMANCE and etype in REQUIRES_EVIDENCE_FIRST:
            return self._wrong(
                message=(
                    f"❌ '{act}' won't fix [{etype}] on {pname}.\n"
                    f"   This is a load script / SQL design problem.\n"
                    f"   Restarting a slow query just runs the same slow query again."
                ),
                hint="Use inspect_logs or analyze_query first.",
                reward=-0.20,
            )

        # ── Rule 4: escalate_with_evidence — must have evidence first ─────
        if act == "escalate_with_evidence":
            if ev_score < EVIDENCE_THRESHOLD:
                return self._wrong(
                    message=(
                        f"❌ Not enough evidence to escalate {pname}.\n"
                        f"   Evidence score: {ev_score:.2f} (need ≥ {EVIDENCE_THRESHOLD})\n"
                        f"   Going to stakeholders without proof = blame game.\n"
                        f"   Run inspect_logs, analyze_query, or compare_runs first."
                    ),
                    hint="Gather evidence first. Then escalate.",
                    reward=-0.25,
                )
            if etype in ("LOAD_SCRIPT_INEFFICIENCY", "MISSING_INDEX",
                         "CONCURRENCY_PRESSURE", "FULL_RELOAD_OVERLOAD"):
                self._bitlog.append("INFO", cat, pid, "ESCALATED",
                                    self._state.step_count, self._state.episode_id)
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Escalated {etype} on {pname} with evidence.\n"
                        f"   Evidence score: {ev_score:.2f}\n"
                        f"   Query plans, load script logs, and run comparisons attached.\n"
                        f"   Stakeholder meeting scheduled with proof — no blame game."
                    ),
                    evidence_score=ev_score,
                    reward=self._reward_for_severity(sev) + ev_score * 0.2,
                )

        # ── Rule 5: NULL_SPIKE_PARTIAL — inspect logs first ───────────────
        if etype == "NULL_SPIKE_PARTIAL" and act == "clean_data":
            if not self._evidence_gathered.get(pid, False):
                return self._wrong(
                    message=(
                        f"❌ Don't clean data blindly on {pname}.\n"
                        f"   NULL spike is PARTIAL — only in one region/segment.\n"
                        f"   Cleaning globally masks a source system bug.\n"
                        f"   Use inspect_logs first to identify which segment."
                    ),
                    hint="inspect_logs → identify the segment → then clean_data",
                    reward=-0.15,
                )
            self._bitlog.append("INFO", "API", pid, "DATA_CLEANED",
                                 self._state.step_count, self._state.episode_id)
            self._state.events_resolved += 1
            self._current_event = None
            return WatchdogObservation(
                success=True,
                message=(
                    f"✅ Correct! Cleaned partial NULL spike on {pname}.\n"
                    f"   Segment identified first — source system bug reported separately."
                ),
                reward=0.25,
            )

        # ── Rule 6: TASK_CLASH → deprioritize_job ────────────────────────
        if etype == "TASK_CLASH":
            if act == "deprioritize_job":
                self._bitlog.append("INFO", "SYSTEM", pid, "JOB_DEPRIORITIZED",
                                    self._state.step_count, self._state.episode_id)
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Deprioritized lower-priority job on {pname}.\n"
                        f"   Priority pipeline now has exclusive DB connection.\n"
                        f"   Staggered restart scheduled — clash resolved."
                    ),
                    reward=0.25,
                )
            if act in ("restart_job", "rollback_deployment"):
                return self._wrong(
                    message=(
                        f"❌ '{act}' on TASK_CLASH restarts into the same clash.\n"
                        f"   Both jobs will fight for the same DB connection again."
                    ),
                    hint="deprioritize_job — remove the clash first, then restart.",
                    reward=-0.20,
                )

        # ── Rule 7: SLOW_QUERY → analyze_query first ─────────────────────
        if etype == "SLOW_QUERY":
            if act == "analyze_query":
                self._bitlog.append("WARN", "SCRIPT", pid, "QUERY_ANALYZED",
                                    self._state.step_count, self._state.episode_id)
                self._state.evidence_steps += 1
                self._query_analyzed[pid] = True
                self._evidence_gathered[pid] = True
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Ran EXPLAIN / query plan on {pname}.\n"
                        f"   Finding: Seq Scan on orders (100M rows), cost=450000.\n"
                        f"   Finding: Hash Join on customer_id — no index, full scan.\n"
                        f"   Finding: Filter on updated_at — no index.\n"
                        f"   Next: escalate_with_evidence to DBA to add index,\n"
                        f"   OR use add_incremental_load to reduce rows per run."
                    ),
                    evidence_score=self._bitlog.evidence_score(pid),
                    reward=0.25,
                )
            return self._wrong(
                message=(
                    f"❌ Don't act on SLOW_QUERY without running analyze_query first.\n"
                    f"   You need the query execution plan to know WHERE it's slow."
                ),
                hint="analyze_query — run EXPLAIN first to find the bottleneck.",
                reward=-0.15,
            )

        # ── Rule 8: MISSING_INDEX → escalate with evidence ────────────────
        if etype == "MISSING_INDEX":
            if act == "escalate_with_evidence":
                if ev_score < EVIDENCE_THRESHOLD:
                    return self._wrong(
                        message=(
                            f"❌ Need more evidence before escalating MISSING_INDEX on {pname}.\n"
                            f"   Evidence score: {ev_score:.2f} — run analyze_query first."
                        ),
                        hint="analyze_query → then escalate_with_evidence to DBA.",
                        reward=-0.20,
                    )
                self._bitlog.append("INFO", "SCRIPT", pid, "ESCALATED",
                                    self._state.step_count, self._state.episode_id)
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Escalated MISSING_INDEX on {pname} to DBA with proof.\n"
                        f"   Query plan attached: full table scan confirmed.\n"
                        f"   DBA will add index on customer_id and updated_at."
                    ),
                    evidence_score=ev_score,
                    reward=self._reward_for_severity(sev),
                )

        # ── Rule 9: FULL_RELOAD_OVERLOAD → add_incremental_load ──────────
        if etype == "FULL_RELOAD_OVERLOAD":
            if act == "add_incremental_load":
                self._bitlog.append("INFO", "SCRIPT", pid, "INCREMENTAL_LOAD_ADDED",
                                    self._state.step_count, self._state.episode_id)
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Replaced full reload with incremental load on {pname}.\n"
                        f"   Load script now: WHERE updated_at > '$(vLastExecTime)'\n"
                        f"   Reload time: 45 min → ~3 min. Memory pressure resolved."
                    ),
                    reward=self._reward_for_severity(sev),
                )
            if act in ("restart_job", "rollback_deployment", "clean_data"):
                return self._wrong(
                    message=(
                        f"❌ '{act}' won't fix FULL_RELOAD_OVERLOAD on {pname}.\n"
                        f"   The load script reloads ALL rows every run — that IS the problem.\n"
                        f"   Restarting just runs the same 45-minute full scan again."
                    ),
                    hint="add_incremental_load — rewrite script to fetch only new/changed rows.",
                    reward=-0.20,
                )

        # ── Rule 10: LOAD_SCRIPT_INEFFICIENCY → escalate with evidence ────
        if etype == "LOAD_SCRIPT_INEFFICIENCY":
            if act == "escalate_with_evidence":
                if ev_score < EVIDENCE_THRESHOLD:
                    return self._wrong(
                        message=(
                            f"❌ Need more evidence before escalating load script issues.\n"
                            f"   Evidence score: {ev_score:.2f} — run inspect_logs first."
                        ),
                        hint="inspect_logs or analyze_query first, then escalate.",
                        reward=-0.20,
                    )
                self._bitlog.append("INFO", "SCRIPT", pid, "ESCALATED",
                                    self._state.step_count, self._state.episode_id)
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Escalated load script issues on {pname} with evidence.\n"
                        f"   Log comparison: script runtime grew 3x since last release.\n"
                        f"   Load script review + SQL refactor scheduled for next sprint."
                    ),
                    evidence_score=ev_score,
                    reward=self._reward_for_severity(sev),
                )

        # ── Rule 11: DESIGN_DEBT_DATA / MEMORY_PRESSURE → flag ───────────
        if etype in ("DESIGN_DEBT_DATA", "MEMORY_PRESSURE"):
            if act == "flag_for_optimization":
                self._bitlog.append("INFO", "SCRIPT", pid, "OPTIMIZATION_FLAGGED",
                                    self._state.step_count, self._state.episode_id)
                self._state.events_resolved += 1
                self._current_event = None
                return WatchdogObservation(
                    success=True,
                    message=(
                        f"✅ Correct! Flagged {pname} for SQL/data model optimization.\n"
                        f"   Incremental load + query restructure added to next sprint.\n"
                        f"   This needs proper requirements — can't be fixed at 2 AM."
                    ),
                    reward=self._reward_for_severity(sev),
                )
            if act in ("restart_job", "clean_data", "rollback_deployment"):
                return self._wrong(
                    message=(
                        f"❌ '{act}' won't fix design debt on {pname}.\n"
                        f"   The SQL / data model is inefficient.\n"
                        f"   Quick fixes don't solve poor architecture."
                    ),
                    hint="flag_for_optimization — schedule a proper fix with clear requirements.",
                    reward=-0.15,
                )

        # ── Rule 12: INFO events → ignore ────────────────────────────────
        if sev == "INFO" and act == "ignore":
            self._state.events_resolved += 1
            self._current_event = None
            return WatchdogObservation(
                success=True,
                message=f"✅ Correct. INFO event on {pname} safely ignored.",
                reward=0.05,
            )

        # ── Rule 13: Ignoring ERROR / CRITICAL is always wrong ────────────
        if sev in ("ERROR", "CRITICAL") and act == "ignore":
            self._state.events_missed += 1
            return self._wrong(
                message=f"❌ Ignored a {sev} [{etype}] on {pname}. This will recur.",
                hint=f"Correct action: {CORRECT_ACTION.get(etype, 'see rules')}",
                reward=-0.35,
            )

        # ── Default ───────────────────────────────────────────────────────
        correct = CORRECT_ACTION.get(etype, "see rules")
        return self._wrong(
            message=(
                f"❌ '{act}' is not the right response to [{etype}] "
                f"(severity={sev}) on {pname}.\n"
                f"   Think: is this a deployment issue, SQL issue, or load script issue?"
            ),
            hint=f"Expected: {correct}",
            reward=-0.10,
        )

    # ── Diagnostic actions ────────────────────────────────────────────────

    def _handle_diagnostic(self, action: WatchdogAction) -> WatchdogObservation:
        act   = action.action.lower().strip()
        pid   = action.pipeline_id or (
            self._current_event["pipeline_id"] if self._current_event else 1
        )
        pname = PIPELINES.get(pid, f"pipeline_{pid}")
        ev    = self._current_event or {}

        if act == "inspect_logs":
            events  = self._bitlog.get_by_pipeline(pid)[-8:]
            log_str = "\n".join(
                f"  [{e['severity']}][{e['category']}] {e['event_type']} t={e['timestamp']}"
                for e in events
            ) or "  No prior events on this pipeline."
            self._bitlog.append("INFO", "SYSTEM", pid, "LOGS_INSPECTED",
                                 self._state.step_count, self._state.episode_id)
            self._state.evidence_steps += 1
            self._evidence_gathered[pid] = True
            return WatchdogObservation(
                success=True,
                message=f"🔍 Inspect logs: {pname}\n{log_str}\n   Unresolved: {self._bitlog.count_unresolved(pid)}",
                event_type=ev.get("event_type"),
                event_severity=ev.get("severity"),
                event_category=ev.get("category"),
                pipeline_id=pid,
                pipeline_name=pname,
                evidence_score=self._bitlog.evidence_score(pid),
                log_summary=self._bitlog.summary(),
                reward=0.0,
            )

        if act == "classify_layer":
            script_errs = len(self._bitlog.get_by_category("SCRIPT"))
            api_errs    = len(self._bitlog.get_by_category("API"))
            net_errs    = len(self._bitlog.get_by_category("NETWORK"))
            verdict = "SCRIPT/SQL layer" if script_errs >= api_errs else "API/NETWORK layer"
            self._bitlog.append("INFO", "SYSTEM", pid, "LAYER_CLASSIFIED",
                                 self._state.step_count, self._state.episode_id)
            self._state.evidence_steps += 1
            self._evidence_gathered[pid] = True
            return WatchdogObservation(
                success=True,
                message=(
                    f"🔎 Layer classification for {pname}:\n"
                    f"   SCRIPT/SQL errors : {script_errs}\n"
                    f"   API errors        : {api_errs}\n"
                    f"   NETWORK errors    : {net_errs}\n"
                    f"   → Bottleneck      : {verdict}"
                ),
                event_type=ev.get("event_type"),
                event_severity=ev.get("severity"),
                event_category=ev.get("category"),
                pipeline_id=pid,
                pipeline_name=pname,
                evidence_score=self._bitlog.evidence_score(pid),
                reward=0.0,
            )

        if act == "compare_runs":
            if not self._prev_episode:
                self._bitlog.append("INFO", "SYSTEM", pid, "EVIDENCE_GATHERED",
                                     self._state.step_count, self._state.episode_id)
                self._state.evidence_steps += 1
                self._evidence_gathered[pid] = True
                return WatchdogObservation(
                    success=True,
                    message=(
                        "ℹ️  No previous episode to compare yet.\n"
                        "   Logged as evidence step — use inspect_logs or analyze_query\n"
                        "   to build further evidence before escalating."
                    ),
                    event_type=ev.get("event_type"),
                    event_severity=ev.get("severity"),
                    event_category=ev.get("category"),
                    pipeline_id=pid,
                    pipeline_name=pname,
                    evidence_score=self._bitlog.evidence_score(pid),
                    log_summary=self._bitlog.summary(),
                    reward=0.0,
                )
            diff     = self._bitlog.compare_episodes(self._prev_episode, self._state.episode_id)
            diff_str = "\n".join(
                f"   {cat}: {d['before']} → {d['after']} errors ({d['verdict']})"
                for cat, d in diff.items()
            ) or "   No significant change between runs."
            self._bitlog.append("INFO", "SYSTEM", pid, "EVIDENCE_GATHERED",
                                 self._state.step_count, self._state.episode_id)
            self._state.evidence_steps += 1
            self._evidence_gathered[pid] = True
            return WatchdogObservation(
                success=True,
                message=f"📊 Cross-run comparison:\n{diff_str}",
                event_type=ev.get("event_type"),
                event_severity=ev.get("severity"),
                event_category=ev.get("category"),
                pipeline_id=pid,
                pipeline_name=pname,
                evidence_score=self._bitlog.evidence_score(pid),
                reward=0.0,
            )

        if act == "analyze_query":
            self._bitlog.append("INFO", "SCRIPT", pid, "QUERY_ANALYZED",
                                 self._state.step_count, self._state.episode_id)
            self._state.evidence_steps += 1
            self._query_analyzed[pid] = True
            self._evidence_gathered[pid] = True
            return WatchdogObservation(
                success=True,
                message=(
                    f"🔧 Query EXPLAIN plan for {pname}:\n"
                    f"   Seq Scan on orders (100M rows)     cost=450000\n"
                    f"   Hash Join on customer_id            no index → full scan\n"
                    f"   Filter on updated_at                no index\n"
                    f"   Estimated runtime: ~45 min\n\n"
                    f"   Options:\n"
                    f"     1. add_incremental_load — fetch only WHERE updated_at > last_run\n"
                    f"     2. escalate_with_evidence — DBA adds index on customer_id + updated_at"
                ),
                event_type=ev.get("event_type"),
                event_severity=ev.get("severity"),
                event_category=ev.get("category"),
                pipeline_id=pid,
                pipeline_name=pname,
                evidence_score=self._bitlog.evidence_score(pid),
                reward=0.0,
            )

        return WatchdogObservation(
            success=False,
            message=f"Unknown diagnostic action: '{act}'",
            reward=0.0,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _next_event_obs(self, prefix: str) -> WatchdogObservation:
        if self._event_index >= len(self._event_queue):
            self._state.done = True
            return WatchdogObservation(
                success=True,
                message=prefix + "\n\n✅ All events processed.",
                log_summary=self._bitlog.summary(),
                reward=0.0,
            )

        sev, cat, pid, etype = self._event_queue[self._event_index]
        self._event_index += 1
        self._bitlog.append(sev, cat, pid, etype,
                             self._state.step_count, self._state.episode_id)
        self._current_event = {
            "severity": sev, "category": cat,
            "pipeline_id": pid, "event_type": etype,
        }

        recent = "\n".join(
            f"  [{e['severity']}][{e['category']}] {e['event_type']}"
            for e in self._bitlog.get_by_pipeline(pid)[-4:]
        ) or "  No prior events."

        return WatchdogObservation(
            success=True,
            message=(
                f"{prefix}\n\n"
                f"🚨 NEW EVENT\n"
                f"   Pipeline : {PIPELINES.get(pid)} (id={pid})\n"
                f"   Event    : {etype}\n"
                f"   Severity : {sev}\n"
                f"   Layer    : {cat}\n"
            ),
            event_type=etype,
            event_severity=sev,
            event_category=cat,
            pipeline_id=pid,
            pipeline_name=PIPELINES.get(pid),
            recent_events=recent,
            unresolved_count=self._bitlog.count_unresolved(pid),
            evidence_score=self._bitlog.evidence_score(pid),
            log_summary=self._bitlog.summary(),
            reward=0.0,
        )

    def _wrong(self, message: str, hint: str, reward: float) -> WatchdogObservation:
        ev  = self._current_event or {}
        pid = ev.get("pipeline_id")
        recent = "\n".join(
            f"  [{e['severity']}][{e['category']}] {e['event_type']}"
            for e in self._bitlog.get_by_pipeline(pid)[-4:]
        ) if pid is not None else ""

        return WatchdogObservation(
            success=False,
            message=message,
            hint=hint,
            event_type=ev.get("event_type"),
            event_severity=ev.get("severity"),
            event_category=ev.get("category"),
            pipeline_id=pid,
            pipeline_name=PIPELINES.get(pid) if pid is not None else None,
            recent_events=recent or None,
            unresolved_count=self._bitlog.count_unresolved(pid) if pid is not None else None,
            evidence_score=self._bitlog.evidence_score(pid) if pid is not None else None,
            log_summary=self._bitlog.summary(),
            reward=reward,
        )

    def _reward_for_severity(self, sev: str) -> float:
        return {"INFO": 0.05, "WARN": 0.20, "ERROR": 0.30, "CRITICAL": 0.40}.get(sev, 0.20)
