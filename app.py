import streamlit as st
from environment import PipelineWatchdogEnv
from models import WatchdogAction

# ── Rule-based agent ──────────────────────────────────────────────────────
# Mirrors the real engineering rules in environment.py.
# Tracks state so it knows when evidence has been gathered.

class RuleBasedAgent:
    """
    Deterministic agent that follows the correct engineering rules.

    Valid actions (must match environment exactly):
      Diagnostic (free): inspect_logs, classify_layer, compare_runs, analyze_query
      Fix:               rollback_deployment, verify_and_redeploy,
                         flag_for_optimization, escalate_with_evidence,
                         deprioritize_job, add_incremental_load, clean_data, ignore
    """

    def __init__(self):
        self.diag_steps : dict[int, int]  = {}   # pipeline_id → diagnostic steps taken

    def reset(self):
        self.diag_steps = {}

    def choose(self, obs: object) -> str:
        etype    = obs.event_type
        sev      = obs.event_severity
        pid      = obs.pipeline_id or 1

        # Use internal step counter instead of obs.evidence_score
        # obs.evidence_score reflects state BEFORE this step's action
        # Internal counter is always accurate
        diag_steps = self.diag_steps.get(pid, 0)

        if etype is None:
            return "inspect_logs"

        # ── Deployment ────────────────────────────────────────────────────
        if etype in ("VERSION_MISMATCH", "RELOAD_FAILURE"):
            return "rollback_deployment"

        if etype == "ROLLBACK_DONE":
            return "verify_and_redeploy"

        if etype == "REDEPLOY_SUCCESS" or sev == "INFO":
            return "ignore"

        # ── SQL performance ───────────────────────────────────────────────
        if etype == "SLOW_QUERY":
            self._log_diag(pid)
            return "analyze_query"

        if etype == "FULL_RELOAD_OVERLOAD":
            return "add_incremental_load"

        if etype == "MISSING_INDEX":
            if diag_steps < 1:
                self._log_diag(pid)
                return "analyze_query"
            if diag_steps < 2:
                self._log_diag(pid)
                return "classify_layer"
            return "escalate_with_evidence"

        # ── Load script / frontend debt ───────────────────────────────────
        if etype in ("LOAD_SCRIPT_INEFFICIENCY", "DESIGN_DEBT_FRONTEND"):
            if diag_steps < 1:
                self._log_diag(pid)
                return "inspect_logs"
            if diag_steps < 2:
                self._log_diag(pid)
                return "classify_layer"
            return "escalate_with_evidence"

        if etype in ("SCOPE_OVERCOMMIT", "UNKNOWN_LIMITATION", "CONCURRENCY_PRESSURE"):
            if diag_steps < 1:
                self._log_diag(pid)
                return "inspect_logs"
            if diag_steps < 2:
                self._log_diag(pid)
                return "classify_layer"
            return "escalate_with_evidence"

        # ── Data model debt ───────────────────────────────────────────────
        if etype in ("DESIGN_DEBT_DATA", "MEMORY_PRESSURE"):
            return "flag_for_optimization"

        # ── Schedule clash ────────────────────────────────────────────────
        if etype == "TASK_CLASH":
            return "deprioritize_job"

        # ── Data quality ──────────────────────────────────────────────────
        if etype == "NULL_SPIKE_PARTIAL":
            if diag_steps < 1:
                self._log_diag(pid)
                return "inspect_logs"
            return "clean_data"

        # ── Default ───────────────────────────────────────────────────────
        if diag_steps < 2:
            self._log_diag(pid)
            return "inspect_logs"
        return "escalate_with_evidence"

    def _log_diag(self, pid: int):
        """Increment internal diagnostic step counter for this pipeline."""
        self.diag_steps[pid] = self.diag_steps.get(pid, 0) + 1


# ── Session state init ────────────────────────────────────────────────────
if "total_reward" not in st.session_state:
    st.session_state.total_reward = 0.0
if "env" not in st.session_state:
    st.session_state.env = PipelineWatchdogEnv()
if "agent" not in st.session_state:
    st.session_state.agent = RuleBasedAgent()

# ── UI ────────────────────────────────────────────────────────────────────
st.title("🔍 Pipeline Watchdog AI")
st.caption("An RL environment where an AI agent monitors and repairs data pipelines.")

task_id      = st.selectbox("Select Task", ["easy", "medium", "hard"])
default_steps = {"easy": 8, "medium": 15, "hard": 25}
steps        = st.slider("Max steps to simulate", min_value=1, max_value=40,
                         value=default_steps[task_id])

if st.button("▶️ Run Simulation"):
    env   = PipelineWatchdogEnv()          # fresh env every run — no stale BitLog
    agent = RuleBasedAgent()
    st.session_state.total_reward = 0.0

    obs = env.reset(task_id=task_id)

    st.subheader("🚀 Initial State")
    st.info(obs.message)
    if obs.log_summary:
        st.caption(f"📊 {obs.log_summary}")

    for step in range(steps):
        if env.state.done:
            break

        st.divider()

        action_str = agent.choose(obs)
        action     = WatchdogAction(action=action_str, pipeline_id=obs.pipeline_id or 1)
        obs        = env.step(action)

        reward = obs.reward or 0.0
        st.session_state.total_reward += reward
        st.session_state.total_reward  = round(st.session_state.total_reward, 2)

        # ── Step header ───────────────────────────────────────────────────
        st.subheader(f"Step {step + 1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Action:** `{action_str}`")
            st.markdown(f"**Pipeline:** `{obs.pipeline_name or 'N/A'}`")
            st.markdown(f"**Event:** `{obs.event_type or 'resolved'}`")
            st.markdown(f"**Severity:** `{obs.event_severity or 'N/A'}`")
        with col2:
            st.metric("Step Reward", f"{reward:+.2f}",
                      delta_color="normal" if reward >= 0 else "inverse")
            st.metric("Total Reward", f"{st.session_state.total_reward:+.2f}")
        with col3:
            if obs.evidence_score is not None:
                st.metric("Evidence Score", f"{obs.evidence_score:.2f}")
            if obs.unresolved_count is not None:
                st.metric("Unresolved", obs.unresolved_count)

        # ── Result message ────────────────────────────────────────────────
        if obs.success:
            st.success(obs.message)
        else:
            st.error(obs.message)
            if obs.hint:
                st.warning(f"💡 Hint: {obs.hint}")

        if obs.log_summary:
            st.caption(f"📊 {obs.log_summary}")

    # ── Final score ───────────────────────────────────────────────────────
    st.divider()
    final_score = env.grade(trajectory=[])
    resolved    = env.state.events_resolved
    total       = env.state.events_total
    missed      = env.state.events_missed

    st.subheader("🏆 Episode Complete")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Score",      f"{final_score:.2f}")
    c2.metric("Events Resolved",  f"{resolved}/{total}")
    c3.metric("Events Missed",    missed)
    c4.metric("Total Reward",     f"{st.session_state.total_reward:+.2f}")

    if final_score >= 0.8:
        st.balloons()
        st.success("🎉 Excellent! Agent handled the pipeline correctly.")
    elif final_score >= 0.5:
        st.info("👍 Good run. Some events were missed or mishandled.")
    else:
        st.warning("⚠️ Needs improvement. Review the hints above.")
