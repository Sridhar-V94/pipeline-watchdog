"""
bitlog.py — Bit-packed log storage v2

Each log entry is ONE 32-bit integer:

 31    28 27   24 23    18 17      10 9          0
 ┌───────┬───────┬────────┬──────────┬────────────┐
 │  sev  │  cat  │ pipe_id│ evt_type │  timestamp │
 │ 4 bits│ 4 bits│ 6 bits │  8 bits  │  10 bits   │
 └───────┴───────┴────────┴──────────┴────────────┘

Categories (new in v2): SCRIPT, NETWORK, API, RENDER, SYSTEM
This enables cross-layer blame analysis — e.g. "was it the script or the API?"

HashMap indexes:
  pipeline_id  → positions   O(1)
  severity     → positions   O(1)
  category     → positions   O(1)  ← new: enables layer-by-layer comparison
  episode_id   → positions   O(1)  ← new: enables cross-run comparison
"""

from collections import defaultdict

# ── Severity ──────────────────────────────────────────────────────────────
SEVERITY     = {"INFO": 0, "WARN": 1, "ERROR": 2, "CRITICAL": 3}
SEVERITY_INV = {v: k for k, v in SEVERITY.items()}

# ── Log categories (which layer of the stack) ─────────────────────────────
CATEGORY     = {"SYSTEM": 0, "SCRIPT": 1, "NETWORK": 2, "API": 3, "RENDER": 4}
CATEGORY_INV = {v: k for k, v in CATEGORY.items()}

# ── Event types ───────────────────────────────────────────────────────────
EVENT_TYPE = {
    # Deployment events
    "VERSION_MISMATCH":       0,
    "RELOAD_FAILURE":         1,
    "DEPLOY_SUCCESS":         2,
    "ROLLBACK_DONE":          3,
    "REDEPLOY_SUCCESS":       4,

    # Performance — data layer
    "DESIGN_DEBT_DATA":       10,
    "TASK_CLASH":             11,
    "MEMORY_PRESSURE":        12,
    "NULL_SPIKE_PARTIAL":     13,

    # Performance — frontend layer
    "DESIGN_DEBT_FRONTEND":   20,
    "SCOPE_OVERCOMMIT":       21,
    "HYPERCUBE_OVERLOAD":     22,
    "UNKNOWN_LIMITATION":     23,
    "CONCURRENCY_PRESSURE":   24,

    # Agent resolution events (logged when agent acts)
    "ROLLBACK_INITIATED":     50,
    "REDEPLOY_INITIATED":     51,
    "LOGS_INSPECTED":         52,
    "LAYER_CLASSIFIED":       53,
    "OPTIMIZATION_FLAGGED":   54,
    "EVIDENCE_GATHERED":      55,
    "ESCALATED":              56,
    "JOB_DEPRIORITIZED":      57,
    "QUERY_ANALYZED":         58,   # analyze_query diagnostic/fix
    "INCREMENTAL_LOAD_ADDED": 59,   # add_incremental_load fix
    "DATA_CLEANED":           60,
    "IGNORED":                61,
}
EVENT_TYPE_INV = {v: k for k, v in EVENT_TYPE.items()}

# ── Bit layout ────────────────────────────────────────────────────────────
_SEV_BITS   = 4;  _SEV_SHIFT   = 28;  _SEV_MASK   = (1 << _SEV_BITS)  - 1
_CAT_BITS   = 4;  _CAT_SHIFT   = 24;  _CAT_MASK   = (1 << _CAT_BITS)  - 1
_PIPE_BITS  = 6;  _PIPE_SHIFT  = 18;  _PIPE_MASK  = (1 << _PIPE_BITS) - 1
_ETYPE_BITS = 8;  _ETYPE_SHIFT = 10;  _ETYPE_MASK = (1 << _ETYPE_BITS)- 1
_TS_BITS    = 10; _TS_SHIFT    = 0;   _TS_MASK    = (1 << _TS_BITS)   - 1


def encode(severity: str, category: str, pipeline_id: int,
           event_type: str, timestamp: int) -> int:
    return (
        (SEVERITY.get(severity, 0)    & _SEV_MASK)   << _SEV_SHIFT  |
        (CATEGORY.get(category, 0)    & _CAT_MASK)   << _CAT_SHIFT  |
        (pipeline_id                  & _PIPE_MASK)  << _PIPE_SHIFT |
        (EVENT_TYPE.get(event_type,0) & _ETYPE_MASK) << _ETYPE_SHIFT|
        (timestamp                    & _TS_MASK)    << _TS_SHIFT
    )


def decode(packed: int) -> dict:
    sev   = (packed >> _SEV_SHIFT)   & _SEV_MASK
    cat   = (packed >> _CAT_SHIFT)   & _CAT_MASK
    pipe  = (packed >> _PIPE_SHIFT)  & _PIPE_MASK
    etype = (packed >> _ETYPE_SHIFT) & _ETYPE_MASK
    ts    = (packed >> _TS_SHIFT)    & _TS_MASK
    return {
        "severity":    SEVERITY_INV.get(sev,   f"SEV_{sev}"),
        "category":    CATEGORY_INV.get(cat,   f"CAT_{cat}"),
        "pipeline_id": pipe,
        "event_type":  EVENT_TYPE_INV.get(etype, f"EVT_{etype}"),
        "timestamp":   ts,
    }


# ── BitLog ────────────────────────────────────────────────────────────────

class BitLog:
    """
    Compact event log: list of 32-bit ints + 4 HashMap indexes.

    Indexes (all O(1) lookup):
      _idx_pipeline  : pipeline_id → [positions]
      _idx_severity  : severity    → [positions]
      _idx_category  : category    → [positions]   ← blame analysis
      _idx_episode   : episode_id  → [positions]   ← cross-run comparison
    """

    def __init__(self):
        self._log: list[int] = []
        self._idx_pipeline: defaultdict[int, list[int]]  = defaultdict(list)
        self._idx_severity: defaultdict[str, list[int]]  = defaultdict(list)
        self._idx_category: defaultdict[str, list[int]]  = defaultdict(list)
        self._idx_episode:  defaultdict[str, list[int]]  = defaultdict(list)

    def append(self, severity: str, category: str, pipeline_id: int,
               event_type: str, timestamp: int, episode_id: str = "") -> int:
        packed = encode(severity, category, pipeline_id, event_type, timestamp)
        pos = len(self._log)
        self._log.append(packed)
        self._idx_pipeline[pipeline_id].append(pos)
        self._idx_severity[severity].append(pos)
        self._idx_category[category].append(pos)
        self._idx_episode[episode_id].append(pos)
        return packed

    # ── Lookup methods (all O(1) index access) ────────────────────────────

    def get_by_pipeline(self, pid: int)    -> list[dict]:
        return [decode(self._log[p]) for p in self._idx_pipeline.get(pid, [])]

    def get_by_severity(self, sev: str)    -> list[dict]:
        return [decode(self._log[p]) for p in self._idx_severity.get(sev, [])]

    def get_by_category(self, cat: str)    -> list[dict]:
        return [decode(self._log[p]) for p in self._idx_category.get(cat, [])]

    def get_by_episode(self, eid: str)     -> list[dict]:
        return [decode(self._log[p]) for p in self._idx_episode.get(eid, [])]

    def get_recent(self, n: int = 5)       -> list[dict]:
        return [decode(e) for e in self._log[-n:]]

    # ── Cross-run comparison (blame analysis) ─────────────────────────────

    def compare_episodes(self, ep1: str, ep2: str) -> dict:
        """
        Compare two episodes side by side.
        Returns which categories had MORE errors in ep2 vs ep1.
        This is the 'proof' the agent gathers before escalating.

        E.g.: RENDER errors went from 1 → 5 between runs =
              frontend is the problem, not the data layer.
        """
        def category_counts(episode_id: str) -> dict:
            counts = defaultdict(int)
            for e in self.get_by_episode(episode_id):
                if e["severity"] in ("ERROR", "CRITICAL"):
                    counts[e["category"]] += 1
            return dict(counts)

        c1 = category_counts(ep1)
        c2 = category_counts(ep2)
        all_cats = set(c1) | set(c2)

        diff = {}
        for cat in all_cats:
            before = c1.get(cat, 0)
            after  = c2.get(cat, 0)
            if after != before:
                diff[cat] = {"before": before, "after": after,
                             "delta": after - before,
                             "verdict": "WORSE" if after > before else "IMPROVED"}
        return diff

    def evidence_score(self, pipeline_id: int) -> float:
        """
        How much evidence has the agent gathered for this pipeline?
        Score 0.25 per diagnostic step taken, max 1.0.
        Used to gate escalation: agent must gather evidence before escalating.
        """
        events = self.get_by_pipeline(pipeline_id)
        resolution_types = {
            "LOGS_INSPECTED",      # inspect_logs
            "LAYER_CLASSIFIED",    # classify_layer
            "QUERY_ANALYZED",      # analyze_query
            "EVIDENCE_GATHERED",   # compare_runs
        }
        resolution_steps = sum(
            1 for e in events if e["event_type"] in resolution_types
        )
        return min(1.0, resolution_steps * 0.25)

    def count_unresolved(self, pipeline_id: int) -> int:
        events = self.get_by_pipeline(pipeline_id)
        resolution_events = {
            "ROLLBACK_DONE", "REDEPLOY_SUCCESS", "OPTIMIZATION_FLAGGED",
            "ESCALATED", "DATA_CLEANED", "JOB_DEPRIORITIZED",
        }
        unresolved = 0
        for e in events:
            if e["severity"] in ("ERROR", "CRITICAL"):
                unresolved += 1
            elif e["event_type"] in resolution_events:
                unresolved = max(0, unresolved - 1)
        return unresolved

    # ── Stats ──────────────────────────────────────────────────────────────

    @property
    def total_events(self) -> int:
        return len(self._log)

    @property
    def storage_bytes(self) -> int:
        return len(self._log) * 4   # 4 bytes per 32-bit int

    def summary(self) -> str:
        total  = self.total_events
        crits  = len(self._idx_severity.get("CRITICAL", []))
        errors = len(self._idx_severity.get("ERROR", []))
        warns  = len(self._idx_severity.get("WARN", []))
        cats   = {c: len(self._idx_category.get(c, [])) for c in CATEGORY}
        cat_str = " ".join(f"{k}={v}" for k, v in cats.items() if v > 0)
        return (
            f"BitLog | {total} events | "
            f"CRITICAL={crits} ERROR={errors} WARN={warns} | "
            f"Layers: {cat_str} | "
            f"Storage: {self.storage_bytes}B "
            f"(vs ~{total * 120}B as dicts)"
        )
