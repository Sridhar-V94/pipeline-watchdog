---
title: Pipeline Watchdog AI
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
app_file: server/app.py
pinned: false
license: mit
---

# 🔍 PipelineWatchdog v2 — OpenEnv Environment

> Autonomous AI system for real-world data pipeline reliability and optimization

Built for the **OpenEnv AI Hackathon** (Meta × Hugging Face × PyTorch)

---

## 🔍 Problem
Modern data pipelines fail due to:
- Bad deployments (wrong version, missing load script changes)
- SQL performance issues (full table scans, missing indexes)
- Full reload overload (loading 50M+ rows every run)
- Unclear ownership when issues span multiple layers (blame game)

Manual debugging is slow and costly, especially during production incidents.

---

## 💡 Solution
A reinforcement learning environment where an AI agent learns to:
- Diagnose pipeline failures across deployment, SQL, and script layers
- Take corrective actions following real engineering rules
- Gather evidence before escalating (no proof = blame game)
- Minimize downtime through correct triage

---

## 🎮 Environment Design

### Observation Space
- `event_type` — what happened (VERSION_MISMATCH, SLOW_QUERY, FULL_RELOAD_OVERLOAD...)
- `event_severity` — INFO / WARN / ERROR / CRITICAL
- `event_category` — SYSTEM / SCRIPT / API / NETWORK
- `pipeline_id` / `pipeline_name` — which pipeline
- `evidence_score` — how much proof has been gathered (0.0–1.0)
- `unresolved_count` — open issues on this pipeline
- `hint` — shown when agent picks wrong action

---

### Actions

| Action | When to use |
|--------|-------------|
| `rollback_deployment` | VERSION_MISMATCH or RELOAD_FAILURE — always first |
| `verify_and_redeploy` | After rollback + fix + retest |
| `inspect_logs` | Free — build evidence, see history |
| `classify_layer` | Free — is it SQL/script or API/network? |
| `compare_runs` | Free — what changed between episodes? |
| `analyze_query` | Free — run EXPLAIN to find SQL bottleneck |
| `flag_for_optimization` | SQL/data model debt — schedule next sprint |
| `escalate_with_evidence` | With proof (evidence_score ≥ 0.5) |
| `deprioritize_job` | TASK_CLASH — stagger conflicting jobs |
| `add_incremental_load` | FULL_RELOAD_OVERLOAD — fetch only new rows |
| `clean_data` | NULL_SPIKE_PARTIAL — after inspect |
| `ignore` | INFO events only |

---

### Reward
- Correct action: +0.05 to +0.40 (based on severity)
- Wrong action: -0.10 to -0.40
- Ignoring CRITICAL: -0.35
- Evidence bonus: up to +0.20 for thorough diagnosis

---

## 📋 Tasks

### Easy — Post-Deployment Version Mismatch
Dev pushed wrong version → load script fails → rollback immediately → fix → redeploy

### Medium — Rushed Project Issues
Full reload on 50M rows + task clash at 6AM + partial NULLs in one region

### Hard — SQL Performance + Missing Index
No index on 100M-row join → 45-min reloads → concurrent users amplify problem → DBA escalation needed with query plan proof

---

## 🗄️ BitLog Architecture

Every event stored as **one 32-bit integer**:
```
[4-bit severity][4-bit category][6-bit pipeline_id][8-bit event_type][10-bit timestamp]
```

4 HashMap indexes — all O(1) lookup:
- `pipeline_id` → event positions
- `severity` → event positions
- `category` → event positions (blame analysis)
- `episode_id` → event positions (cross-run comparison)

**Storage: 4 bytes/event vs ~120 bytes as dict = 30x more efficient**

---

## ▶️ Setup Instructions

```bash
pip install -r requirements.txt

# Run Streamlit demo
streamlit run app.py

# Run baseline inference (OpenAI)
export OPENAI_API_KEY=sk-...
python inference.py

# Run baseline inference (HuggingFace)
export OPENAI_API_KEY=hf_...
export OPENAI_API_BASE=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
python inference.py
```

## 🐳 Docker

```bash
docker build -t pipeline-watchdog .
docker run -p 8501:8501 pipeline-watchdog
```
