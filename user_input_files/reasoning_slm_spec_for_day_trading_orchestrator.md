# Reasoning SLM (Small Language Model) – Full Specification for Day‑Trading Orchestrator

A production‑ready spec to deploy a reasoning‑centric SLM to orchestrate a modular intraday trading system: trend, stat‑arb, cross‑venue arbitrage, event/news reaction, and RAG research. Emphasis on safe tool use, strict JSON I/O, deterministic execution, and auditable governance.

---

## 0) Executive Summary
- **Primary SLM**: DeepSeek‑R1‑Distill‑Qwen‑14B (int8/int4) → best open reasoning at small/medium scale.
- **Fast Path**: Qwen2.5‑7B‑Instruct → summaries, RAG fetch, low‑risk tool calls.
- **Tiny Path**: Phi‑3.5‑mini‑instruct → watchdogs, keep‑alives, non‑critical transforms.
- **Runtime**: vLLM or TGI with **JSON schema enforcement** and **tool/function calling**.
- **Role**: Orchestrator/Analyst only. **No direct order placement**. All trade actions flow through deterministic policies + risk engine.

---

## 1) Core Objectives & Non‑Goals
### Objectives
1. Produce **bounded, verifiable decisions** (signals, params, hypotheses) via tool calls.
2. Improve research iteration speed (RAG queries, experiment configs) without increasing operational risk.
3. Maintain **low latency SLOs** suitable for intraday operation.
4. Preserve **auditability** (why, what, when, with which data).

### Non‑Goals
- SLM is **not** a price predictor nor an execution engine.
- No free‑form natural language outputs in critical paths; everything is **schema‑validated JSON**.

---

## 2) System Context & Data Flows
```
[Market Feeds] --> [Feature Service] --> [Predictors] --> [Policy Engine] --> [OMS/Execution]
                                             ^                 ^
                                             |                 |
                                     [Reasoning SLM] <--------+-- [RAG KB]
                                             |
                                       [Tool Layer]
```
**SLM responsibilities**
- Fetch features, retrieve KB items, propose parameter ranges, pick strategy playbooks, draft backtest configs, score hypotheses, generate incident postmortems.
- Return **decisions and plans** that downstream services validate and execute.

---

## 3) Hardware & Deployment Profiles
| Tier | Use Case | Model | Quant | Host | VRAM/CPU | Latency p95 | Notes |
|---|---|---|---|---|---|---|---|
| A | Primary Reasoner | R1‑Distill‑Qwen‑14B | int8 (AWQ) | vLLM | ≥16GB GPU | 250–600 ms | Main tool‑calling brain |
| B | Fast Path | Qwen2.5‑7B‑Instruct | int8/gguf | vLLM/llama.cpp server | 8–12GB | 80–250 ms | Summaries, RAG, low‑risk ops |
| C | Tiny | Phi‑3.5‑mini | int4 | llama.cpp server | ≤8GB / CPU OK | 120–300 ms | Health checks, glue tasks |

**Throughput goals**: ≥20 req/s burst. Horizontal autoscale via HPA on tokens/s and queue depth.

---

## 4) Runtime Config (vLLM / TGI)
**vLLM baseline flags** (illustrative):
```
vllm serve <model> \
  --dtype=auto \
  --max-model-len 8192 \
  --enforce-eager \
  --tokenizer <tok> \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enable-json-output \
  --tool-call-parser openai \
  --trust-remote-code false
```
**TGI**: enable `--json-output` and `--tool` plugins; set `--stop-sequences` conservatively to avoid tail chatter.

**Safety decoding**
- Temperature ≤0.2 critical; ≤0.5 research.
- `response_format={"type":"json_object"}` for JSON routes.
- Length caps and **grammar/JSON schema** enforced in gateway (see §6).

---

## 5) Prompting & System Messages
### Global System Prompt (orchestrator)
- You are a **risk‑aware trading analyst**. You **must** call tools to get data. You **never** place or suggest orders directly. Return **JSON** that strictly matches the provided schema. When uncertain, request **specific tools** to reduce uncertainty. Cite KB IDs in the `citations` field.

### Tool‑Use Style Guide
- Prefer **idempotent** reads first, then sims/backtests.
- Output **param ranges** (min/max/step) instead of single numbers unless instructed.
- Include `assumptions` and `risk_flags` arrays.

---

## 6) Canonical JSON Schemas (Gateway‑enforced)
### 6.1 Hypothesis Proposal
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["symbol", "strategy", "hypothesis", "tests"],
  "properties": {
    "symbol": {"type": "string"},
    "strategy": {"enum": ["trend", "mean_revert", "pairs", "cross_venue_arb"]},
    "hypothesis": {"type": "string", "maxLength": 400},
    "tests": {
      "type": "array",
      "items": { "$ref": "#/definitions/testSpec" },
      "minItems": 1
    },
    "citations": {"type": "array", "items": {"type": "string"}},
    "assumptions": {"type": "array", "items": {"type": "string"}},
    "risk_flags": {"type": "array", "items": {"type": "string"}}
  },
  "definitions": {
    "testSpec": {
      "type": "object",
      "required": ["backtest_config"],
      "properties": {
        "backtest_config": {
          "type": "object",
          "required": ["lookback_days", "horizons", "params"],
          "properties": {
            "lookback_days": {"type": "integer", "minimum": 30},
            "horizons": {"type": "array", "items": {"type": "integer"}},
            "params": {"type": "object"}
          }
        }
      }
    }
  }
}
```

### 6.2 Parameter Proposal (bounded)
```json
{
  "type": "object",
  "required": ["strategy_id", "param_ranges"],
  "properties": {
    "strategy_id": {"type": "string"},
    "param_ranges": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["min", "max", "step"],
        "properties": {
          "min": {"type": "number"},
          "max": {"type": "number"},
          "step": {"type": "number"}
        }
      }
    },
    "justification": {"type": "string", "maxLength": 300},
    "citations": {"type": "array", "items": {"type": "string"}}
  }
}
```

### 6.3 Risk Check Request
```json
{
  "type": "object",
  "required": ["intent"],
  "properties": {
    "intent": {
      "type": "object",
      "required": ["strategy_id", "symbol", "target_notional", "urgency"],
      "properties": {
        "strategy_id": {"type": "string"},
        "symbol": {"type": "string"},
        "target_notional": {"type": "number", "minimum": 0},
        "urgency": {"enum": ["low", "medium", "high"]},
        "max_slippage_bps": {"type": "integer", "minimum": 0}
      }
    },
    "explain": {"type": "string", "maxLength": 250}
  }
}
```

---

## 7) Tooling API (Contracts)
### 7.1 Feature Fetch
**Name**: `get_features`
- **Input**: `{ symbol: string, horizon_s: int, fields?: string[] }`
- **Output**: `{ ts: int64, features: { field: number } }`
- **SLO**: p95 ≤ 60 ms
- **Notes**: Reject if data fresheness > 500 ms.

### 7.2 Backtest
**Name**: `backtest`
- **Input**: `{ strategy_id, params, universe, start, end, latency_model, costs }`
- **Output**: `{ metrics: { sharpe, sortino, max_dd, pnl, turnover }, by_day: [...], by_symbol: [...] }`
- **SLO**: submit job ≤ 200 ms; results async via job id.

### 7.3 RAG Query
**Name**: `rag`
- **Input**: `{ query, filters?: {symbol?, date_range?, doc_type?}, top_k?: int }`
- **Output**: `{ passages: [{id, text, source, ts}], usage: {...} }`
- **SLO**: p95 ≤ 120 ms.

### 7.4 News Sentiment
**Name**: `news_sentiment`
- **Input**: `{ symbol, lookback_min }`
- **Output**: `{ score: -1..1, velocity: number, items: [{id, ts, novelty}] }`

### 7.5 Risk Limits
**Name**: `risk_limits`
- **Input**: `{}`
- **Output**: `{ per_symbol_caps, global_caps, borrow_constraints, restricted_list }`

> All tool I/O validated by the gateway. Any mismatch → auto‑retry once with tightened temperature.

---

## 8) Routing & Policy Logic
### Router (pseudo)
```
if route == "critical": use R1-14B, temp=0.15, json-only, tools enabled
elif route == "fast": use Qwen2.5-7B, temp=0.2, json-only, tools enabled
else: use Phi-3.5, temp=0.3
```
**Critical routes**: param proposals for live, risk checks, incident postmortems, strategy selection.
**Fast routes**: RAG fetch/summarize, doc lookup, non‑critical configs.

**Backoff**: If JSON validation fails → retry with `temperature=0.0` and `top_p=0.8`. If still failing → escalate to bigger model once.

---

## 9) Latency, Throughput & SLOs
- **Gateway p95**: ≤ 40 ms overhead.
- **Model p95**: see §3 table.
- **End‑to‑end (critical tool call)**: ≤ 500–900 ms.
- **Error budgets**: 99.5% monthly for critical; 99.0% for research.

Autoscale by **queue depth** & **tokens/s** (target utilization 60–70%).

---

## 10) Observability & Logging
- **Per request**: `req_id`, `route`, `model`, `prompt_hash`, `schema_id`, `tool_calls[]`, `latency_ms`, `tokens_in/out`, `retry_count`, `json_valid`.
- **Model health**: refusal rate, invalid‑JSON rate, tool miss rate, factual‑error flags (from unit tests), drift of output fields.
- **Audit trails**: store prompts, tool results, final JSON, and **KB citation IDs**.

Dashboards: latency histograms, error budget burndown, invalid‑JSON heatmap by prompt family.

---

## 11) Safety, Governance & Permissions
- **No direct OMS access**. SLM can only emit **intents**; policy engine converts to orders after risk.
- **Feature flags** gate any new tool or schema.
- **Two‑person approval** for prompt or schema changes affecting critical routes.
- **Kill‑Switches**: circuit breaker on invalid JSON spike, on risk‑intent volume spike, or on RAG failure > threshold.

---

## 12) Evaluation Framework
### Datasets
- **Tool‑grounded eval**: synthetic stubs for `get_features`, `rag`, `backtest` with deterministic outputs.
- **Task sets**: param‑proposal tasks, hypothesis generation with constraints, risk‑intent formulation, incident RCA.

### Metrics
- **Functional**: JSON validity %, schema coverage %, tool choice accuracy, constraint satisfaction.
- **Economic proxy**: improvement in backtest exploration efficiency (jobs to first viable Sharpe), reduction in human review time.
- **Robustness**: adversarial prompts (tempt to place orders), hallucination rate, citation presence.

### CI
- Run nightly regression with fixed seeds + replay of live prompts (PII stripped).

---

## 13) Prompt Libraries (Templates)
### 13.1 Param Proposal Template
```
SYSTEM: You are a risk-aware trading analyst. Output JSON only.
TOOLS: get_features, backtest, rag, news_sentiment, risk_limits
SCHEMA: ParameterProposal v1

USER: Propose bounded parameters for {strategy_id} on {symbol} for horizons {H}. Respect risk limits and today’s news velocity.
```

### 13.2 Hypothesis Template
```
SYSTEM: Output JSON matching HypothesisProposal v1. Use rag() for prior incidents and playbooks.
USER: Given {context}, generate one falsifiable hypothesis and 1-3 backtest configs.
```

### 13.3 Incident Postmortem Template
```
SYSTEM: Output JSON {cause_tags[], lessons[], kb_refs[]}.
USER: Analyze slippage spike on {symbol} at {ts}. Fetch venue health logs and news velocity.
```

---

## 14) KB & RAG Design
- **Corpus**: strategy playbooks, risk policies, exchange/broker docs, fee tables, incident logs, venue health reports, postmortems.
- **Chunking**: 1–2k tokens, overlap 10–15%.
- **Embeddings**: bge‑m3 or E5‑Mistral; re‑rank with cross‑encoder for top‑k.
- **Retrievers**: symbol‑aware, time‑aware (bias to last 90 days), doc‑type filters.
- **Citations**: return `{id, page, ts}` for each passage; SLM must echo IDs.

---

## 15) Security & Secrets
- **KMS/HSM** for broker keys (not accessible to SLM).
- **Network**: SLM pods cannot reach OMS subnets; only Tool Gateway with policy checks can.
- **PII scrub** on logs; rotate prompt caches daily.

---

## 16) Rollout Plan
1. **Sandbox**: Wire SLM to **simulated tools** only; pass eval suite (JSON≥99.5%, tools≥98%).
2. **Paper‑trade**: Read‑only market feeds; generate intents; policy engine runs in shadow.
3. **Limited capital**: Enable a single strategy family; hard caps; 24/7 on‑call; auto demotion on breach.
4. **Gradual expansion**: Add strategies/venues; A/B across model tiers; periodic postmortems.

---

## 17) Failure Modes & Mitigations
- **Invalid JSON spikes** → grammar hard mode, temp=0.0, switch to Fast Path.
- **Tool hallucination** → strict registry: unknown tool names rejected; retry with tool list reminder.
- **RAG miscites** → require `citations[]` non‑empty; gateway verifies IDs exist.
- **Prompt drift** → pin prompt versions; changelog + two‑person review.

---

## 18) Cost & Capacity Planning (ballpark)
- **A tier**: ~3–6M tokens/day → single 24GB GPU node or 2×16GB; autoscale to 3 nodes on news days.
- **B tier**: offload 40–60% of volume; tiny tier handles 10–20% glue.
- **Storage**: 30–60GB/month logs + 5–10GB/month KB.

---

## 19) Example End‑to‑End Trace (Critical Route)
1. Router classifies: `param_proposal` → **critical** → A‑tier model.
2. SLM calls `risk_limits()` → gets caps.
3. SLM calls `news_sentiment(symbol, 120)` → sees high velocity → tightens ranges.
4. SLM calls `backtest()` with a small grid; gateway spins job; returns job id.
5. SLM returns JSON **ParameterProposal v1** with bounded ranges + `citations`.
6. Policy engine validates; schedules async backtests; promotion rules apply when results available.

---

## 20) Artifacts & Versioning
- **Schemas**: `schemas/param_proposal.v1.json`, `schemas/hypothesis.v1.json`.
- **Prompts**: `prompts/<route>/<version>.md`.
- **Tool contracts**: `tools/*.yaml` (OpenAPI‑ish).
- **Changelogs**: `CHANGELOG_SLMonly.md`.

---

## 21) Acceptance Criteria (Phase 1)
- JSON validity ≥99.5% on critical routes in CI.
- Tool selection accuracy ≥98% vs gold.
- Postmortem generation with ≥1 validated KB citation in 95% of cases.
- p95 latency ≤900 ms E2E on critical calls under peak.

---

## 22) Stretch Goals
- **Contextual bandit** to pick which prompt template works best per symbol/regime.
- **Self‑healing**: dynamic narrowing of param ranges based on morning session microstructure.
- **Multi‑agent debate** (budgeted) for incident root cause, adjudicated by deterministic rules.

---

## 23) Quick Start Checklist
1. Stand up vLLM for R1‑14B and Qwen2.5‑7B.
2. Implement Gateway with JSON Schema enforcement + tool registry.
3. Wire 5 tools: `get_features`, `backtest`, `rag`, `news_sentiment`, `risk_limits`.
4. Load KB; build retriever with symbol/time filters.
5. Add prompts & CI eval suite. Gate to paper‑trade.
6. Add dashboards, alerts, and kill‑switch paths.

---

## 24) Appendix: Example Gateway Policy (YAML)
```yaml
routes:
  critical:
    model: r1-qwen-14b
    temperature: 0.15
    json: true
    max_tokens: 800
    tools: [get_features, backtest, rag, news_sentiment, risk_limits]
    retries: 1
  fast:
    model: qwen2.5-7b
    temperature: 0.2
    json: true
    max_tokens: 600
    tools: [get_features, rag]
    retries: 1
  tiny:
    model: phi-3.5-mini
    temperature: 0.3
    json: true
    max_tokens: 400
    tools: []
    retries: 0
validation:
  json_schema_required: true
  citation_required: ["hypothesis", "postmortem"]
  forbidden_phrases: ["place an order", "send order", "buy now", "sell now"]
```

