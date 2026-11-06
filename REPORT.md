# Qwen3.8-27B (MLX 8-bit) — FinOps Evaluation Report

> **Living report**: updated after every phase. Status of each phase is at the top; sections fill in as results land.
> Every number here comes from a run whose raw output is saved under `~/finops-eval/` (paths given inline).

## Status

| Phase | Status | Notes |
|---|---|---|
| 0. Preflight | ✅ done | M5 Max, 64 GiB, 1.7 TiB free → 8-bit (no 4-bit fallback) |
| 1. Download BF16 | ✅ done | 55.6 GB in 1h46m; 18/18 shards SHA-256 verified |
| 2. Quantize to 8-bit | ✅ done | affine 8-bit g64 → 27 GB, 8.501 bpw; sanity 2/3 (arithmetic failed) |
| 3. Local server | ✅ done | mlx_lm.server 127.0.0.1:8080; native tool calls OK; sleep-stall bug fixed with caffeinate |
| 4.1 ITBench-Lite FinOps | 🟡 non-thinking done; thinking running (started 09:42) | non-thinking: **48.9% ± 3.8%** (22/45 runs) |
| 4.2 Full ITBench FinOps | ⛔ skipped | needs Kubernetes + container runtime (see Skipped) |
| 4.3 aws-bench | ⛔ skipped (by design) | provisions real AWS; review only |
| 4.4 Synthetic FinOps eval | ✅ done | 43 tasks × 3 seeds: **93.0% ± 2.3%** (120/129) |
| 5. Analysis | ⏳ pending | |

## 1. Setup

- **Hardware**: Apple M5 Max, 64 GiB unified memory (68,719,476,736 bytes), macOS 26.5.1 (25F80), 1.7 TiB free disk.
- **Model source**: `Qwen/Qwen3.8-27B` (HF revision `8c0b3d30f3ae96f5c7f774de704bf0ada83e422a`), BF16, 27.78 B params, arch `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`, image-text-to-text).
- **Quantization**: MLX affine 8-bit, group size 64 (`--q-bits 8 --q-group-size 64 --q-mode affine`), via `mlx_lm.convert` 0.32.2, from the verified BF16 weights. Output 27 GB, **8.501 bits/weight** effective (8-bit codes + fp16 scale/bias per 64-weight group). Text decoder only; vision tower dropped (all benchmarks are text). Path: `models/qwen3.8-27b-mlx-8bit`.
- **Download verification**: 18/18 shards match Hub LFS SHA-256 and size. The repo's `crc32.txt` (which covers only 8 config/tokenizer files) is stale for 3 of them: the local copies are byte-identical to the Hub blobs (see DECISIONS #14).
- **Software**: mlx 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.12, transformers 5.22.1 (full lock: `requirements.lock`).
- **Sampling (model card, non-thinking/instruct)**: temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0.
- **Sampling (model card, thinking)**: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.
- **Thinking mode**: off by default (`chat_template_kwargs: {"enable_thinking": false}`).

## 2a. Sanity check (Phase 2), 8-bit, non-thinking, in-process mlx-lm

Raw output: `results/sanity/sanity_perf_8bit.json`.

| Prompt | Output (truncated) | Pass | Prefill tok/s | Decode tok/s | Peak mem |
|---|---|---|---|---|---|
| Arithmetic 1234×56−789 (=68,315) | `67542` | ❌ | 10.8* | 22.19 | 28.856 GB |
| "What is a reserved instance" | A reserved instance is a billing model that allows organizations to commit to using a spec… | ✅ | 107.51 | 18.41 | 28.865 GB |
| Strict JSON object | `{   "service": "EC2",   "monthly_cost": 1234.5,   "tags": [ ` | ✅ | 254.32 | 18.54 | 28.913 GB |

\* first call includes Metal kernel warm-up; prompt only 30 tokens.

Long-prompt prefill probe (same run): 5,530 tokens → 841.23 tok/s prefill, peak 31.71 GB; 55,030 tokens → 597.37 tok/s prefill, peak 34.991 GB; 220,030 tokens → 263.02 tok/s prefill, peak 54.802 GB.

**Tool calling (Phase 3)**: `mlx_lm.server` parsed the model's XML tool calls into OpenAI `tool_calls` correctly (healthcheck: `results/sanity/healthcheck.json`). All benchmark results use the **native** protocol; the ReAct text fallback was implemented but not needed. Thinking mode returns reasoning in a separate `reasoning` field.

## 2. Results

### 2.1 ITBench-Lite FinOps (official metric), 15 scenarios × 3 seeds

Raw output: `results/itbench_lite/nothink_native/` (`<Scenario>/seed_<s>/trajectory.json`, `scores.jsonl`, `summary.json`). Scoring follows the official ITBench-Evaluations FinOps rubric (`root_cause_resource`: every ground-truth resource must be matched by name **and** type). I applied it deterministically, without an LLM judge (DECISIONS #7). Agent: native tool calling with one `run_python` tool, at most 20 steps, non-thinking, model-card sampling.

| Config | root_cause_resource (official) | n | Pass / fail | Strict set match | Name-only match | F1 (name,type) | Answered (parsed) |
|---|---|---|---|---|---|---|---|
| **8-bit, non-thinking** | **48.9% ± 3.8%** (per seed 46.7 / 46.7 / 53.3) | 45 runs (15 × 3) | 22 / 23 | 35.6% ± 7.7% | 48.9% ± 3.8% | 0.444 ± 0.000 | 28 / 45 |
| 8-bit, thinking | *queued* | | | | | | |
| *Non-LLM heuristic (reference)* | *60.0%* (top instance family by cost delta) | 15 | 9 / 6 | | | | |
| *Non-LLM heuristic, auto-dimension* | *46.7%* | 15 | 7 / 8 | 46.7% | | 0.526 | |

± is the std across the 3 seed-level accuracies. The heuristic baselines are `results/itbench_lite/_heuristic_baseline/` (`scripts/itbench_heuristic_baseline.py`). They are a sanity reference for task difficulty, **not** an agent result.

**What the numbers mean**
- **The step budget is the main failure.** 17 of 45 runs (38%) used all 20 steps and never gave an answer. Of the 28 runs that did answer, **22 were correct (78.6%)**. The model can usually find the root cause but often fails to commit to it. It keeps re-running near-identical pandas analyses (e.g. `Scenario-4/seed_2`: steps 7–18 are variations of one group-by/delta script, and step 19 says "The root cause is clear. Let me verify…" before the budget runs out).
- **Wrong dimension.** All 6 answered-but-wrong runs picked the wrong *column*. S6 (20 Application root causes) got Instance Families twice. S3 (one Application root cause) got Instance Families. S1 and S7 got an Application where the answer was an Instance Family. S8 got Cloud Services.
- **Over-inclusion** (passes the official metric but not strict set match): 6 runs listed an extra resource. S5 did this in all 3 seeds.
- **Per scenario** (pass rate across seeds): S5, S10 and S13 are 3/3. S3, S4, S6, S11 and S15 are 0/3. S4, S11 and S15 failed only by step-cap timeouts. S3 and S6 mix timeouts with wrong-dimension answers: these are the two scenarios whose root cause is an Application.

![ITBench-Lite per scenario](report_assets/itbench_lite_per_scenario.png)

### 2.2 Synthetic FinOps eval (supplementary, NOT an official benchmark)

This is my own benchmark, generated by `scripts/gen_synthetic.py` (seed 20261106, ground-truth sha256[:16] `6f909dfab764aed0`). It has 43 tasks over FOCUS-format billing exports, inventories and pure commitment arithmetic. Scoring is exact match or numeric tolerance per field, and a task passes only if every field matches. Same agent and sampling as 2.1 (non-thinking). Raw output: `results/synthetic/nothink_native/`.

**Overall: 93.0% ± 2.3%** (per seed 93.0 / 90.7 / 95.3). n = 129 runs (43 × 3): 120 pass, 9 fail. Mean 42 s/run.

| Category | Tasks | Accuracy (mean ± std over seeds) | Pass / runs |
|---|---|---|---|
| spend_total | 8 | 100% ± 0 | 24 / 24 |
| mom_change | 6 | 100% ± 0 | 18 / 18 |
| rightsizing | 6 | 100% ± 0 | 18 / 18 |
| anomaly_detection | 6 | 100% ± 0 | 18 / 18 |
| commitment_math | 6 | 88.9% ± 9.6% | 16 / 18 |
| top_drivers | 5 | 80.0% ± 20.0% | 12 / 15 |
| idle_resources | 6 | 77.8% ± 9.6% | 14 / 18 |

Failures:
- **6 step-cap timeouts.** Two each on idle_03, idle_05 and top_drivers (01, 02). idle_05's correct answer is an *empty list*, and the model kept searching rather than conclude that nothing qualifies.
- **2 wrong computations.** commit_04: Savings-Plan overflow pricing, answered 80.88 instead of 81.22. commit_06: answered 20 RIs / $905.20 instead of 12 RIs / $8,342.44. $905.20 = 20 × $0.062 × 730 h is one *month* of RI cost, so it mixed monthly and yearly figures and chose the wrong RI count.
- **1 wrong entity.** top_drivers_04 seed 2 returned the pandas row index `"5"` as the ResourceId. The increase it reported was correct.

The synthetic anomaly tasks give the date, account and service of a single injected spike, and they are much easier than ITBench-Lite. The model scored 100% on them but 48.9% on ITBench-Lite. **This number must not be compared with the ITBench scores.** It shows the model's arithmetic and pandas skills are sound when the task is tightly specified.

![Synthetic per category](report_assets/synthetic_per_category.png)

### 2.3 Full ITBench FinOps and aws-bench

Not run (see section 7 and section 8).

## 3. Comparison to published reference points

| Reference | Number | Our closest number |
|---|---|---|
| ITBench paper: FinOps, excluding anomaly detection | ~25.8% | none (those scenarios need Kubernetes; not run) |
| ITBench paper: FinOps anomaly detection | F1 0.35 | ITBench-Lite non-thinking: F1 **0.444**, root_cause_resource **48.9%** |

**These are not like-for-like.** Read the comparison as "same order of magnitude, plausibly a bit better", not as a ranking.
1. **Different task form.** The paper's agents worked on live Kubernetes/OpenCost environments. ITBench-Lite gives static snapshots: an `anomaly.json` plus a `data.csv`, with the anomaly date and account already given. Lite is the easier and narrower task.
2. **Different scorer.** The official evaluator is an LLM judge. I implemented its binary rubric deterministically (DECISIONS #7). It may be stricter or more lenient on near-miss names and types than a judge would be.
3. **Different agent and prompt.** I used my own single-tool Python agent and prompt (DECISIONS #8), not the paper's agent stack. The prompt tells the model how the type label is formed, which a judge-scored setup might not need.
4. **Different model and precision.** Qwen3.8-27B at 8-bit, versus the frontier and open models in the paper.
5. **Small n.** 15 scenarios × 3 seeds. One scenario is about 6.7 percentage points.

**Internal reference.** A non-LLM heuristic (largest instance-family cost delta) reaches 60% on the same 15 scenarios. The 8-bit model, non-thinking, does not beat it, mostly because 38% of runs hit the step cap.

## 4. Error analysis

*Pending.*

## 5. Performance

All figures are measured on this M5 Max (64 GiB), 8-bit model, single request at a time.

**Raw engine** (in-process mlx-lm, `results/sanity/sanity_perf_8bit.json`)

| Metric | Value |
|---|---|
| Decode speed | 18.4–22.2 tok/s |
| Prefill speed | 841 tok/s @ 5.5k tokens · 597 tok/s @ 55k · 263 tok/s @ 220k |
| Peak memory | 28.9 GB (short prompts) · 35.0 GB @ 55k-token prompt · 54.8 GB @ 220k-token prompt |
| Model load | 27 GB of weights on disk |

**Agent loop via `mlx_lm.server`** (per run, from `scores.jsonl`)

| Benchmark (non-thinking) | Runs | Mean / median / p90 / max wall per run | Mean steps | Effective output rate* | Total compute |
|---|---|---|---|---|---|
| ITBench-Lite | 45 | 261 / 234 / 417 / 508 s | 14.2 | 13.4 tok/s | 3.26 h |
| Synthetic | 129 | 42 / 31 / 89 / 147 s | 5.5 | 13.8 tok/s | 1.50 h |
| ITBench-Lite, thinking | *running* | | | | |

\* Output tokens divided by time spent inside server calls (98–99% of wall time; tool execution is negligible). It is lower than raw decode because it includes prefill of each new step's observation. The server's prompt cache (10 sequences, ~4.5 GB) avoids re-prefilling the conversation prefix.

![Latency per task](report_assets/latency_per_task.png)

**Operational note.** Before the fix, generation stalled for about 16 minutes at a time. macOS idle sleep (`sleep 1` with Power Nap dark-wakes about every 15 minutes) suspended the GPU mid-run. Running the server and the benchmark chain under `caffeinate -is` fixed it (DECISIONS #15). Anyone running unattended local evals on a Mac needs this.

## 6. Quantization caveat

MLX affine 8-bit (per-group scale + bias, group size 64, integer codes) is **not** FP8 (E4M3/E5M2 floating-point). Apple Silicon has no FP8 hardware path, so this is the closest practical equivalent. BF16-vs-8-bit degradation: *pending*.

## 7. Skipped items

See `DECISIONS.md` for full reasoning.

- **Full ITBench FinOps scenarios (37/38)**: need a Kubernetes cluster (kind/minikube → Docker/Podman), helm, kubectl. None installed, no Homebrew, and installing a container VM means system-level changes, which the hard rules forbid.
- **aws-bench execution**: provisions real AWS resources and costs money. Review only (section 8).

## 8. aws-bench review (not executed)

Reviewed `aws-bench/aws-bench` @ `ec7ec56` and `aws-bench/aws-bench-datasets` (cloned to `repos/`). **Not run**: it provisions real AWS Organizations member accounts and deploys real infrastructure, which costs money (hard rule).

**What running it would take**
- An AWS management account allowed to create an AWS Organization and member accounts, plus Docker with Compose v2 and buildx ≥ 0.17 (not installed here).
- `aws-bench env init` → `env setup` (CDK-deploys each scenario into its own account; opt-in regions/quota increases can take hours), `aws-bench run -d <dataset> -a <agent> -m <model>`, then `env cleanup` (stops charges) and optionally `env terminate` (accounts suspended for 90 days, then closed).
- **Judge**: the 99 read-only "introspection" tasks are scored by an **LLM judge (Anthropic model via Bedrock or `ANTHROPIC_API_KEY`)**, which is a paid API. Only the 35 "mutation" tasks use programmatic `check.py` verifiers.
- **Pointing it at this local model**: the `mini-swe-agent` adapter accepts any LiteLLM provider, so `-m openai/<name>` with `OPENAI_API_BASE=http://host:8080/v1` should work in principle. The agent container must be able to reach the Mac's server (e.g. `host.docker.internal`).

**Size**: 134 tasks across 8 scenarios (`aws-bench-basic` = 78 tasks/4 scenarios, `aws-bench-advanced` = 47/3, `aws-bench-quickstart` = 9/1).

**FinOps-relevant tasks** (by instruction text; most of the benchmark is general ops):
- `databases-and-storage/estimate-ec-running-instance-costs`: total hourly on-demand cost of running EC2 in us-east-1, broken down by instance type and OS. This is the most direct FinOps task.
- `databases-and-storage/list-buckets-with-lifecycle-policy`, `serverless-apps/check-s-lifecycle-tag-filter-rules`: storage lifecycle / expiration policy (storage cost hygiene).
- `serverless-apps/list-s-buckets-with-metrics-tag-filters`, `serverless-apps/s-bucket-metrics-tag-filter-check`, `serverless-apps/find-lambda-functions-tagged-blueprint`: tag-based resource discovery (cost allocation prerequisites).
- `compute-and-data/analyze-b2b-commerce-summary-and-lifecycle` (mutation: adds a lifecycle transition policy).
- That is about 7 of 134 tasks (~5%). There are no billing/CUR, Cost Explorer, Savings Plan or anomaly tasks.

**Estimated cost and effort** (rough order of magnitude from resource counts in the CDK stacks, **not measured**): across all 8 scenarios the stacks declare ~70 VPCs (several with 1–2 NAT gateways), 28 EC2 instances + 7 ASGs, 14 Fargate services, 17 ALB/NLBs, 6 RDS/Aurora, 3 Redshift clusters + 1 serverless workgroup, 3 OpenSearch domains, 3 MSK clusters, 1 ElastiCache. Standing infrastructure is therefore likely on the order of **$100–250/day** while deployed. A focused run of just `databases-and-storage` + `serverless-apps` for one day is plausibly **$20–60** plus judge-API tokens (a few dollars). Effort: about 0.5–1 engineer-day for account/Org setup and first deploy (longer if quota or opt-in-region approvals are needed), then hours per run. With a local agent model, the agent's own inference is free but slow (see Performance).

## 9. Recommendations

*Pending.*
