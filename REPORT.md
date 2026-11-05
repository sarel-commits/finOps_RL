# Qwen3.8-27B (MLX 8-bit) — FinOps Evaluation Report

> **Living report**: updated after every phase. Status of each phase is at the top; sections fill in as results land.
> Every number here comes from a run whose raw output is saved under `~/finops-eval/` (paths given inline).

## Status

| Phase | Status | Notes |
|---|---|---|
| 0. Preflight | ✅ done | M5 Max, 64 GiB, 1.7 TiB free → 8-bit (no 4-bit fallback) |
| 1. Download BF16 | ⏳ in progress | ~34 Mbps link; 55.6 GB |
| 2. Quantize to 8-bit | ⏳ pending | mlx-lm 0.32.2, affine 8-bit, group size 64 |
| 3. Local server | ⏳ pending | mlx_lm.server on 127.0.0.1:8080 |
| 4.1 ITBench-Lite FinOps | ⏳ pending | 15 scenarios × 3 seeds |
| 4.2 Full ITBench FinOps | ⛔ skipped | needs Kubernetes + container runtime (see Skipped) |
| 4.3 aws-bench | ⛔ skipped (by design) | provisions real AWS; review only |
| 4.4 Synthetic FinOps eval | 🟡 data built | 43 tasks, deterministic (seed 20261106) |
| 5. Analysis | ⏳ pending | |

## 1. Setup

- **Hardware**: Apple M5 Max, 64 GiB unified memory (68,719,476,736 bytes), macOS 26.5.1 (25F80), 1.7 TiB free disk.
- **Model source**: `Qwen/Qwen3.8-27B` (HF revision `8c0b3d30f3ae96f5c7f774de704bf0ada83e422a`), BF16, 27.78 B params, arch `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`, image-text-to-text).
- **Quantization**: MLX affine 8-bit, group size 64, via `mlx_lm.convert` (text decoder only; vision tower dropped). *Pending.*
- **Software**: mlx 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.12, transformers 5.22.1 (full lock: `requirements.lock`).
- **Sampling (model card, non-thinking/instruct)**: temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0.
- **Sampling (model card, thinking)**: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0.
- **Thinking mode**: off by default (`chat_template_kwargs: {"enable_thinking": false}`).

## 2. Results

*Pending.*

## 3. Comparison to published reference points

*Pending.*

## 4. Error analysis

*Pending.*

## 5. Performance

*Pending.*

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
