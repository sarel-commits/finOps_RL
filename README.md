# finOps_RL

Evaluation of **Qwen3.8-27B** (run locally as MLX 8-bit on an M5 Max) as a FinOps agent, plus an RL environment for
training it on cost-anomaly root-cause analysis.

- **Eval**: ITBench-Lite FinOps (15 scenarios × 3 seeds, official `root_cause_resource` metric, scored deterministically)
  and a supplementary synthetic FinOps benchmark (43 tasks × 3 seeds). Same single-tool (`run_python`) agent across
  local and API models.
- **RL**: `rl/` is an RLVR environment (procedural task generator, sandboxed tool, deterministic reward) with a
  prime-rl config for multi-GPU and an Unsloth GRPO trainer for a single H100.

Full write-up: [REPORT.md](REPORT.md) · decision log: [DECISIONS.md](DECISIONS.md) · run log: [PROGRESS.md](PROGRESS.md)
· Hebrew executive summary: [EXECUTIVE_SUMMARY_HE.md](EXECUTIVE_SUMMARY_HE.md) · RL environment: [rl/README.md](rl/README.md)

## Key results

ITBench-Lite FinOps, official metric, 45 runs per config. A one-line heuristic (largest instance-family cost delta)
scores 60%.

| Model | Harness | Official | Strict set match |
|---|---|---|---|
| Qwen3.8-27B 8-bit (local) | original | 48.9% | 35.6% |
| Qwen3.8-27B 8-bit (local) | nudges + prompt v2 | 84.4% | 35.6% |
| Qwen3.8-27B 8-bit (local) | + thinking (medium) + repeat guard | **100%** | 60.0% |
| Qwen3.8-Flash (Together) | original | 20.0% | 20.0% |
| Qwen3.8-Flash (Together) | + thinking (medium) + nudges + v2 + guard | 95.6% | **93.3%** |

Synthetic benchmark (27B): 93.0% → 99.2% with nudges + prompt v2.

- **The harness mattered more than the model.** Most baseline failures were runs that hit the 20-step cap without
  answering. Showing the remaining budget, forcing a final answer, and asking the model to compare every column
  fixed most of them; thinking at medium effort fixed most of the rest.
- **On the 23 ITBench-Lite runs the 27B baseline failed**, with the original harness: Claude Opus 5.5 solved 22,
  Claude Sonnet 5.5 and Kimi-K3.1-Pro 21, GLM-5.3 12, Qwen3.8-Flash 1 ([REPORT.md §2.3, §2.6](REPORT.md)).
- **Caveats**: prompt v2 and the guard were tuned on the same 15 scenarios, so the improved-harness numbers are an
  upper bound until confirmed on unseen scenarios. The official metric is recall-only; strict set match penalizes
  extra resources.

## Repository layout

```
REPORT.md, DECISIONS.md, PROGRESS.md   report, decision log, progress log
data/
  itbench-lite/      ITBench-Lite FinOps snapshots (anomaly.json + data.csv per scenario)
  synthetic/         synthetic tasks, workspaces and ground truth (scripts/gen_synthetic.py)
scripts/
  agent.py           single-tool agent loop (native tool calls or ReAct; OpenAI-compatible or Anthropic)
  run_bench.py       checkpointed benchmark runner (harness options: nudges, prompt v2, guard, thinking)
  score.py           deterministic scoring -> scores.jsonl / summary.json
  analyze.py, followup_analysis.py, compare_failedruns.py, compare_all_subset.py   analysis + charts
  server.sh          start/stop the local mlx_lm server (under caffeinate)
  run_*.sh           the exact run chains used for each phase
results/
  itbench_lite/<config>/, synthetic/<config>/   trajectories, scores and summaries per config
  analysis/          error analysis and cross-model comparisons (JSON)
  sanity/            model sanity, performance and server probes
report_assets/       charts used in REPORT.md
rl/                  RL environment and trainers (see rl/README.md)
```

`models/` (weights), `repos/` (cloned benchmarks) and `.env` (API keys) are not in the repo.

## Reproducing the eval

Requires Apple Silicon with ≥ 64 GB memory for the local 27B model. Python dependencies are pinned in
`requirements.lock`.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.lock

# Model: download BF16, then quantize to MLX 8-bit (affine, group size 64)
bash scripts/download.sh
python -m mlx_lm convert --hf-path models/qwen3.8-27b-bf16 --mlx-path models/qwen3.8-27b-mlx-8bit \
  -q --q-bits 8 --q-group-size 64 --q-mode affine

# Serve locally, run a config, score it
bash scripts/server.sh start
python scripts/run_bench.py --bench itbench_lite --config nothink_native --seeds 1 2 3
python scripts/score.py itbench_lite nothink_native

# Improved harness with thinking
python scripts/run_bench.py --bench itbench_lite --config think_medium_nudges_v2_guard --seeds 1 2 3 \
  --thinking --reasoning-effort medium --budget-nudges --prompt-version v2 --repeat-guard
```

API models use the same runner with `--base-url`, `--model` and `--api-key-env` (OpenAI-compatible, e.g. Together),
or `--provider anthropic` for Claude; keys are read from environment variables and never logged. See
`scripts/run_*.sh` for the exact commands behind each result.

On macOS, run long unattended evals under `caffeinate -is` (`server.sh` does this); idle sleep otherwise stalls the
GPU mid-run.

## RL environment

`rl/` turns the ITBench-Lite task into an RL environment: a seeded generator produces unlimited labelled billing
datasets (clean and noisy regimes, ambiguous dimensions, 1–20 root causes), the `run_python` tool is sandboxed so the
policy cannot read labels, and the reward is `0.5 × F1 + 0.5 × exact set match`. ITBench-Lite is held out for
evaluation. Training runs either with prime-rl (multi-GPU LoRA-GRPO) or `rl/scripts/train_grpo_unsloth.py` on a
single H100 with the FP8 checkpoint. Details and run notes: [rl/README.md](rl/README.md).

### Training experiments (1× H100, AWS p5.4xlarge)

LoRA-GRPO on `Qwen/Qwen3.8-27B-FP8` with `rl/scripts/train_grpo_unsloth.py`: 6 generated tasks × 8 samples per step,
LoRA r=16 (attention, MLP and GatedDeltaNet projections), lr 2e-5, 20 tool steps per episode, 32k context. Every 8
steps the policy is evaluated on the 15 held-out ITBench-Lite scenarios (1 sample each, exact set match). Raw
outputs (synced from the instance, not in git): `rl/results/aws_train*/` (`config.json`, `metrics.jsonl`,
`eval.jsonl`, `train.log`, LoRA checkpoints every 5 steps).

| Run | Change | Outcome |
|---|---|---|
| 1 `run1_lockstep` | rollouts in lockstep (every episode waits for the slowest one each turn) | stopped during step 0; replaced by asynchronous rollouts (each episode advances as soon as its turn or tool call finishes) |
| 2 `run2_bf16kv` | asynchronous rollouts, bf16 KV cache (~492k tokens) | 1 step: 1,333 s (rollout 917 s, train 416 s); stopped to enlarge the KV cache |
| 3 `aws_train` | `--fp8-kv` (KV cache 837k tokens, 25 concurrent 32k sequences) | **in progress**, 80 steps planned, deadline 2026-11-08 08:03 IST; ~10–15 min/step (rollout 340–640 s) |

Run 3 on held-out ITBench-Lite, as of step 24 (2026-11-08 06:13 IST):

| Eval step | Exact (of 15) | Mean reward | Answered | Hit 20-step cap | Mean steps |
|---|---|---|---|---|---|
| 0 (base) | 4 | 0.333 | 8 | 7 | 16.4 |
| 8 | 6 | 0.400 | 6 | 9 | 15.8 |
| 16 | 6 | 0.444 | 11 | 4 | 13.3 |
| 24 | 6 | 0.489 | 12 | 2 | 12.3 |

On training tasks (48 episodes per step), the answered rate went from 0.58 at step 0 to 0.75–0.92 over steps 21–23,
and mean tool steps from 16.4 to 11–14. Per-step pass rate is noisy (0.06–0.54).

- **What is moving:** the policy learns to stop and answer. Step-cap hits on ITBench-Lite fell from 7 to 2 and
  mean steps from 16.4 to 12.3, the same failure the harness nudges fixed in the eval.
- **What is not yet clear:** exact answers went from 4 to 6 of 15, i.e. two scenarios with one sample each, which is
  within run-to-run noise (the base model scored 3 and 4 of 15 in the step-0 evals of runs 1 and 2). Of the 4 extra
  answers, 2 are right and 2 are wrong.
- **Health checks:** train/vLLM log-prob mismatch stays at ~0.025 nats/token, no OOMs, 0–5 long episodes per step
  skipped from backprop (> 28k tokens), and 0–6 sandbox audit flags per step (logged, not penalized).

## License

MIT, see [LICENSE](LICENSE).
