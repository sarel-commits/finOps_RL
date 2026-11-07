# finops-rca: RL environment for FinOps cost-anomaly root-cause analysis

RLVR environment for training an agent (target: Qwen3.8-27B) on ITBench-Lite-style FinOps RCA:
given `anomaly.json` + `data.csv` and one sandboxed `run_python` tool, find the resource(s) that caused the cost
anomaly. Reward is a deterministic verifier — no LLM judge.

```
rl/
  finops_gym/            framework-agnostic core
    generator.py         procedural task generator (seeded, deterministic, label-validated)
    oracle.py            reference detector = the task semantics, made executable
    reward.py            verifier / reward (F1 + exact bonus, format gate)
    sandbox.py           run_python isolation (bwrap on Linux, sandbox-exec on macOS)
    episode.py           reference episode loop vs any OpenAI-compatible endpoint
    prompts.py           verbatim copies of the eval prompts (tested for parity)
    tasks.py             manifests + split hygiene
  finops_rca/            verifiers v1 taskset (prime-rl / vf-run / vf-validate): id "finops-rca"
  scripts/
    build_dataset.py     write manifests for train/val/test_synth + itbench_lite (eval only)
    calibrate.py         pass@k of the current policy per task -> GRPO-informative subset
    validate_oracle.py   check the oracle against the real ITBench-Lite labels
    export_harbor.py     export eval tasks as Harbor tasks (benchmark other agents)
  configs/prime-rl/finops-rca-27b-lora.toml   one-node 8xH200 LoRA-GRPO config
  data/                  manifests (seed + label + data hash per task; CSVs are regenerated, ~1.4 MB total)
  harbor/                exported Harbor tasks (itbench_lite: 15, test_synth: 50)
  tests/                 35 tests (reward, determinism, label validity, contamination, sandbox, prompt parity)
```

## Quick start

```bash
# venv with verifiers 0.4.1 + this package (already created here as rl/.venv-vf)
uv venv rl/.venv-vf --python 3.12 && VIRTUAL_ENV=rl/.venv-vf uv pip install -e "rl[verifiers,dev]"
rl/.venv-vf/bin/python -m pytest rl/tests -q

python rl/scripts/build_dataset.py --train 2000 --val 200 --test-synth 300
rl/.venv-vf/bin/vf-validate finops-rca --taskset.split val -n 3 --runtime.type subprocess   # gold=1.0, noop=0
python rl/scripts/calibrate.py --split train --n-tasks 200 --k 8 --base-url http://<vllm>:8000/v1 --workers 64
```

Training (prime-rl, on the GPU node; see the config header for the fields still to verify):
```bash
uv run rl @ rl/configs/prime-rl/finops-rca-27b-lora.toml
```

## Task, tool, reward

* **Prompt / tool / budget** are byte-identical to the baseline eval (`scripts/agent.py`, `scripts/run_bench.py`,
  v1 prompt, 20 tool steps, answer in `<answer>{"resource": [...]}</answer>`). Enforced by `tests/test_env.py`.
* **Reward** (`finops_gym/reward.py`): `0.5 * F1 + 0.5 * exact_set_match` over normalized (name, type) pairs;
  0 when there is no parseable `<answer>`. The official ITBench metric passes a *superset* of the labels; as an RL
  reward that teaches listing extra resources, so extras are penalized here (the official metric is still logged).
  Optional `step_cost` (off) only applies to exact answers.
* **Sandbox**: the policy is optimized against the reward, so label access is the obvious hack. The tool can read
  only its working directory (anomaly.json, data.csv) and the Python install; no network; writes only to the
  workdir. Verified: reading `rl/data/*.jsonl`, ITBench `ground_truth.yaml`, `$HOME`, writing outside, and HTTP
  all fail. A static `audit_code` tripwire logs suspicious code as a metric. `backend="none"` exists for
  debugging only and marks every output.

## Generator

Every task = `(seed, GenConfig)`; `data.csv` is regenerated at rollout time and checked against the manifest's
SHA-256 (fails loudly on generator/library drift; verified identical across two pandas versions here).
Schema, hierarchy (csp > cloud_service > application > instance_family), `cost = price * n * 24` and row counts
(~14k) match ITBench-Lite. Two regimes, mirroring what ITBench-Lite contains (see findings below):

| regime | share | root cause | label check |
|---|---|---|---|
| clean | 70% | persistent level shift on the date: count or price shift, several families, app-wide shift, new app(s), new family | `oracle.detect` must reproduce it exactly (else resampled) |
| noisy | 30% | volatile baselines; the largest increase vs the previous week, onset 0-3 days before the date | must be top-ranked with >=1.5x margin |

Clean-regime distractors (1-2 day bursts, volatile families at a peak, drops, old step changes) are built so the
naive "largest delta" shortcut fails on most clean tasks.

Train split (2000 tasks): 1637 single-cause, 363 multi-cause (up to 20); 1534 Instance Family / 466 Application
labels; naive top-delta solves 30% of clean / 96% of noisy tasks (~50% overall; it solves 60% of ITBench-Lite).

## Eval data findings (ITBench-Lite FinOps)

* **Two labelling regimes.** S1-S10: a clean persistent shift; the oracle reproduces 9/10 (S9 adds one family the
  label omits). S11-S15: noisy series where the label is the largest recent increase, which in S11/S13/S15 began
  2-3 days *before* the reported date and in S12 is short-lived. No single "persistent shift on the date" rule
  explains all 15 labels; the generator therefore produces both regimes.
* **Name case.** In S6, 16 of 20 label names differ from `data.csv` only by letter case (`9ac` vs `9aC`). Our scorer
  is case-insensitive, so S6 is solvable here; a case-sensitive scorer would make it unsolvable.
* The naive top-delta heuristic solves 9/15 (60%) — the reference number in `REPORT.md`.

## Frameworks

* **prime-rl + verifiers (chosen).** prime-rl lists dense Qwen3.5 (`qwen3_5`, our model's architecture) as
  supported, including LoRA with a frozen vision encoder; its `wiki-search` example uses the same pattern as
  this environment (verifiers taskset + `null` harness + `subprocess` runtime + LoRA). Note: Qwen3.5 is hybrid
  (48/64 layers GatedDeltaNet); the config adds `in_proj_qkv`, `in_proj_z`, `out_proj` to the LoRA targets because
  unknown target names are silently ignored and the defaults would skip those layers.
* **Harbor** is an eval harness (from the Terminal-Bench authors) with no RL loop; verifiers can load Harbor
  tasks (`HarborTaskset`). `export_harbor.py` produces Harbor tasks so other agents/models can be benchmarked on
  the same eval set. The local verifier check passes (gold 1.0, wrong 0.0); not run under Docker here (no Docker).

## Not done / to verify on the GPU node

* The prime-rl TOML has not been schema-validated (installing `prime-rl-configs` was not permitted in this session);
  fields marked `VERIFY` are assumptions (optimizer type name, renderer name/thinking flag, model id).
* `bwrap` backend is written but untested here (macOS); test it on the node before training.
* Calibration so far: a 4-task x 2-sample smoke run on the local 8-bit model (`results/calibration/train/`):
  2/8 exact, 2/4 tasks informative (0 < pass < 1), 3/8 rollouts hit the 20-step budget without answering,
  no audit flags, ~13 min/rollout.
  A real calibration (>=200 tasks x 8) should run on vLLM on the node.
* Generator labels are self-consistent by construction; whether training on them transfers to ITBench-Lite is the
  experiment itself (eval source `itbench-lite` in the config, every 10 steps).
