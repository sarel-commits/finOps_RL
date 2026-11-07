#!/usr/bin/env python
"""Run a benchmark against the local server with checkpointing.

  python scripts/run_bench.py --bench itbench_lite --config nothink_native --seeds 1 2 3
  python scripts/run_bench.py --bench synthetic   --config nothink_native --seeds 1 2 3

Writes results/<bench>/<config>/<item>/seed_<s>/trajectory.json and
results/<bench>/<config>/run_config.json. Scoring is separate (score_*.py).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from agent import Agent, ROOT

ITB = ROOT / "data/itbench-lite/snapshots/finops/v0.1-finops-anomaly"
SYN = ROOT / "data/synthetic"

ITB_PROMPT = """A cost anomaly was detected in a cloud billing dataset. Your job is root cause analysis: identify the resource(s) responsible for the anomalous cost change.

Anomaly details (from anomaly.json, also in your working directory):
{anomaly}

The file data.csv in your working directory contains the daily cost data for the accounts. Investigate it with code (inspect the columns first) and determine which resource(s) caused the anomalous cost for this account on this date. A resource is identified by its name (the exact value as it appears in data.csv) and its type (the data.csv column that value comes from, written in title case with spaces instead of underscores, e.g. a value from a column named some_column has type "Some Column").

Required final answer format (JSON inside <answer></answer> tags):
{{"resource": [{{"name": "<resource name>", "type": "<resource type>"}}]}}
List every resource that is a root cause; include only root causes."""

ITB_METHOD_V2 = """Recommended method (aim to finish in about 6-8 tool calls):
1. Inspect the columns and filter to the anomalous account. Compare the anomaly date with a baseline of the preceding normal days.
2. For EVERY categorical column (not only one), compute the cost change per value between the baseline and the anomaly date. Do this for all columns in a single tool call.
3. Decide which column is the root-cause dimension: the one whose values explain the total cost change most fully and most specifically. A change in one column often shows up indirectly in other columns, so compare the columns before choosing; do not assume a particular column.
4. The root cause can be one value or many values of that column. Include every value that materially drives the change, and nothing else.
5. Answer.

"""
_SPLIT = "Required final answer format"
ITB_PROMPT_V2 = ITB_PROMPT.replace(_SPLIT, ITB_METHOD_V2 + _SPLIT, 1)
assert ITB_PROMPT_V2.count(_SPLIT) == 1 and ITB_PROMPT_V2 != ITB_PROMPT


def items(bench, prompt_version="v1"):
    tmpl = ITB_PROMPT_V2 if prompt_version == "v2" else ITB_PROMPT
    if bench == "itbench_lite":
        for d in sorted(ITB.iterdir(), key=lambda p: int(p.name.split("-")[1])):
            anomaly = (d / "anomaly.json").read_text().strip()
            yield d.name, tmpl.format(anomaly=anomaly), [d / "anomaly.json", d / "data.csv"]
    else:
        for line in open(SYN / "tasks.jsonl"):
            t = json.loads(line)
            files = [SYN / "workspaces" / t["id"] / f for f in t["files"]]
            prompt = t["prompt"]
            if files:
                prompt += "\n\nData files in your working directory: " + ", ".join(t["files"]) + "."
            else:
                prompt += "\n\n(No data files are needed for this task; you may still use Python for the arithmetic.)"
            prompt += f"\n\nRequired final answer format (JSON inside <answer></answer> tags):\n{t['answer_format']}"
            yield t["id"], prompt, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["itbench_lite", "synthetic"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--protocol", default="native", choices=["native", "react"])
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--model-label", default="qwen3.8-27b-mlx-8bit")
    ap.add_argument("--budget-nudges", action="store_true",
                    help="step counter + 5-left warning in observations; forced no-tools final turn at budget end")
    ap.add_argument("--prompt-version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="default_model")
    ap.add_argument("--api-key-env", default=None, help="name of env var holding the API key (value never logged)")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sampling-profile", default=None, choices=["nothink", "think", "provider_nothink", "provider_think"])
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "xhigh"],
                    help="Qwen3.8 chat-template reasoning_effort (thinking mode only); template default is xhigh")
    ap.add_argument("--repeat-guard", action="store_true",
                    help="append a note to the observation when code is >=90%% similar to an earlier call")
    ap.add_argument("--only-pairs", nargs="*", default=None, help="exact item:seed pairs to run")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                    help="anthropic = Claude via the official SDK (scripts/anthropic_chat.py)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude output_config.effort (adaptive thinking)")
    a = ap.parse_args()
    out = ROOT / "results" / a.bench / a.config
    out.mkdir(parents=True, exist_ok=True)
    agent = Agent(protocol=a.protocol, thinking=a.thinking, max_steps=a.max_steps, max_tokens=a.max_tokens,
                  budget_nudges=a.budget_nudges, prompt_version=a.prompt_version,
                  base_url=a.base_url, model=a.model, api_key_env=a.api_key_env, stream=a.stream, retries=a.retries,
                  sampling_profile=a.sampling_profile, reasoning_effort=a.reasoning_effort,
                  repeat_guard=a.repeat_guard, provider=a.provider, effort=a.effort)
    cfg = dict(bench=a.bench, config=a.config, seeds=a.seeds, model_label=a.model_label, agent=agent.config(),
               itbench_prompt_template=(ITB_PROMPT_V2 if a.prompt_version == "v2" else ITB_PROMPT) if a.bench == "itbench_lite" else None,
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1))
    todo = [(iid, p, f) for iid, p, f in items(a.bench, a.prompt_version) if not a.only or iid in a.only]
    pairs = {(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in a.only_pairs} if a.only_pairs else None
    if pairs:
        cfg["only_pairs"] = sorted(f"{i}:{s}" for i, s in pairs)
        (out / "run_config.json").write_text(json.dumps(cfg, indent=1))
    for seed in a.seeds:
        for iid, prompt, files in todo:
            if pairs and (iid, seed) not in pairs:
                continue
            t0 = time.time()
            tr = agent.run(prompt, files, out / iid / f"seed_{seed}", seed=seed * 1000,
                           meta=dict(item=iid, bench=a.bench, config=a.config, model=a.model_label))
            print(f"[{a.config}] seed={seed} {iid}: status={tr['status']} err={tr.get('error')} "
                  f"steps={tr.get('n_steps')} forced={tr.get('forced_final', False)} ans={json.dumps(tr.get('answer'))[:120]} "
                  f"tok_in={tr.get('tokens_in')} tok_out={tr.get('tokens_out')} cost=${tr.get('cost_usd')} wall={tr.get('wall_s')}s "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
