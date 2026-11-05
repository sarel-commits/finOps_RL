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

The file data.csv in your working directory contains the daily cost data for the accounts. Investigate it with code (inspect the columns first) and determine which resource(s) caused the anomalous cost for this account on this date. A resource is identified by its name (the value in the data) and its type (the dimension/column it belongs to, written in title case, e.g. "Cloud Service").

Required final answer format (JSON inside <answer></answer> tags):
{{"resource": [{{"name": "<resource name>", "type": "<resource type>"}}]}}
List every resource that is a root cause; include only root causes."""


def items(bench):
    if bench == "itbench_lite":
        for d in sorted(ITB.iterdir(), key=lambda p: int(p.name.split("-")[1])):
            anomaly = (d / "anomaly.json").read_text().strip()
            yield d.name, ITB_PROMPT.format(anomaly=anomaly), [d / "anomaly.json", d / "data.csv"]
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
    a = ap.parse_args()
    out = ROOT / "results" / a.bench / a.config
    out.mkdir(parents=True, exist_ok=True)
    agent = Agent(protocol=a.protocol, thinking=a.thinking, max_steps=a.max_steps, max_tokens=a.max_tokens)
    cfg = dict(bench=a.bench, config=a.config, seeds=a.seeds, model_label=a.model_label, agent=agent.config(),
               itbench_prompt_template=ITB_PROMPT if a.bench == "itbench_lite" else None,
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1))
    todo = [(iid, p, f) for iid, p, f in items(a.bench) if not a.only or iid in a.only]
    for seed in a.seeds:
        for iid, prompt, files in todo:
            t0 = time.time()
            tr = agent.run(prompt, files, out / iid / f"seed_{seed}", seed=seed * 1000,
                           meta=dict(item=iid, bench=a.bench, config=a.config, model=a.model_label))
            print(f"[{a.config}] seed={seed} {iid}: status={tr['status']} err={tr.get('error')} "
                  f"steps={tr.get('n_steps')} ans={json.dumps(tr.get('answer'))[:120]} "
                  f"tok_in={tr.get('tokens_in')} tok_out={tr.get('tokens_out')} wall={tr.get('wall_s')}s "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
