#!/usr/bin/env python
"""Difficulty calibration: run the current policy k times per task, keep tasks with 0 < pass-rate < 1.

  python rl/scripts/calibrate.py --split train --n-tasks 100 --k 8 --base-url http://127.0.0.1:8080/v1
  python rl/scripts/calibrate.py --split train --summarize           # only (re)write the summary/filtered manifest

Resumable: each (task, sample) appends one line to rl/results/calibration/<split>/rollouts.jsonl (full
messages included, for inspection). Pass = reward.exact == 1 (the strict set match the RL reward targets).
Outputs: summary.json and rl/data/<split>_calibrated.jsonl (the GRPO-informative subset).
GRPO learns nothing from a task whose k samples all get the same reward, so the filtered subset is what to train on;
always-solved tasks can be dropped, always-failed tasks are kept aside for a later curriculum stage.
"""
import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finops_gym import tasks as T  # noqa: E402
from finops_gym.episode import EpisodeConfig, run_episode  # noqa: E402

OUT = T.RL_ROOT / "results" / "calibration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default=None, help="default: first model the server lists")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--workers", type=int, default=1, help="1 for mlx_lm.server; 32+ for vLLM")
    ap.add_argument("--sample-seed", type=int, default=0, help="which tasks to pick from the split")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    roll = out / "rollouts.jsonl"
    tasks = T.load(T.manifest_path(a.split))
    random.Random(a.sample_seed).shuffle(tasks)
    tasks = tasks[: a.n_tasks]

    if not a.summarize:
        import os
        import openai
        client = openai.OpenAI(base_url=a.base_url, api_key=os.environ[a.api_key_env] if a.api_key_env else "local",
                               timeout=1800, max_retries=2)
        model = a.model or client.models.list().data[0].id
        done = set()
        if roll.exists():
            for line in roll.read_text().splitlines():
                r = json.loads(line)
                done.add((r["task_id"], r["sample"]))
        todo = [(t, s) for t in tasks for s in range(a.k) if (t.task_id, s) not in done]
        print(f"{len(todo)} rollouts to run ({len(done)} already done), model={model}", flush=True)

        def one(ts):
            t, s = ts
            r = run_episode(client, model, t, EpisodeConfig(), seed=s + 1)
            r.update(sample=s, regime=t.regime, mechanism=t.mechanism)
            with open(roll, "a") as f:
                f.write(json.dumps(r) + "\n")
            rw = r["reward"]
            print(f"{t.task_id} s{s} {t.regime}/{t.mechanism} exact={rw['exact']} f1={rw['f1']:.2f} "
                  f"steps={r['steps']} err={r['error']} flags={r['audit_flags']} {r['wall_s']}s", flush=True)

        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, todo))
        print(f"done in {time.time() - t0:.0f}s")

    rows = [json.loads(l) for l in roll.read_text().splitlines()] if roll.exists() else []
    by = {}
    for r in rows:
        by.setdefault(r["task_id"], []).append(r)
    per = {tid: dict(n=len(v), pass_rate=sum(x["reward"]["exact"] for x in v) / len(v),
                     mean_reward=sum(x["reward"]["reward"] for x in v) / len(v),
                     answered=sum(x["reward"]["format_ok"] for x in v) / len(v),
                     regime=v[0]["regime"], mechanism=v[0]["mechanism"]) for tid, v in by.items()}
    full = {k: v for k, v in per.items() if v["n"] >= a.k}
    inf = [k for k, v in full.items() if 0 < v["pass_rate"] < 1]
    summ = dict(split=a.split, k=a.k, tasks_complete=len(full), rollouts=len(rows),
                informative=len(inf), always_pass=sum(v["pass_rate"] == 1 for v in full.values()),
                always_fail=sum(v["pass_rate"] == 0 for v in full.values()),
                mean_pass=sum(v["pass_rate"] for v in full.values()) / max(1, len(full)),
                answered=sum(v["answered"] for v in full.values()) / max(1, len(full)),
                audit_flagged_rollouts=sum(bool(r["audit_flags"]) for r in rows), per_task=per)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    keep = set(inf)
    T.save([t for t in T.load(T.manifest_path(a.split)) if t.task_id in keep],
           T.manifest_path(a.split, f"{a.split}_calibrated"))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_task"}))


if __name__ == "__main__":
    main()
