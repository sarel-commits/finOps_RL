#!/usr/bin/env python
"""Live local dashboard for the remote training run (run 4: run 3 LoRA -> SFT on Opus demos -> GRPO).
Nothing leaves this Mac except the ssh to the instance.

  python rl/scripts/aws/dashboard.py            # then open http://127.0.0.1:8765

Every --interval seconds: one ssh to the instance (rl/.aws-run/ssh_config, host "calib") that returns compact
aggregates of the trainer's output files. Earlier runs (run 3, run 2), the Opus demonstration set and the local
benchmark scores are read from local files. The page (dashboard.html next to this file) polls data.json.
"""
import argparse, json, subprocess, threading, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RL = Path(__file__).resolve().parents[2]
FINOPS = RL.parent
OUT = RL / "results" / "dashboard"
SSH = ["ssh", "-F", str(RL / ".aws-run" / "ssh_config"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "calib"]

REMOTE = r'''
import json, os, subprocess
td = os.path.expanduser(TRAIN_DIR)
def jl(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except Exception: return []
def evagg(rows):
    agg = {}
    for r in rows:
        a = agg.setdefault(r["step"], [0, 0, 0, 0, 0]); a[0] += 1; a[1] += r["exact"]; a[2] += r["reward"]; a[3] += r["format_ok"]; a[4] += r["steps"]
    return [dict(step=s, n=a[0], pass_rate=a[1] / a[0], reward=a[2] / a[0], answered=a[3] / a[0], mean_steps=a[4] / a[0]) for s, a in sorted(agg.items())]
out = dict(train_metrics=jl(os.path.join(td, "metrics.jsonl")), sft=jl(os.path.join(td, "sft_metrics.jsonl")),
           train_eval=evagg(jl(os.path.join(td, "eval.jsonl"))), train_val=evagg(jl(os.path.join(td, "eval_val.jsonl"))))
mech = {}
for r in jl(os.path.join(td, "rollouts.jsonl")):
    if r.get("kind") == "train":
        m = mech.setdefault(r["mechanism"], [0, 0]); m[0] += 1; m[1] += r["exact"]
out["mech"] = {k: dict(n=v[0], exact=v[1] / v[0]) for k, v in mech.items()}
try: out["train_log"] = subprocess.run(["tail", "-n", "8", os.path.expanduser(TRAIN_LOG)], capture_output=True, text=True).stdout
except Exception: out["train_log"] = ""
try: out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
except Exception: out["gpu"] = ""
print(json.dumps(out))
'''

# Earlier experiments shown next to the live run (local files).
COMPARE = [("run 3", RL / "results" / "run3"), ("run 2", RL / "results" / "aws_train")]
OPUS = RL / "data" / "sft_opus" / "stats.json"
BENCH = [("base · original harness", "nothink_native"), ("base · improved harness", "nothink_nudges_v2"),
         ("RL run 2 step 32 · improved", "rl32_nothink_nudges_v2"), ("RL run 3 step 25 · improved", "rl3_nothink_nudges_v2"),
         ("RL run 4 · improved", "rl4_nothink_nudges_v2")]


def jl_local(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except OSError:
        return []


def evagg(rows):
    agg = {}
    for r in rows:
        a = agg.setdefault(r["step"], [0, 0, 0, 0])
        a[0] += 1; a[1] += r["exact"]; a[2] += r["format_ok"]; a[3] += r["steps"]
    return [dict(step=s, n=a[0], pass_rate=a[1] / a[0], answered=a[2] / a[0], mean_steps=a[3] / a[0])
            for s, a in sorted(agg.items())]


def load_compare(exact_weight=0.7):
    """Per-step train metrics, evals and per-mechanism train pass of earlier runs. The train reward is rescored with
    the live run's exact weight (reward_w = (1-w)*F1 + w*exact; F1/exact are already 0 when the format fails)."""
    runs = []
    for name, d in COMPARE:
        met = jl_local(d / "metrics.jsonl")
        by, mech = {}, {}
        for r in jl_local(d / "rollouts.jsonl"):
            if r.get("kind") == "train":
                b = by.setdefault(r["step"], [0, 0.0])
                b[0] += 1; b[1] += (1 - exact_weight) * r["f1"] + exact_weight * r["exact"]
                m = mech.setdefault(r["mechanism"], [0, 0]); m[0] += 1; m[1] += r["exact"]
        for m in met:
            if m["step"] in by:
                m["reward_w"] = by[m["step"]][1] / by[m["step"]][0]
        runs.append(dict(name=name, metrics=met, eval=evagg(jl_local(d / "eval.jsonl")),
                         val=evagg(jl_local(d / "eval_val.jsonl")),
                         mech={k: dict(n=v[0], exact=v[1] / v[0]) for k, v in mech.items()}))
    return runs


def load_bench():
    rows = []
    for label, cfg in BENCH:
        try:
            d = json.loads((FINOPS / "results" / "itbench_lite" / cfg / "score_stdout.txt").read_text())
        except (OSError, ValueError):
            continue
        rows.append(dict(label=label, config=cfg, n=d["n_runs"], official=d["mean_acc"], exact=d["strict_set_match_mean"],
                         precision=d["precision_mean"], f1=d["f1_mean"], tokens_in=d["tokens_in_mean"],
                         parse_fail=d["parse_fail"]))
    return rows


def poll(train_dir, train_log):
    script = f"TRAIN_DIR = {train_dir!r}\nTRAIN_LOG = {train_log!r}\n" + REMOTE
    p = subprocess.run(SSH + ["python3", "-"], input=script, capture_output=True, text=True, timeout=90)
    if p.returncode != 0:
        return dict(status=f"ssh failed: {p.stderr.strip()[-200:]}")
    d = json.loads(p.stdout)
    gpu = None
    if d.get("gpu"):
        u, used, total = [float(x) for x in d["gpu"].split(",")]
        gpu = dict(util=u, mem=used / total)
    return dict(d, gpu=gpu, status="ok")


def loop(interval, train_dir, train_log):
    last = {}
    while True:
        try:
            live = poll(train_dir, train_log)
        except Exception as e:  # keep polling through transient ssh/json errors
            live = dict(status=f"poll error: {e}")
        if live["status"] == "ok":
            last = live
        opus = json.loads(OPUS.read_text()) if OPUS.exists() else None
        (OUT / "data.json").write_text(json.dumps(dict(last, status=live["status"], updated=time.time(),
                                                       compare=load_compare(), bench=load_bench(), opus=opus)))
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--train-dir", default="~/rl/results/train", help="trainer output dir on the instance")
    ap.add_argument("--train-log", default="~/rl/results/train.log", help="trainer stdout log on the instance")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text((Path(__file__).parent / "dashboard.html").read_text())
    threading.Thread(target=loop, args=(a.interval, a.train_dir, a.train_log), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), partial(SimpleHTTPRequestHandler, directory=str(OUT)))
    print(f"dashboard: http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
