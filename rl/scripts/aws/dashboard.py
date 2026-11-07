#!/usr/bin/env python
"""Live local dashboard for the remote calibration run. Nothing leaves this Mac except the ssh to the instance.

  python rl/scripts/aws/dashboard.py            # then open http://127.0.0.1:8765

Every --interval seconds: one ssh to the instance (rl/.aws-run/ssh_config, host "calib") that returns a compact view
of the rollouts files (messages and labels stripped) + a vLLM /metrics snapshot. Time series are kept locally in
rl/results/dashboard/state.json (rollout completion times are the time this script first saw them, ~interval
resolution). The page (dashboard.html next to this file) polls data.json.
"""
import argparse, json, re, subprocess, threading, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RL = Path(__file__).resolve().parents[2]
OUT = RL / "results" / "dashboard"
SSH = ["ssh", "-F", str(RL / ".aws-run" / "ssh_config"), "-o", "BatchMode=yes", "calib"]

REMOTE = r'''
import json, os, glob, subprocess
out = {"rollouts": {}, "log": ""}
for f in glob.glob(os.path.expanduser("~/rl/results/calibration/*/rollouts.jsonl")):
    split = f.split("/")[-2]; rows = []
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        w = r["reward"]
        rows.append([r["task_id"], r["sample"], r.get("regime"), r.get("mechanism"), w["exact"], round(w["f1"], 3),
                     w["format_ok"], r["steps"], r["error"], len(r["audit_flags"]), r["tokens_in"], r["tokens_out"], r["wall_s"]])
    out["rollouts"][split] = rows
try: out["log"] = subprocess.run(["tail", "-n", "4", os.path.expanduser("~/run.log")], capture_output=True, text=True).stdout
except Exception: pass
try:
    import urllib.request
    out["metrics"] = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5).read().decode()
except Exception: out["metrics"] = ""
try: out["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
except Exception: out["gpu"] = ""
def jl(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except Exception: return []
td = os.path.expanduser(TRAIN_DIR)
out["train_metrics"] = jl(os.path.join(td, "metrics.jsonl"))
agg = {}
for r in jl(os.path.join(td, "eval.jsonl")):
    a = agg.setdefault(r["step"], [0, 0, 0, 0]); a[0] += 1; a[1] += r["exact"]; a[2] += r["reward"]; a[3] += r["format_ok"]
out["train_eval"] = [dict(step=s, n=a[0], pass_rate=a[1] / a[0], reward=a[2] / a[0], answered=a[3] / a[0]) for s, a in sorted(agg.items())]
try: out["train_log"] = subprocess.run(["tail", "-n", "6", os.path.expanduser(TRAIN_LOG)], capture_output=True, text=True).stdout
except Exception: out["train_log"] = ""
print(json.dumps(out))
'''

FIELDS = ["task_id", "sample", "regime", "mechanism", "exact", "f1", "format_ok", "steps", "error", "audit", "tok_in", "tok_out", "wall_s"]


def metric(text, name):
    tot, found = 0.0, False
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                tot += float(line.rsplit(" ", 1)[1]); found = True
            except ValueError:
                pass
    return tot if found else None


def poll(state, train_dir, train_log):
    script = f"TRAIN_DIR = {train_dir!r}\nTRAIN_LOG = {train_log!r}\n" + REMOTE
    p = subprocess.run(SSH + ["python3", "-"], input=script, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        state["status"] = f"ssh failed: {p.stderr.strip()[-200:]}"
        return
    d = json.loads(p.stdout)
    now = time.time()
    seen = state.setdefault("seen", {})
    for split, rows in d["rollouts"].items():
        for r in rows:
            k = f"{split}|{r[0]}|{r[1]}"
            if k not in seen:
                seen[k] = now
    rollouts = {s: [dict(zip(FIELDS, r), t=seen[f"{s}|{r[0]}|{r[1]}"]) for r in rows] for s, rows in d["rollouts"].items()}
    m = d.get("metrics", "")
    gen = metric(m, "vllm:generation_tokens_total"); prm = metric(m, "vllm:prompt_tokens_total")
    pt = dict(t=now, running=metric(m, "vllm:num_requests_running"), waiting=metric(m, "vllm:num_requests_waiting"),
              kv=metric(m, "vllm:kv_cache_usage_perc") if metric(m, "vllm:kv_cache_usage_perc") is not None
              else metric(m, "vllm:gpu_cache_usage_perc"), gen_total=gen, prompt_total=prm,
              hit=metric(m, "vllm:prefix_cache_hits_total"), queries=metric(m, "vllm:prefix_cache_queries_total"))
    gpu = d.get("gpu", "")
    if gpu:
        u, used, total = [float(x) for x in re.split(r",\s*", gpu)]
        pt.update(gpu_util=u, gpu_mem=used / total * 100)
    series = state.setdefault("series", [])
    if series and gen is not None and series[-1].get("gen_total") is not None:
        dt = now - series[-1]["t"]
        pt["gen_tps"] = (gen - series[-1]["gen_total"]) / dt
        pt["prompt_tps"] = (prm - series[-1]["prompt_total"]) / dt if prm is not None and series[-1].get("prompt_total") is not None else None
    series.append(pt)
    del series[:-2000]
    state["status"] = "ok" if m else "vLLM not serving yet"
    state["log"] = d.get("log", "")
    state["updated"] = now
    (OUT / "data.json").write_text(json.dumps(dict(rollouts=rollouts, series=series, status=state["status"],
                                                   log=state["log"], updated=now, total={"train": 1600, "itbench_lite": 120},
                                                   train_metrics=d.get("train_metrics", []), train_eval=d.get("train_eval", []),
                                                   train_log=d.get("train_log", ""))))
    (OUT / "state.json").write_text(json.dumps({k: state[k] for k in ("seen", "series")}))


def loop(interval, train_dir, train_log):
    state = json.loads((OUT / "state.json").read_text()) if (OUT / "state.json").exists() else {}
    while True:
        try:
            poll(state, train_dir, train_log)
        except Exception as e:  # keep polling through transient ssh/json errors
            state["status"] = f"poll error: {e}"
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--train-dir", default="~/rl/results/train", help="trainer output dir on the instance")
    ap.add_argument("--train-log", default="~/train.log", help="trainer stdout log on the instance")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text((Path(__file__).parent / "dashboard.html").read_text())
    threading.Thread(target=loop, args=(a.interval, a.train_dir, a.train_log), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), partial(SimpleHTTPRequestHandler, directory=str(OUT)))
    print(f"dashboard: http://127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
