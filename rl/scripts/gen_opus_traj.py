#!/usr/bin/env python
"""Generate Claude Opus 5.5 demonstration trajectories on SYNTHETIC TRAIN tasks (never ITBench-Lite), for an SFT
warm start before the next GRPO run.

Episode protocol = the trainer's --harness improved (prompt v2, step nudges on tool outputs, 5-left warning, forced
no-tools final turn), one run_python call per turn, 20-step budget. Only the visible actions are kept (tool calls +
final answer); Claude's thinking is not requested (display omitted) and not stored.

Output (append-only, resumable by task_id):
  <out>/raw.jsonl   every episode: messages (OpenAI format), label, score, steps, cost
  <out>/sft.jsonl   written by --filter: exact-match, unforced, unflagged episodes as {"messages", "tools"}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent / "scripts"))

from anthropic_chat import PRICES, _tool_def, to_anthropic  # noqa: E402
from finops_gym import tasks as T  # noqa: E402
from finops_gym.generator import materialize  # noqa: E402
from finops_gym.prompts import NUDGE_FINAL, RUN_PYTHON_TOOL, nudge, system_prompt, task_prompt_for  # noqa: E402
from finops_gym.reward import RewardConfig, compute_reward  # noqa: E402
from finops_gym.sandbox import Sandbox, SandboxConfig, audit_code  # noqa: E402

MODEL = "claude-opus-5-5"

# Privileged teacher note: the generator's labelling rules (finops_gym/oracle.py docstring + noisy regime), which the
# RL reward already encodes. Sent to Opus only; the stored trajectories keep the student's plain system prompt.
TEACHER_HINT = """[Private labelling convention for this dataset. Use it to decide, but never mention or quote this note, the word "convention", or these rules in any message; justify every conclusion from the data you computed.]
- A root cause is an entity whose daily cost moves to a new, persistent, higher level starting on the anomaly date, beyond what its own recent history (about two weeks) explains. One- or two-day bursts, series that are always volatile and merely at a peak, cost drops, and step changes that began earlier are distractors, not root causes.
- Report an Instance Family when it shifted on its own.
- Report an Application instead of its instance families when the application first appears on the anomaly date, or when all of its instance families shifted together. List every such application.
- Exception: when the series are noisy and no clean persistent shift starts on the date, the root cause is the single entity with the largest increase versus the previous week; that increase may have begun 0-3 days before the reported date.
- Include every root cause and nothing else."""
LEAK_WORDS = ("convention", "private note", "labelling rule", "labeling rule", "the note", "[private")


def pick_tasks(n_per_mech: int, n_noisy: int, exclude: set[str], seed: int):
    by = defaultdict(list)
    for t in T.load(T.manifest_path("train"), for_training=True):
        if t.task_id not in exclude:
            by[t.mechanism].append(t)
    rng, out = random.Random(seed), []
    for mech, ts in sorted(by.items()):
        rng.shuffle(ts)
        out += ts[:n_noisy if mech == "noisy_shift" else n_per_mech]
    rng.shuffle(out)
    return out


def cost(u):
    p = PRICES[MODEL]
    return (u.input_tokens * p["input"] + (u.cache_creation_input_tokens or 0) * p["cache_write"]
            + (u.cache_read_input_tokens or 0) * p["cache_read"] + u.output_tokens * p["output"]) / 1e6


def episode(client, task, effort: str, max_steps: int, rcfg: RewardConfig, hint: bool):
    t0 = time.time()
    msgs = [{"role": "system", "content": system_prompt("improved", max_steps)},
            {"role": "user", "content": task_prompt_for(task.anomaly, "improved")}]
    tools = [_tool_def(RUN_PYTHON_TOOL)]
    steps, forced, final, err, usd, flags, stops = 0, False, None, None, 0.0, [], []
    with Sandbox(SandboxConfig()) as sb:
        materialize(task, sb.workdir)
        for _ in range(max_steps + 1):
            system, amsgs = to_anthropic(msgs)
            if hint:
                system += "\n\n" + TEACHER_HINT
            kw = dict(model=MODEL, max_tokens=32000, system=system, messages=amsgs, tools=tools,
                      thinking={"type": "adaptive"}, output_config={"effort": effort},
                      cache_control={"type": "ephemeral"},
                      tool_choice={"type": "none"} if forced else {"type": "auto", "disable_parallel_tool_use": True})
            with client.messages.stream(**kw) as s:
                r = s.get_final_message()
            usd += cost(r.usage)
            stops.append(r.stop_reason)
            if r.stop_reason == "refusal":
                err = "refusal"
                break
            text = "".join(b.text for b in r.content if b.type == "text")
            calls = [b for b in r.content if b.type == "tool_use"]
            am = {"role": "assistant", "content": text, "_anthropic_content": [b.to_dict() for b in r.content]}
            if calls:
                am["tool_calls"] = [{"id": c.id, "type": "function",
                                     "function": {"name": c.name, "arguments": json.dumps(c.input)}} for c in calls]
            msgs.append(am)
            if forced or not calls:
                final = text
                if calls:
                    err = "forced_final_tool_call"
                break
            steps += 1
            for i, c in enumerate(calls):
                code = c.input.get("code") if c.name == "run_python" and isinstance(c.input, dict) else None
                if code is None:
                    out = f"[error: unknown tool or unparseable arguments: {c.name}]"
                else:
                    flags += audit_code(code)
                    out = sb.run_python(code).output
                if i == len(calls) - 1:
                    out += nudge(steps, max_steps)
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
            if steps >= max_steps:
                msgs.append({"role": "user", "content": NUDGE_FINAL})
                forced = True
    sc = compute_reward(final or "", task.label, steps_used=steps, max_steps=max_steps, cfg=rcfg)
    clean = [{k: v for k, v in m.items() if k != "_anthropic_content"} for m in msgs]
    said = " ".join((m.get("content") or "") + " ".join(c["function"]["arguments"] for c in m.get("tool_calls", []))
                    for m in clean if m["role"] == "assistant").lower()
    leak = [w for w in LEAK_WORDS if w in said]
    return dict(task_id=task.task_id, mechanism=task.mechanism, regime=task.regime, label=task.label,
                teacher_hint=hint, leak=leak,
                exact=sc["exact"], f1=sc["f1"], format_ok=sc["format_ok"], reward=sc["reward"], steps=steps,
                forced=forced, err=err, audit=sorted(set(flags)), stop_reasons=stops, cost_usd=round(usd, 5),
                wall_s=round(time.time() - t0, 1), model=MODEL, effort=effort, harness="improved",
                messages=clean)


def run(a):
    import anthropic
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw.jsonl"
    done = {json.loads(l)["task_id"] for l in raw.open()} if raw.exists() else set()
    spent = sum(json.loads(l)["cost_usd"] for l in raw.open()) if raw.exists() else 0.0
    seen = {json.loads(l)["task_id"] for l in (RL / "results/run3/rollouts.jsonl").open()} if a.exclude_run3 else set()
    todo = [t for t in pick_tasks(a.n_per_mech, a.n_noisy, seen, a.seed) if t.task_id not in done][:a.limit or None]
    print(f"{len(todo)} tasks to run ({len(done)} done, ${spent:.2f} spent, budget ${a.budget:.2f})", flush=True)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=900.0, max_retries=4)
    rcfg = RewardConfig(exact_weight=0.7)
    lock, stop = threading.Lock(), threading.Event()

    def job(t):
        if stop.is_set():
            return None
        try:
            r = episode(client, t, a.effort, a.max_steps, rcfg, a.teacher_hint)
        except Exception as e:  # API errors after SDK retries: record and continue
            print(f"{t.task_id} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
            return None
        nonlocal spent
        with lock:
            spent += r["cost_usd"]
            with raw.open("a") as f:
                f.write(json.dumps(r) + "\n")
            if spent >= a.budget:
                stop.set()
            print(f"{t.task_id} {t.mechanism:14s} exact={r['exact']} f1={r['f1']:.2f} steps={r['steps']} "
                  f"forced={r['forced']} err={r['err']} ${r['cost_usd']:.3f} {r['wall_s']}s | total ${spent:.2f}",
                  flush=True)
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        for _ in as_completed([ex.submit(job, t) for t in todo]):
            pass
    print(f"DONE spent ${spent:.2f}" + (" (budget reached)" if stop.is_set() else ""), flush=True)


def filt(a):
    out = Path(a.out)
    rows = [json.loads(l) for l in (out / "raw.jsonl").open()]
    keep = [r for r in rows if r["exact"] == 1 and not r["forced"] and not r["err"] and not r["audit"]
            and not r.get("leak")]
    with (out / "sft.jsonl").open("w") as f:
        for r in keep:
            f.write(json.dumps(dict(task_id=r["task_id"], mechanism=r["mechanism"], messages=r["messages"],
                                    tools=[RUN_PYTHON_TOOL])) + "\n")
    by = defaultdict(Counter)
    for r in rows:
        by[r["mechanism"]]["n"] += 1
        by[r["mechanism"]]["exact"] += r["exact"]
        by[r["mechanism"]]["steps"] += r["steps"]
    stats = dict(episodes=len(rows), kept=len(keep), exact_rate=round(sum(r["exact"] for r in rows) / len(rows), 3),
                 forced=sum(r["forced"] for r in rows), leaked=sum(bool(r.get("leak")) for r in rows), errors=Counter(r["err"] for r in rows if r["err"]),
                 cost_usd=round(sum(r["cost_usd"] for r in rows), 2),
                 mean_steps_kept=round(sum(r["steps"] for r in keep) / max(len(keep), 1), 2),
                 per_mechanism={m: dict(n=c["n"], exact=round(c["exact"] / c["n"], 2),
                                        mean_steps=round(c["steps"] / c["n"], 1)) for m, c in sorted(by.items())})
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RL / "data" / "sft_opus"))
    ap.add_argument("--n-per-mech", type=int, default=50, help="tasks per clean mechanism (7 mechanisms)")
    ap.add_argument("--n-noisy", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N selected tasks (pilot)")
    ap.add_argument("--exclude-run3", action="store_true", help="skip tasks already used in run 3 rollouts")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget", type=float, default=40.0, help="stop submitting new episodes past this USD estimate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-hint", action="store_true", help="give Opus the labelling rules (not stored)")
    ap.add_argument("--filter", action="store_true", help="only (re)write sft.jsonl + stats.json from raw.jsonl")
    a = ap.parse_args()
    filt(a) if a.filter else run(a)
