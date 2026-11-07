#!/usr/bin/env python
"""Cross-model / cross-harness comparison from EXISTING results only (no new runs).
Table A: the 32 runs the local 27B baseline failed (23 ITBench-Lite + 9 synthetic), extracted from every config
that covers them (full-run configs are subset to those pairs). Table B: full ITBench-Lite official score where run.
Cost: Claude = per-call cost recorded in trajectories; Together = tokens x list price; local = $0.
Writes results/analysis/all_models_compare.json."""
import json, statistics as st
from pathlib import Path
RES = Path.home() / "finops-eval" / "results"
PAIRS = {b: {(l.rsplit(":", 1)[0], int(l.rsplit(":", 1)[1])) for l in open(RES / b / "failed_pairs_nothink_native.txt").read().split()}
         for b in ["itbench_lite", "synthetic"]}
PRICE = {"flash": (0.15, 0.47), "glm": (1.4, 4.4), "kimi": (2.0, 10.0)}
ROWS = [  # model, provider/precision, thinking, harness, config, price key
    ("Qwen3.8-27B", "local MLX 8-bit", "off", "original (v1)", "nothink_native", None),
    ("Qwen3.8-27B", "local MLX 8-bit", "off", "nudges + v2", "nothink_nudges_v2", None),
    ("Qwen3.8-27B", "local MLX 8-bit", "off", "nudges + v2 + guard", "nothink_nudges_v2_guard", None),
    ("Qwen3.8-27B", "local MLX 8-bit", "medium", "nudges + v2 + guard", "think_medium_nudges_v2_guard", None),
    ("Qwen3.8-Flash", "Together", "off", "original (v1)", "flash_failedruns", "flash"),
    ("Qwen3.8-Flash", "Together", "off", "original (v1), full run", "flash_nothink_v1", "flash"),
    ("Qwen3.8-Flash", "Together", "off", "nudges + v2", "flash_nudges_v2", "flash"),
    ("Qwen3.8-Flash", "Together", "off", "nudges + v2 + guard", "flash_nudges_v2_guard", "flash"),
    ("Qwen3.8-Flash", "Together", "medium", "nudges + v2 + guard", "flash_think_medium_nudges_v2_guard", "flash"),
    ("GLM-5.3", "Together", "off", "original (v1)", "glm53_failedruns", "glm"),
    ("Kimi-K3.1-Pro", "Together", "off", "original (v1)", "kimik31_failedruns", "kimi"),
    ("Claude Sonnet 5.5", "Anthropic", "medium", "original (v1)", "sonnet55_failedruns", "claude"),
    ("Claude Opus 5.5", "Anthropic", "medium", "original (v1)", "opus55_failedruns", "claude"),
]


def traj(b, cfg, r):
    return json.loads((RES / b / cfg / r["item"] / f"seed_{r['seed']}" / "trajectory.json").read_text())


def subset(b, cfg, pk):
    f = RES / b / cfg / "scores.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if (r["item"], r["seed"]) in PAIRS[b]]
    if len(rows) != len(PAIRS[b]):
        return None if not rows else dict(partial=len(rows))
    if pk is None:
        cost = 0.0
    elif pk == "claude":
        cost = sum(traj(b, cfg, r)["cost_usd"] for r in rows)
    else:
        pin, pout = PRICE[pk]
        cost = sum((r["tokens_in"] or 0) * pin + (r["tokens_out"] or 0) * pout for r in rows) / 1e6
    return dict(n=len(rows), passed=sum(r["passed"] for r in rows),
                exact=sum(r.get("strict_set_match", 0) for r in rows) if b == "itbench_lite" else None, answered=sum(1 for r in rows if r["parse_ok"]),
                mean_steps=round(st.mean(r["n_steps"] for r in rows), 1), mean_wall_s=round(st.mean(r["wall_s"] for r in rows)),
                cost_usd=round(cost, 2))


out = []
for m, prov, th, h, cfg, pk in ROWS:
    full = RES / "itbench_lite" / cfg / "summary.json"
    fs = json.loads(full.read_text()) if full.exists() else None
    out.append(dict(model=m, provider=prov, thinking=th, harness=h, config=cfg,
                    itb_subset=subset("itbench_lite", cfg, pk), syn_subset=subset("synthetic", cfg, pk),
                    itb_full=(dict(n=fs["n_runs"], official=round(fs["mean_acc"], 3), strict=round(fs["strict_set_match_mean"], 3))
                              if fs and fs.get("n_runs") == 45 else None)))
(RES / "analysis/all_models_compare.json").write_text(json.dumps(out, indent=1))
for o in out:
    print(o["model"], "|", o["thinking"], "|", o["harness"], "| ITB", o["itb_subset"], "| SYN", o["syn_subset"], "| FULL", o["itb_full"])
