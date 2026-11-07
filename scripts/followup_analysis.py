#!/usr/bin/env python
"""Follow-up experiment metrics (DECISIONS #19). Reads scores.jsonl only; writes results/analysis/followup.json.

Per config: official root_cause_resource (mean +- std over seeds, per-seed), 95% Wilson CI over all runs,
answered rate, accuracy when answered, forced_final count + accuracy, wrong-dimension answers, mean steps, wall.
wrong_dimension = answered with a parseable resource list whose set of (normalized) types != GT type set,
                  and failed the official metric.
"""
import json, math, re, statistics as st, sys
from pathlib import Path
ROOT = Path.home() / "finops-eval"
RES = ROOT / "results"


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def nt(s):
    return re.sub(r"[\s_\-]+", " ", str(s or "")).strip().lower()


def metrics(bench, cfg):
    f = RES / bench / cfg / "scores.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in open(f)]
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {s: st.mean([r["passed"] for r in rows if r["seed"] == s]) for s in seeds}
    k, n = sum(r["passed"] for r in rows), len(rows)
    ans = [r for r in rows if r["parse_ok"]]
    ff = [r for r in rows if r.get("forced_final")]
    m = dict(config=cfg, n_runs=n, seeds=seeds, n_per_seed={s: sum(1 for r in rows if r["seed"] == s) for s in seeds},
             official_mean=st.mean(per_seed.values()), official_std=st.stdev(per_seed.values()) if len(seeds) > 1 else 0.0,
             per_seed=per_seed, pass_total=k, wilson95=wilson(k, n),
             answered=len(ans), answered_rate=len(ans) / n,
             acc_when_answered=(sum(r["passed"] for r in ans) / len(ans)) if ans else None,
             forced_final=len(ff), forced_final_answered=sum(1 for r in ff if r["parse_ok"]),
             forced_final_pass=sum(r["passed"] for r in ff),
             forced_final_acc=(sum(r["passed"] for r in ff) / len(ff)) if ff else None,
             step_cap_no_answer=sum(1 for r in rows if not r["parse_ok"] and (r.get("n_steps") or 0) >= 20),
             mean_steps=st.mean([r["n_steps"] or 0 for r in rows]),
             wall_s_mean=st.mean([r["wall_s"] or 0 for r in rows]),
             wall_h_total=sum(r["wall_s"] or 0 for r in rows) / 3600)
    if bench == "itbench_lite":
        wd = []
        for r in ans:
            if r["passed"] or not isinstance(r["answer"], dict):
                continue
            pt = {nt(x.get("type")) for x in (r["answer"].get("resource") or []) if isinstance(x, dict)}
            gtt = {nt(t) for _, t in r["gt"]}
            if pt and pt != gtt:
                wd.append(f"{r['item']}/s{r['seed']}")
        m.update(wrong_dimension=len(wd), wrong_dimension_runs=wd,
                 strict_set_match=st.mean([r["strict_set_match"] for r in rows]),
                 f1_mean=st.mean([st.mean([r["f1"] for r in rows if r["seed"] == s]) for s in seeds]),
                 per_item_pass_rate={i: st.mean([r["passed"] for r in rows if r["item"] == i])
                                     for i in sorted({r["item"] for r in rows}, key=lambda x: int(x.split("-")[1]))})
    else:
        cats = sorted({r["category"] for r in rows})
        m["per_category"] = {c: st.mean([r["passed"] for r in rows if r["category"] == c]) for c in cats}
    return m


def main():
    out = {"itbench_lite": {}, "synthetic": {}}
    for cfg in ["nothink_native", "nothink_nudges", "nothink_nudges_v2", "nothink_nudges_v2_guard",
                "think_medium_nudges_v2_guard", "flash_nothink_v1", "flash_nudges_v2", "flash_nudges_v2_guard",
                "flash_think_medium_nudges_v2_guard"]:
        out["itbench_lite"][cfg] = metrics("itbench_lite", cfg)
    for cfg in ["nothink_native", "nothink_nudges_v2", "flash_nudges_v2"]:
        out["synthetic"][cfg] = metrics("synthetic", cfg)
    h = json.loads((RES / "itbench_lite/_heuristic_baseline/summary.json").read_text())
    out["itbench_lite"]["_heuristic"] = dict(official=h["top_instance_family_acc"], n=15,
                                            wilson95=wilson(round(h["top_instance_family_acc"] * 15), 15),
                                            auto=h["auto_acc"], auto_wilson95=wilson(round(h["auto_acc"] * 15), 15))
    (RES / "analysis/followup.json").write_text(json.dumps(out, indent=1))
    for b, d in out.items():
        for c, m in d.items():
            if m and not c.startswith("_"):
                print(b, c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
                             if k not in ("per_item_pass_rate", "wrong_dimension_runs", "per_category")})
    print("heuristic", out["itbench_lite"]["_heuristic"])


if __name__ == "__main__":
    main()
