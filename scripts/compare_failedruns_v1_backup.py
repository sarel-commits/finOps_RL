#!/usr/bin/env python
"""Compare API models on the runs the local 8-bit baseline failed. Reads scores.jsonl only.
Writes results/analysis/failedruns_compare.json and report_assets/failedruns_rescue.png.
Cost = API usage tokens x Together list price (estimate, not invoice)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path.home() / "finops-eval"
RES = ROOT / "results"
MODELS = [("flash_failedruns", "Qwen3.8-Flash", 0.15, 0.47),
          ("glm53_failedruns", "GLM-5.3", 1.4, 4.4),
          ("kimik31_failedruns", "Kimi-K3.1-Pro", 2.0, 10.0)]
out = {}
for cfg, name, pin, pout in MODELS:
    out[name] = {}
    for b in ["itbench_lite", "synthetic"]:
        f = RES / b / cfg / "scores.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f)]
        ti = sum(r["tokens_in"] or 0 for r in rows); to = sum(r["tokens_out"] or 0 for r in rows)
        wd = 0
        if b == "itbench_lite":
            for r in rows:
                if r["parse_ok"] and not r["passed"] and isinstance(r["answer"], dict):
                    pt = {str(x.get("type", "")).strip().lower() for x in r["answer"].get("resource") or [] if isinstance(x, dict)}
                    if pt != {t.lower() for _, t in r["gt"]}:
                        wd += 1
        out[name][b] = dict(config=cfg, n=len(rows), rescued=sum(r["passed"] for r in rows),
                            answered=sum(1 for r in rows if r["parse_ok"]),
                            step_cap=sum(1 for r in rows if r["error"] == "max_steps"),
                            errors=sum(1 for r in rows if (r["error"] or "").startswith("exception")),
                            wrong_dimension=wd if b == "itbench_lite" else None,
                            mean_steps=sum(r["n_steps"] for r in rows) / len(rows),
                            mean_wall_s=sum(r["wall_s"] for r in rows) / len(rows),
                            tokens_in=ti, tokens_out=to, est_cost_usd=round(ti * pin / 1e6 + to * pout / 1e6, 3),
                            rescued_items=sorted(f"{r['item']}:s{r['seed']}" for r in rows if r["passed"]))
(RES / "analysis/failedruns_compare.json").write_text(json.dumps(out, indent=1))
for m, d in out.items():
    for b, v in d.items():
        print(m, b, {k: v[k] for k in v if k != "rescued_items"}, v["rescued_items"])
# chart
names = [m for m in out if out[m]]
fig, ax = plt.subplots(figsize=(9, 3.6))
cols = {"itbench_lite": "#2a78d6", "synthetic": "#eb6834"}
w = 0.38
for j, b in enumerate(["itbench_lite", "synthetic"]):
    xs = [i + (j - 0.5) * w for i in range(len(names))]
    vals = [100 * out[m][b]["rescued"] / out[m][b]["n"] if b in out[m] else 0 for m in names]
    ax.bar(xs, vals, w * 0.92, color=cols[b], label={"itbench_lite": "ITBench-Lite failures (n=23)", "synthetic": "Synthetic failures (n=9)"}[b])
    for x, m, v in zip(xs, names, vals):
        if b in out[m]:
            ax.text(x, v + 2, f"{out[m][b]['rescued']}/{out[m][b]['n']}", ha="center", fontsize=9, color="#52514e")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
ax.set_ylim(0, 112); ax.set_ylabel("rescued (%)")
ax.set_facecolor("#fcfcfb"); fig.set_facecolor("#fcfcfb")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
ax.set_title("Runs the local 8-bit model failed: share each API model gets right", loc="left", fontsize=11)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(ROOT / "report_assets/failedruns_rescue.png", dpi=150)
