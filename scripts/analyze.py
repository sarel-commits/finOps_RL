#!/usr/bin/env python
"""Error analysis + charts from saved scores/trajectories.

Failure categories (first matching rule wins, applied to failed runs only):
  timeout            : error in {timeout_budget, max_steps, max_tokens_no_answer} or any tool timeout
  tool_call_failure  : harness exception, or no tool call at all on a data task, or >=50% tool calls errored
  format_parse_error : no parseable JSON answer, or wrong answer schema (missing keys / ITBench 'resource' list)
  hallucinated_data  : answer names an identifier that does not occur anywhere in the task's data files
  wrong_entity       : string/set/list fields wrong (ITBench: name or type mismatch)
  wrong_computation  : numeric fields wrong
Writes results/analysis/error_analysis.json and report_assets/*.png
"""
import json, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.home() / "finops-eval"
RES, ASSETS = ROOT / "results", ROOT / "report_assets"
ASSETS.mkdir(exist_ok=True)
(RES / "analysis").mkdir(exist_ok=True)
C = dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100", surface="#fcfcfb", text="#0b0b0b",
         text2="#52514e", grid="#e4e3df", muted="#8a8984")
plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "axes.edgecolor": C["grid"],
                     "axes.labelcolor": C["text2"], "xtick.color": C["text2"], "ytick.color": C["text2"],
                     "text.color": C["text"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True})
SERIES = [C["s1"], C["s2"], C["s3"], C["s4"]]

_data_cache = {}


def data_text(bench, item):
    key = (bench, item)
    if key not in _data_cache:
        if bench == "itbench_lite":
            p = ROOT / "data/itbench-lite/snapshots/finops/v0.1-finops-anomaly" / item / "data.csv"
            _data_cache[key] = p.read_text()
        else:
            d = ROOT / "data/synthetic/workspaces" / item
            _data_cache[key] = "".join(f.read_text() for f in d.glob("*")) if d.exists() else ""
    return _data_cache[key]


def answer_ids(bench, ans):
    out = []
    if not isinstance(ans, dict):
        return out
    if bench == "itbench_lite":
        for r in ans.get("resource") or []:
            if isinstance(r, dict) and r.get("name"):
                out.append(str(r["name"]))
    else:
        for v in ans.values():
            if isinstance(v, str) and re.search(r"[A-Za-z]", v) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                out.append(v)
            elif isinstance(v, list):
                out += [str(x) for x in v if isinstance(x, str)]
    return out


def classify(bench, r, tr):
    err = r.get("error") or ""
    if err in ("timeout_budget", "max_steps", "max_tokens_no_answer") or (tr.get("n_tool_timeouts") or 0) > 0:
        return "timeout"
    if err.startswith("exception"):
        return "tool_call_failure"
    data_task = bench == "itbench_lite" or not r.get("item", "").startswith("commit_")
    ntc = r.get("n_tool_calls") or 0
    if data_task and ntc == 0:
        return "tool_call_failure"
    if ntc and (r.get("n_tool_errors") or 0) / ntc >= 0.5 and not r.get("parse_ok"):
        return "tool_call_failure"
    if not r.get("parse_ok"):
        return "format_parse_error"
    ans = r.get("answer")
    if bench == "itbench_lite":
        if not r.get("format_ok"):
            return "format_parse_error"
    else:
        if not isinstance(ans, dict) or any(k not in ans for k in r["gold"]):
            return "format_parse_error"
    txt = data_text(bench, r["item"])
    if txt and any(i not in txt for i in answer_ids(bench, ans)):
        return "hallucinated_data"
    if bench == "itbench_lite":
        return "wrong_entity"
    bad = [k for k, ok in r["fields"].items() if not ok]
    num_bad = [k for k in bad if isinstance(r["gold"][k], (int, float)) and not isinstance(r["gold"][k], bool)]
    return "wrong_computation" if num_bad and len(num_bad) == len(bad) else "wrong_entity"


def summarize_traj(tr, n=600):
    codes = [tc.get("code") or "" for s in tr.get("steps", []) for tc in s.get("tool_calls", [])]
    return dict(n_steps=tr.get("n_steps"), last_code=(codes[-1][:n] if codes else None),
                final_text=(tr.get("final_text") or "")[-n:], error=tr.get("error"))


def main():
    configs = []
    for bench in ["itbench_lite", "synthetic"]:
        for d in sorted((RES / bench).glob("*/summary.json")):
            if d.parent.name.startswith("_"):
                continue
            configs.append((bench, d.parent.name))
    analysis = {}
    for bench, cfg in configs:
        rows = [json.loads(l) for l in open(RES / bench / cfg / "scores.jsonl")]
        cats, examples = Counter(), defaultdict(list)
        for r in rows:
            if r["passed"]:
                continue
            tf = RES / bench / cfg / r["item"] / f"seed_{r['seed']}" / "trajectory.json"
            tr = json.loads(tf.read_text())
            c = classify(bench, r, tr)
            cats[c] += 1
            if len(examples[c]) < 3 and not any(e["item"] == r["item"] for e in examples[c]):
                ex = dict(item=r["item"], seed=r["seed"], answer=r.get("answer"),
                          expected=r.get("gt") if bench == "itbench_lite" else r.get("gold"),
                          path=str(tf.relative_to(ROOT)), **summarize_traj(tr))
                examples[c].append(ex)
        analysis[f"{bench}/{cfg}"] = dict(n_runs=len(rows), n_fail=sum(1 - r["passed"] for r in rows),
                                          categories=dict(cats), examples=dict(examples))
    (RES / "analysis/error_analysis.json").write_text(json.dumps(analysis, indent=1, default=str))

    # ---------- charts ----------
    summ = {f"{b}/{c}": json.loads((RES / b / c / "summary.json").read_text()) for b, c in configs}
    # 1. ITBench-Lite per-scenario pass rate per config
    itb = [k for k in summ if k.startswith("itbench_lite/")]
    if itb:
        items = list(summ[itb[0]]["per_item_pass_rate"].keys())
        fig, ax = plt.subplots(figsize=(11, 4.2))
        w = 0.8 / len(itb)
        for j, k in enumerate(itb):
            vals = [summ[k]["per_item_pass_rate"].get(i, 0) * 100 for i in items]
            ax.bar([x + j * w - 0.4 + w / 2 for x in range(len(items))], vals, w * 0.92, color=SERIES[j % 4],
                   label=k.split("/")[1])
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([i.replace("Scenario-", "S") for i in items])
        ax.set_ylabel("pass rate across seeds (%)")
        ax.set_ylim(0, 105)
        ax.set_title("ITBench-Lite FinOps: root_cause_resource pass rate per scenario", loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper right", ncols=len(itb))
        fig.tight_layout()
        fig.savefig(ASSETS / "itbench_lite_per_scenario.png", dpi=150)
        plt.close(fig)
    # 2. Synthetic per-category
    syn = [k for k in summ if k.startswith("synthetic/")]
    if syn:
        cats = sorted(summ[syn[0]]["per_category"].keys())
        fig, ax = plt.subplots(figsize=(10, 4.2))
        w = 0.8 / len(syn)
        for j, k in enumerate(syn):
            pc = summ[k]["per_category"]
            m = [pc[c]["mean"] * 100 for c in cats]
            s = [pc[c]["std"] * 100 for c in cats]
            xs = [x + j * w - 0.4 + w / 2 for x in range(len(cats))]
            ax.bar(xs, m, w * 0.92, color=SERIES[j % 4], label=k.split("/")[1], yerr=s,
                   error_kw=dict(ecolor=C["text2"], elinewidth=1, capsize=3))
            for x, v in zip(xs, m):
                ax.text(x, v + 2, f"{v:.0f}", ha="center", fontsize=8, color=C["text2"])
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.replace("_", "\n") for c in cats])
        ax.set_ylabel("accuracy (%), mean ± std over seeds")
        ax.set_ylim(0, 112)
        ax.set_title("Synthetic FinOps eval (supplementary, NOT official): accuracy by category", loc="left", fontsize=11)
        if len(syn) > 1:
            ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(ASSETS / "synthetic_per_category.png", dpi=150)
        plt.close(fig)
    # 3. Error categories (horizontal stacked by config)
    order = ["wrong_entity", "wrong_computation", "hallucinated_data", "format_parse_error", "tool_call_failure", "timeout"]
    keys = [k for k in analysis if analysis[k]["n_fail"]]
    if keys:
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(keys) + 1.6))
        pal = [C["s1"], C["s2"], C["s3"], C["s4"], "#e87ba4", "#6250d6"]
        left = [0] * len(keys)
        for ci, c in enumerate(order):
            vals = [analysis[k]["categories"].get(c, 0) for k in keys]
            ax.barh(range(len(keys)), vals, left=left, color=pal[ci], label=c.replace("_", " "),
                    edgecolor=C["surface"], linewidth=2)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys)
        ax.set_xlabel("failed runs")
        ax.grid(axis="y", visible=False)
        ax.set_title("Failure categories (failed runs only)", loc="left", fontsize=11)
        ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.35 if len(keys) < 3 else -0.2))
        fig.tight_layout()
        fig.savefig(ASSETS / "error_categories.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 4. Latency per task distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    data, labels = [], []
    for b, c in configs:
        rows = [json.loads(l) for l in open(RES / b / c / "scores.jsonl")]
        data.append([r["wall_s"] / 60 for r in rows if r.get("wall_s")])
        labels.append(f"{b}/{c}")
    if data:
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.5,
                        medianprops=dict(color=C["text"], linewidth=2), flierprops=dict(markersize=4, markerfacecolor=C["muted"], markeredgecolor=C["muted"]))
        for p in bp["boxes"]:
            p.set_facecolor(C["s1"] + "55")
            p.set_edgecolor(C["s1"])
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels)
        ax.set_xlabel("wall-clock minutes per task run")
        ax.grid(axis="y", visible=False)
        ax.set_title("Latency per task (agent loop incl. tool execution)", loc="left", fontsize=11)
        fig.tight_layout()
        fig.savefig(ASSETS / "latency_per_task.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: dict(n_fail=v["n_fail"], categories=v["categories"]) for k, v in analysis.items()}, indent=1))


if __name__ == "__main__":
    main()
