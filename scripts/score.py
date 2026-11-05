#!/usr/bin/env python
"""Deterministic scorers (no LLM judge).

ITBench-Lite FinOps — implements the official ITBench-Evaluations FinOps rubric
(prompts/finops_main.py, commit 14f026f) deterministically:
  root_cause_resource = 1 iff every GT resource is matched by name AND type, else 0.
  name: exact after trim, case-insensitive; type: case-insensitive, spaces/_/- normalized.
Secondary (labeled, not official): strict set-equality, name-only match,
precision/recall/F1 over (name,type) pairs.

Synthetic — per-field exact/tolerance match as specified in ground_truth.jsonl.
A task passes iff all fields match.

  python scripts/score.py itbench_lite <config>
  python scripts/score.py synthetic <config>
"""
import json, math, re, sys
from pathlib import Path
import statistics as st

ROOT = Path.home() / "finops-eval"


def norm_t(s):
    return re.sub(r"[\s_\-]+", " ", str(s or "")).strip().lower()


def norm_n(s):
    return str(s or "").strip().lower()


def load_itb_gt():
    import yaml
    gt = {}
    base = ROOT / "data/itbench-lite/snapshots/finops/v0.1-finops-anomaly"
    for d in base.iterdir():
        y = yaml.safe_load((d / "ground_truth.yaml").read_text())
        gt[d.name] = [(r["resource"]["name"], r["resource"]["type"]) for r in y["resource"]
                      if r["resource"].get("root_cause", True)]
    return gt


def score_itb_one(answer, gt):
    pred = []
    if isinstance(answer, dict) and isinstance(answer.get("resource"), list):
        for r in answer["resource"]:
            if isinstance(r, dict):
                pred.append((norm_n(r.get("name")), norm_t(r.get("type"))))
    g = {(norm_n(n), norm_t(t)) for n, t in gt}
    p = set(pred)
    tp = len(g & p)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    gn = {n for n, _ in g}
    pn = {n for n, _ in p}
    return dict(root_cause_resource=int(g <= p), strict_set_match=int(g == p), name_only_match=int(gn <= pn),
                precision=prec, recall=rec, f1=f1, n_pred=len(p), n_gt=len(g),
                format_ok=int(isinstance(answer, dict) and isinstance(answer.get("resource"), list)))


def num_match(a, b, spec):
    try:
        a = float(a)
    except Exception:
        return False
    return math.isclose(a, float(b), rel_tol=spec.get("rel_tol", 0), abs_tol=spec.get("abs_tol", 0))


def field_match(pred, gold, spec):
    t = spec["type"]
    if t == "number":
        return num_match(pred, gold, spec)
    if t == "string":
        return norm_n(pred) == norm_n(gold)
    if t == "set":
        return isinstance(pred, list) and {norm_n(x) for x in pred} == {norm_n(x) for x in gold}
    if t == "ordered_list":
        return isinstance(pred, list) and [norm_n(x) for x in pred] == [norm_n(x) for x in gold]
    raise ValueError(t)


def main(bench, config):
    out = ROOT / "results" / bench / config
    rows = []
    if bench == "itbench_lite":
        gt = load_itb_gt()
    else:
        gt = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/synthetic/ground_truth.jsonl")}
    for tf in sorted(out.glob("*/seed_*/trajectory.json")):
        tr = json.loads(tf.read_text())
        item, seed = tf.parent.parent.name, int(tf.parent.name.split("_")[1])
        base = dict(item=item, seed=seed, status=tr.get("status"), error=tr.get("error"), parse_ok=tr.get("parse_ok"),
                    tokens_in=tr.get("tokens_in"), tokens_out=tr.get("tokens_out"), wall_s=tr.get("wall_s"),
                    gen_s=tr.get("gen_s"), n_steps=tr.get("n_steps"), n_tool_calls=tr.get("n_tool_calls"),
                    n_tool_errors=tr.get("n_tool_errors"), protocol=tr["config"]["protocol"],
                    raw_tool_parsed=tr.get("n_raw_tool_calls_parsed_by_harness", 0),
                    leak=tr.get("leak_check_ground_truth"), answer=tr.get("answer"))
        if bench == "itbench_lite":
            s = score_itb_one(tr.get("answer"), gt[item])
            base.update(s, gt=gt[item], passed=s["root_cause_resource"])
        else:
            g = gt[item]
            ans = tr.get("answer") if isinstance(tr.get("answer"), dict) else {}
            fields = {k: field_match(ans.get(k), v, g["match"][k]) for k, v in g["answer"].items()}
            base.update(category=g["category"], gold=g["answer"], fields=fields, passed=int(all(fields.values())))
        rows.append(base)
    (out / "scores.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    # aggregate per seed
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {s: [r["passed"] for r in rows if r["seed"] == s] for s in seeds}
    accs = [sum(v) / len(v) for v in per_seed.values() if v]
    summ = dict(bench=bench, config=config, n_items=len({r["item"] for r in rows}), n_runs=len(rows), seeds=seeds,
                per_seed_acc={s: sum(v) / len(v) for s, v in per_seed.items() if v},
                per_seed_n={s: len(v) for s, v in per_seed.items()},
                mean_acc=st.mean(accs) if accs else None, std_acc=st.stdev(accs) if len(accs) > 1 else 0.0,
                pass_total=sum(r["passed"] for r in rows), fail_total=sum(1 - r["passed"] for r in rows),
                parse_fail=sum(1 for r in rows if not r["parse_ok"]),
                protocols=sorted({r["protocol"] for r in rows}),
                raw_tool_calls_parsed_by_harness=sum(r["raw_tool_parsed"] or 0 for r in rows),
                leak_flags=sum(1 for r in rows if r["leak"]),
                tokens_in_mean=st.mean([r["tokens_in"] or 0 for r in rows]) if rows else None,
                tokens_out_mean=st.mean([r["tokens_out"] or 0 for r in rows]) if rows else None,
                wall_s_mean=st.mean([r["wall_s"] or 0 for r in rows]) if rows else None,
                wall_s_median=st.median([r["wall_s"] or 0 for r in rows]) if rows else None,
                decode_tps_mean=st.mean([(r["tokens_out"] or 0) / r["gen_s"] for r in rows if r["gen_s"]]) if rows else None)
    if bench == "itbench_lite":
        for k in ["strict_set_match", "name_only_match", "precision", "recall", "f1"]:
            vals = [st.mean([r[k] for r in rows if r["seed"] == s]) for s in seeds]
            summ[k + "_mean"] = st.mean(vals)
            summ[k + "_std"] = st.stdev(vals) if len(vals) > 1 else 0.0
        summ["per_item_pass_rate"] = {i: st.mean([r["passed"] for r in rows if r["item"] == i])
                                      for i in sorted({r["item"] for r in rows}, key=lambda x: int(x.split('-')[1]))}
    else:
        cats = sorted({r["category"] for r in rows})
        summ["per_category"] = {}
        for c in cats:
            vals = [st.mean([r["passed"] for r in rows if r["seed"] == s and r["category"] == c]) for s in seeds]
            summ["per_category"][c] = dict(mean=st.mean(vals), std=st.stdev(vals) if len(vals) > 1 else 0.0,
                                           n_items=len({r["item"] for r in rows if r["category"] == c}),
                                           passes=sum(r["passed"] for r in rows if r["category"] == c),
                                           runs=sum(1 for r in rows if r["category"] == c))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("per_item_pass_rate",)}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
