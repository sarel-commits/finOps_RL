#!/usr/bin/env python
"""NON-LLM reference baseline for ITBench-Lite FinOps (for context only).
For the anomaly account/date: delta = cost(date) - mean(cost over previous 7 days),
per entity of a dimension. Variant IF: predict top instance_family by delta.
Variant AUTO: among dims {instance_family, application}, choose the dimension whose
top entity explains the largest share of the account's total delta; predict entities of
that dimension whose delta >= 50% of the top entity's delta."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); import score
B = Path.home()/"finops-eval/data/itbench-lite/snapshots/finops/v0.1-finops-anomaly"
OUT = Path.home()/"finops-eval/results/itbench_lite/_heuristic_baseline"
gt = score.load_itb_gt(); rows = []
for d in sorted(B.iterdir(), key=lambda p: int(p.name.split('-')[1])):
    a = json.load(open(d/"anomaly.json")); df = pd.read_csv(d/"data.csv"); df["date"] = pd.to_datetime(df.date)
    s = df[df.account_id == a["account_id"]]; t = pd.Timestamp(a["date"])
    def deltas(dim):
        cur = s[s.date == t].groupby(dim).unblended_cost.sum()
        prev = s[(s.date < t) & (s.date >= t - pd.Timedelta(days=7))].groupby([dim, "date"]).unblended_cost.sum().groupby(dim).mean()
        return cur.sub(prev, fill_value=0).sort_values(ascending=False)
    tot = s[s.date == t].unblended_cost.sum() - s[(s.date < t) & (s.date >= t - pd.Timedelta(days=7))].groupby("date").unblended_cost.sum().mean()
    dif, dap = deltas("instance_family"), deltas("application")
    pred_if = {"resource": [{"name": dif.index[0], "type": "Instance Family"}]}
    share = {"instance_family": dif.iloc[0] / tot, "application": dap.iloc[0] / tot}
    dim = max(share, key=share.get); dd = dif if dim == "instance_family" else dap
    names = list(dd[dd >= 0.5 * dd.iloc[0]].index)
    pred_auto = {"resource": [{"name": n, "type": dim.replace("_", " ").title()} for n in names]}
    r = dict(item=d.name, gt=gt[d.name], pred_if=pred_if, pred_auto=pred_auto,
             if_pass=score.score_itb_one(pred_if, gt[d.name])["root_cause_resource"],
             auto=score.score_itb_one(pred_auto, gt[d.name]), total_delta=round(tot, 2), shares=share)
    rows.append(r); print(d.name, "IF:", r["if_pass"], "AUTO:", r["auto"]["root_cause_resource"], r["auto"]["strict_set_match"], round(r["auto"]["f1"], 2), dim, len(names), "gt", len(gt[d.name]))
(OUT/"baseline.jsonl").write_text("".join(json.dumps(r, default=str)+"\n" for r in rows))
summ = dict(top_instance_family_acc=sum(r["if_pass"] for r in rows)/len(rows),
            auto_acc=sum(r["auto"]["root_cause_resource"] for r in rows)/len(rows),
            auto_strict=sum(r["auto"]["strict_set_match"] for r in rows)/len(rows),
            auto_f1=sum(r["auto"]["f1"] for r in rows)/len(rows))
(OUT/"summary.json").write_text(json.dumps(summ, indent=1)); print(summ)
