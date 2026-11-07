#!/usr/bin/env python
"""Check that the oracle's semantics reproduce the real ITBench-Lite FinOps ground truth.

  python rl/scripts/validate_oracle.py
Reads data/itbench-lite (eval data). The oracle is a fixed rule, not fitted to these labels beyond
choosing its default thresholds; this run only *measures* agreement.
"""
import json, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
from finops_gym.oracle import detect_dir, naive_top_delta, load_csv  # noqa: E402
from finops_gym.reward import score_answer  # noqa: E402

ITB = ROOT / "data/itbench-lite/snapshots/finops/v0.1-finops-anomaly"


def main():
    rows = []
    for d in sorted(ITB.iterdir(), key=lambda p: int(p.name.split("-")[1])):
        y = yaml.safe_load((d / "ground_truth.yaml").read_text())
        gt = [dict(name=r["resource"]["name"], type=r["resource"]["type"]) for r in y["resource"]
              if r["resource"].get("root_cause", True)]
        df = load_csv(d / "data.csv")
        names_in_data = set(df["instance_family"]) | set(df["application"]) | set(df["cloud_service"])
        gt_missing = sum(g["name"] not in names_in_data for g in gt)
        pred = detect_dir(d)
        s = score_answer({"resource": [dict(name=n, type=t) for n, t in pred]}, gt)
        nv = score_answer({"resource": [dict(name=n, type=t) for n, t in
                                        naive_top_delta(df, json.loads((d / "anomaly.json").read_text()))]}, gt)
        rows.append((d.name, s["exact"], round(s["f1"], 3), nv["exact"], len(gt), gt_missing, pred[:4]))
        print(f"{d.name:12s} oracle_exact={s['exact']} f1={s['f1']:.3f} naive_exact={nv['exact']} "
              f"n_gt={len(gt)} gt_names_missing_from_data={gt_missing} pred={pred[:4]}{'...' if len(pred) > 4 else ''}")
    n = len(rows)
    print(f"\noracle exact {sum(r[1] for r in rows)}/{n}, mean F1 {sum(r[2] for r in rows)/n:.3f}; "
          f"naive top-delta exact {sum(r[3] for r in rows)}/{n}")
    print("scenarios whose GT names are partly absent from data.csv:", [r[0] for r in rows if r[5]])


if __name__ == "__main__":
    main()
