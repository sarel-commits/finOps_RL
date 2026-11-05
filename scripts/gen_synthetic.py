#!/usr/bin/env python
"""Deterministic generator for the SUPPLEMENTARY SYNTHETIC FinOps eval set.

NOT an official benchmark. All data is synthetic (fixed seed). Ground truth is
computed programmatically from the generated data; scoring is exact/tolerance
match only (see score_synthetic.py).

Layout produced:
  data/synthetic/workspaces/<task_id>/   files the agent may read (copied per run)
  data/synthetic/tasks.jsonl              id, category, prompt, answer_format, files
  data/synthetic/ground_truth.jsonl       id, answer, match spec (never shown to model)
"""
import json, math, shutil, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261106
ROOT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
rng = np.random.default_rng(SEED)

ACCOUNTS = [("111111111111", "prod-core"), ("222222222222", "prod-data"),
            ("333333333333", "staging"), ("444444444444", "sandbox")]
SERVICES = [  # name, category, resource type, base daily cost range per resource
    ("Amazon Elastic Compute Cloud", "Compute", "Instance", (8, 60)),
    ("Amazon Relational Database Service", "Databases", "DBInstance", (15, 90)),
    ("Amazon Simple Storage Service", "Storage", "Bucket", (1, 20)),
    ("AWS Lambda", "Compute", "Function", (0.5, 6)),
    ("Amazon Elastic Kubernetes Service", "Compute", "Cluster", (5, 12)),
    ("Amazon DynamoDB", "Databases", "Table", (2, 25)),
    ("Amazon CloudWatch", "Management and Governance", "LogGroup", (0.5, 5)),
]
REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]
DAYS = pd.date_range("2026-07-01", "2026-09-30", freq="D")


def rid(prefix, n=10):
    return prefix + "".join(rng.choice(list("0123456789abcdef"), n))


def r2(x):
    return float(round(x, 2))


# ----------------------------------------------------------------------------
# 1. Main FOCUS-like billing dataset
# ----------------------------------------------------------------------------
def build_billing():
    rows = []
    resources = []
    for acct, aname in ACCOUNTS:
        scale = {"prod-core": 1.0, "prod-data": 0.8, "staging": 0.35, "sandbox": 0.15}[aname]
        for svc, cat, rtype, (lo, hi) in SERVICES:
            nres = int(rng.integers(1, 4))
            for _ in range(nres):
                base = rng.uniform(lo, hi) * scale
                trend = rng.uniform(-0.004, 0.008)  # daily drift
                env = "prod" if aname.startswith("prod") else aname
                team = str(rng.choice(["payments", "search", "platform", "ml"]))
                resources.append(dict(acct=acct, aname=aname, svc=svc, cat=cat, rtype=rtype,
                                      rid=rid({"Instance": "i-", "DBInstance": "db-", "Bucket": "s3-",
                                               "Function": "fn-", "Cluster": "eks-", "Table": "ddb-",
                                               "LogGroup": "lg-"}[rtype]),
                                      region=str(rng.choice(REGIONS)), base=base, trend=trend,
                                      env=env, team=team))
    for r in resources:
        for i, d in enumerate(DAYS):
            cost = r["base"] * (1 + r["trend"]) ** i * rng.uniform(0.9, 1.1)
            list_cost = cost * 1.0
            # some EC2/RDS usage is covered by commitments: effective < billed
            pricing = "Standard"
            eff = cost
            if r["svc"] in (SERVICES[0][0], SERVICES[1][0]) and r["aname"] == "prod-core":
                pricing = "Committed"
                eff = cost * 0.72
            rows.append({
                "BillingPeriodStart": d.replace(day=1).strftime("%Y-%m-%d"),
                "ChargePeriodStart": d.strftime("%Y-%m-%d"),
                "ChargePeriodEnd": (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "BillingAccountId": "999999999999",
                "SubAccountId": r["acct"], "SubAccountName": r["aname"],
                "ProviderName": "AWS", "ServiceCategory": r["cat"], "ServiceName": r["svc"],
                "ResourceId": r["rid"], "ResourceType": r["rtype"], "RegionId": r["region"],
                "ChargeCategory": "Usage", "PricingCategory": pricing,
                "ListCost": round(list_cost, 4), "BilledCost": round(cost, 4),
                "EffectiveCost": round(eff, 4),
                "Tags": json.dumps({"env": r["env"], "team": r["team"]}),
            })
    df = pd.DataFrame(rows)
    # a few tax / credit rows (ChargeCategory != Usage) to test filtering
    for acct, aname in ACCOUNTS[:2]:
        for m in ["2026-08-01", "2026-09-01"]:
            df.loc[len(df)] = {**df.iloc[0].to_dict(), "SubAccountId": acct, "SubAccountName": aname,
                               "ChargePeriodStart": m, "BillingPeriodStart": m,
                               "ChargePeriodEnd": m, "ServiceName": "Tax", "ServiceCategory": "Other",
                               "ResourceId": "", "ResourceType": "", "ChargeCategory": "Tax",
                               "PricingCategory": "", "ListCost": 0.0,
                               "BilledCost": round(float(rng.uniform(50, 200)), 4),
                               "EffectiveCost": 0.0, "Tags": "{}", "RegionId": ""}
    return df


# ----------------------------------------------------------------------------
# Task helpers
# ----------------------------------------------------------------------------
TASKS, GT = [], []


def add(tid, cat, prompt, fmt, files, answer, match):
    TASKS.append(dict(id=tid, category=cat, prompt=prompt, answer_format=fmt, files=files))
    GT.append(dict(id=tid, category=cat, answer=answer, match=match))


def month_mask(df, ym):
    return df["ChargePeriodStart"].str.startswith(ym)


NUM = lambda rel=0.001, abs_=0.01: {"type": "number", "rel_tol": rel, "abs_tol": abs_}


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ws_root = ROOT / "workspaces"
    ws_root.mkdir(parents=True)
    shared = ROOT / "shared"
    shared.mkdir()

    bill = build_billing()
    bill.to_csv(shared / "focus_billing.csv", index=False)
    usage = bill[bill.ChargeCategory == "Usage"]

    def ws(tid, files):
        d = ws_root / tid
        d.mkdir(parents=True)
        for f in files:
            shutil.copy(shared / f, d / f)
        return files

    BILL_NOTE = ("The file focus_billing.csv is a FOCUS-format (FinOps Open Cost and Usage Specification) "
                 "daily cost export. Unless stated otherwise, 'spend' means the sum of BilledCost, and rows "
                 "of every ChargeCategory are included. ")

    # ---------------- A. spend totals (8) ----------------
    a_specs = [
        ("SubAccountName", "prod-core", "2026-08"), ("SubAccountName", "staging", "2026-09"),
        ("ServiceName", "Amazon Relational Database Service", "2026-07"),
        ("ServiceName", "AWS Lambda", "2026-09"),
    ]
    k = 0
    for col, val, ym in a_specs:
        k += 1
        v = bill[(bill[col] == val) & month_mask(bill, ym)].BilledCost.sum()
        add(f"spend_total_{k:02d}", "spend_total",
            BILL_NOTE + f"What was the total spend for {col} = '{val}' in {ym} (calendar month, by ChargePeriodStart)? Round to 2 decimals.",
            '{"total": <number>}', ws(f"spend_total_{k:02d}", ["focus_billing.csv"]),
            {"total": r2(v)}, {"total": NUM()})
    # by tag
    k += 1
    tagged = bill[bill.Tags.str.contains('"team": "payments"') & month_mask(bill, "2026-09")]
    add(f"spend_total_{k:02d}", "spend_total",
        BILL_NOTE + "What was the total spend in 2026-09 for resources tagged team=payments (Tags column is JSON)? Round to 2 decimals.",
        '{"total": <number>}', ws(f"spend_total_{k:02d}", ["focus_billing.csv"]),
        {"total": r2(tagged.BilledCost.sum())}, {"total": NUM()})
    # usage only, effective cost
    k += 1
    v = usage[(usage.SubAccountName == "prod-core") & month_mask(usage, "2026-09")].EffectiveCost.sum()
    add(f"spend_total_{k:02d}", "spend_total",
        "The file focus_billing.csv is a FOCUS-format daily cost export. What was the total EffectiveCost "
        "(amortized, after commitment discounts) of Usage charges only for SubAccountName 'prod-core' in 2026-09? Round to 2 decimals.",
        '{"total": <number>}', ws(f"spend_total_{k:02d}", ["focus_billing.csv"]),
        {"total": r2(v)}, {"total": NUM()})
    # commitment savings
    k += 1
    s = usage[month_mask(usage, "2026-08")]
    v = (s.BilledCost - s.EffectiveCost).sum()
    add(f"spend_total_{k:02d}", "spend_total",
        "The file focus_billing.csv is a FOCUS-format daily cost export. For Usage charges in 2026-08, what is the total "
        "difference BilledCost - EffectiveCost across all accounts (i.e. commitment-discount savings)? Round to 2 decimals.",
        '{"total": <number>}', ws(f"spend_total_{k:02d}", ["focus_billing.csv"]),
        {"total": r2(v)}, {"total": NUM()})
    # region
    k += 1
    v = bill[(bill.RegionId == "eu-west-1") & month_mask(bill, "2026-07")].BilledCost.sum()
    add(f"spend_total_{k:02d}", "spend_total",
        BILL_NOTE + "What was the total spend in RegionId 'eu-west-1' in 2026-07? Round to 2 decimals.",
        '{"total": <number>}', ws(f"spend_total_{k:02d}", ["focus_billing.csv"]),
        {"total": r2(v)}, {"total": NUM()})

    # ---------------- B. month-over-month (6) ----------------
    mom_specs = [("SubAccountName", "prod-data"), ("SubAccountName", "sandbox"),
                 ("ServiceName", "Amazon Elastic Compute Cloud"), ("ServiceName", "Amazon DynamoDB"),
                 ("ServiceCategory", "Databases"), ("SubAccountName", "staging")]
    for i, (col, val) in enumerate(mom_specs, 1):
        a = bill[(bill[col] == val) & month_mask(bill, "2026-08")].BilledCost.sum()
        b = bill[(bill[col] == val) & month_mask(bill, "2026-09")].BilledCost.sum()
        pct = (b - a) / a * 100
        add(f"mom_{i:02d}", "mom_change",
            BILL_NOTE + f"What is the month-over-month percentage change in spend for {col} = '{val}' from 2026-08 to 2026-09? "
            "Give the percentage (e.g. 12.34 for +12.34%, negative for a decrease), rounded to 2 decimals.",
            '{"pct_change": <number>}', ws(f"mom_{i:02d}", ["focus_billing.csv"]),
            {"pct_change": r2(pct)}, {"pct_change": {"type": "number", "rel_tol": 0, "abs_tol": 0.05}})

    # ---------------- C. top cost drivers (5) ----------------
    i = 0
    for acct in ["prod-core", "staging"]:
        i += 1
        s = usage[(usage.SubAccountName == acct) & month_mask(usage, "2026-09")].groupby("ServiceName").BilledCost.sum()
        top = list(s.sort_values(ascending=False).index[:3])
        add(f"top_drivers_{i:02d}", "top_drivers",
            BILL_NOTE + f"List the top 3 ServiceName values by spend for SubAccountName '{acct}' in 2026-09, highest first.",
            '{"services": ["<ServiceName>", "<ServiceName>", "<ServiceName>"]}', ws(f"top_drivers_{i:02d}", ["focus_billing.csv"]),
            {"services": top}, {"services": {"type": "ordered_list"}})
    i += 1
    s = usage[month_mask(usage, "2026-09")].groupby("ResourceId").BilledCost.sum()
    top = list(s.sort_values(ascending=False).index[:5])
    add(f"top_drivers_{i:02d}", "top_drivers",
        BILL_NOTE + "List the 5 ResourceId values with the highest spend in 2026-09 across all accounts, highest first.",
        '{"resources": ["<ResourceId>", ...]}', ws(f"top_drivers_{i:02d}", ["focus_billing.csv"]),
        {"resources": top}, {"resources": {"type": "ordered_list"}})
    i += 1
    a = usage[month_mask(usage, "2026-08")].groupby("ResourceId").BilledCost.sum()
    b = usage[month_mask(usage, "2026-09")].groupby("ResourceId").BilledCost.sum()
    delta = (b - a).sort_values(ascending=False)
    add(f"top_drivers_{i:02d}", "top_drivers",
        BILL_NOTE + "Which single ResourceId had the largest absolute increase in spend from 2026-08 to 2026-09, and by how much (rounded to 2 decimals)?",
        '{"resource_id": "<ResourceId>", "increase": <number>}', ws(f"top_drivers_{i:02d}", ["focus_billing.csv"]),
        {"resource_id": delta.index[0], "increase": r2(delta.iloc[0])},
        {"resource_id": {"type": "string"}, "increase": NUM()})
    i += 1
    s = usage[month_mask(usage, "2026-07")].groupby("SubAccountName").BilledCost.sum()
    share = s / s.sum() * 100
    add(f"top_drivers_{i:02d}", "top_drivers",
        "The file focus_billing.csv is a FOCUS-format daily cost export. Considering only Usage charges (BilledCost) in 2026-07, "
        "which SubAccountName has the largest share of spend, and what is that share as a percentage rounded to 2 decimals?",
        '{"account": "<SubAccountName>", "share_pct": <number>}', ws(f"top_drivers_{i:02d}", ["focus_billing.csv"]),
        {"account": share.idxmax(), "share_pct": r2(share.max())},
        {"account": {"type": "string"}, "share_pct": {"type": "number", "rel_tol": 0, "abs_tol": 0.05}})

    # ---------------- D. idle / unattached resources (6) ----------------
    inv = []
    for j in range(40):
        kind = str(rng.choice(["ebs_volume", "elastic_ip", "ec2_instance", "load_balancer"], p=[0.4, 0.15, 0.3, 0.15]))
        acct = str(rng.choice([a[1] for a in ACCOUNTS]))
        row = dict(resource_id="", resource_kind=kind, account=acct, region=str(rng.choice(REGIONS)),
                   state="", attached_to="", size_gb="", avg_cpu_pct_14d="", max_network_mb_14d="",
                   requests_14d="", monthly_cost_usd="")
        if kind == "ebs_volume":
            row["resource_id"] = rid("vol-", 12)
            avail = rng.random() < 0.35
            row["state"] = "available" if avail else "in-use"
            row["attached_to"] = "" if avail else rid("i-")
            size = int(rng.choice([20, 50, 100, 200, 500, 1000]))
            row["size_gb"] = size
            row["monthly_cost_usd"] = round(size * 0.08, 2)
        elif kind == "elastic_ip":
            row["resource_id"] = rid("eipalloc-", 10)
            assoc = rng.random() < 0.5
            row["state"] = "associated" if assoc else "unassociated"
            row["attached_to"] = rid("eni-") if assoc else ""
            row["monthly_cost_usd"] = 3.65
        elif kind == "ec2_instance":
            row["resource_id"] = rid("i-", 12)
            row["state"] = str(rng.choice(["running", "running", "running", "stopped"]))
            idle = rng.random() < 0.35
            row["avg_cpu_pct_14d"] = round(float(rng.uniform(0.1, 1.9) if idle else rng.uniform(3, 70)), 2) if row["state"] == "running" else 0.0
            row["max_network_mb_14d"] = round(float(rng.uniform(0.5, 4.5) if idle else rng.uniform(50, 5000)), 1) if row["state"] == "running" else 0.0
            row["monthly_cost_usd"] = round(float(rng.uniform(30, 400)), 2) if row["state"] == "running" else 0.0
        else:
            row["resource_id"] = "arn:aws:elasticloadbalancing:lb/" + rid("", 8)
            row["state"] = "active"
            row["requests_14d"] = int(0 if rng.random() < 0.4 else rng.integers(100, 10_000_000))
            row["monthly_cost_usd"] = 16.43
        inv.append(row)
    invdf = pd.DataFrame(inv)
    invdf.to_csv(shared / "inventory.csv", index=False)
    INV_NOTE = "The file inventory.csv is a cloud resource inventory snapshot with 14-day utilization stats. "
    unatt = invdf[(invdf.resource_kind == "ebs_volume") & (invdf.state == "available")]
    add("idle_01", "idle_resources", INV_NOTE + "List the resource_id of every EBS volume that is not attached to any instance (state 'available').",
        '{"resource_ids": ["..."]}', ws("idle_01", ["inventory.csv"]),
        {"resource_ids": sorted(unatt.resource_id)}, {"resource_ids": {"type": "set"}})
    add("idle_02", "idle_resources", INV_NOTE + "What is the total monthly_cost_usd of all unattached (state 'available') EBS volumes? Round to 2 decimals.",
        '{"total": <number>}', ws("idle_02", ["inventory.csv"]),
        {"total": r2(unatt["monthly_cost_usd"].pipe(pd.to_numeric, errors="coerce").sum())}, {"total": NUM()})
    ueip = invdf[(invdf.resource_kind == "elastic_ip") & (invdf.state == "unassociated")]
    add("idle_03", "idle_resources", INV_NOTE + "List the resource_id of every Elastic IP that is not associated with any network interface.",
        '{"resource_ids": ["..."]}', ws("idle_03", ["inventory.csv"]),
        {"resource_ids": sorted(ueip.resource_id)}, {"resource_ids": {"type": "set"}})
    ec2 = invdf[invdf.resource_kind == "ec2_instance"]
    idle = ec2[(ec2.state == "running") & (ec2["avg_cpu_pct_14d"].pipe(pd.to_numeric, errors="coerce") < 2) & (ec2["max_network_mb_14d"].pipe(pd.to_numeric, errors="coerce") < 5)]
    add("idle_04", "idle_resources", INV_NOTE + "An EC2 instance is considered idle if it is running AND avg_cpu_pct_14d < 2 AND max_network_mb_14d < 5. List the resource_id of every idle EC2 instance.",
        '{"resource_ids": ["..."]}', ws("idle_04", ["inventory.csv"]),
        {"resource_ids": sorted(idle.resource_id)}, {"resource_ids": {"type": "set"}})
    lb = invdf[(invdf.resource_kind == "load_balancer") & (invdf["requests_14d"].pipe(pd.to_numeric, errors="coerce") == 0)]
    add("idle_05", "idle_resources", INV_NOTE + "List the resource_id of every load balancer that received zero requests in the last 14 days.",
        '{"resource_ids": ["..."]}', ws("idle_05", ["inventory.csv"]),
        {"resource_ids": sorted(lb.resource_id)}, {"resource_ids": {"type": "set"}})
    waste = (unatt["monthly_cost_usd"].pipe(pd.to_numeric, errors="coerce").sum() + ueip["monthly_cost_usd"].pipe(pd.to_numeric, errors="coerce").sum()
             + idle["monthly_cost_usd"].pipe(pd.to_numeric, errors="coerce").sum() + lb["monthly_cost_usd"].pipe(pd.to_numeric, errors="coerce").sum())
    add("idle_06", "idle_resources", INV_NOTE + "Compute the total monthly_cost_usd of all waste: unattached EBS volumes (state 'available'), "
        "unassociated Elastic IPs, idle EC2 instances (running AND avg_cpu_pct_14d < 2 AND max_network_mb_14d < 5), and load balancers with zero requests_14d. Round to 2 decimals.",
        '{"total": <number>}', ws("idle_06", ["inventory.csv"]), {"total": r2(waste)}, {"total": NUM()})

    # ---------------- E. rightsizing (6) ----------------
    catalog = []
    for fam, per_vcpu, mem_ratio in [("m7", 0.048, 4), ("c7", 0.0425, 2), ("r7", 0.063, 8)]:
        for size, vcpu in [("large", 2), ("xlarge", 4), ("2xlarge", 8), ("4xlarge", 16), ("8xlarge", 32)]:
            catalog.append(dict(instance_type=f"{fam}.{size}", family=fam, vcpu=vcpu, memory_gib=vcpu * mem_ratio,
                                hourly_usd=round(per_vcpu * vcpu, 4)))
    cat = pd.DataFrame(catalog)
    cat.to_csv(shared / "instance_catalog.csv", index=False)
    util = []
    for j in range(18):
        t = cat.iloc[int(rng.integers(1, len(cat)))]
        util.append(dict(instance_id=rid("i-", 12), instance_type=t.instance_type,
                         p95_cpu_pct=round(float(rng.uniform(5, 95)), 1), p95_mem_pct=round(float(rng.uniform(10, 95)), 1)))
    udf = pd.DataFrame(util)
    udf.to_csv(shared / "utilization.csv", index=False)
    RULE = ("Rightsizing rule: for each instance, required_vcpu = current_vcpu * p95_cpu_pct/100 / 0.7 and "
            "required_mem = current_memory_gib * p95_mem_pct/100 / 0.8. The recommended type is the cheapest instance_type "
            "in the SAME family (from instance_catalog.csv) with vcpu >= required_vcpu and memory_gib >= required_mem. "
            "If no type in the family satisfies both, keep the current type. Monthly cost = hourly_usd * 730. ")
    U_NOTE = "Files: utilization.csv (per-instance p95 CPU and memory utilization) and instance_catalog.csv (instance types with vcpu, memory_gib, hourly_usd). "

    def recommend(row):
        cur = cat[cat.instance_type == row.instance_type].iloc[0]
        rv = cur.vcpu * row.p95_cpu_pct / 100 / 0.7
        rm = cur.memory_gib * row.p95_mem_pct / 100 / 0.8
        c = cat[(cat.family == cur.family) & (cat.vcpu >= rv) & (cat.memory_gib >= rm)].sort_values("hourly_usd")
        rec = cur if c.empty else c.iloc[0]
        return rec.instance_type, (cur.hourly_usd - rec.hourly_usd) * 730

    recs = udf.apply(lambda r: pd.Series(recommend(r), index=["rec", "saving"]), axis=1)
    udf2 = pd.concat([udf, recs], axis=1)
    for q, j in enumerate([0, 5, 11], 1):
        row = udf2.iloc[j]
        add(f"rightsize_{q:02d}", "rightsizing", U_NOTE + RULE + f"What is the recommended instance_type for instance {row.instance_id}?",
            '{"instance_type": "<type>"}', ws(f"rightsize_{q:02d}", ["utilization.csv", "instance_catalog.csv"]),
            {"instance_type": row.rec}, {"instance_type": {"type": "string"}})
    add("rightsize_04", "rightsizing", U_NOTE + RULE + "How many instances should be changed to a different (cheaper or more expensive) instance type?",
        '{"count": <integer>}', ws("rightsize_04", ["utilization.csv", "instance_catalog.csv"]),
        {"count": int((udf2.rec != udf2.instance_type).sum())}, {"count": {"type": "number", "rel_tol": 0, "abs_tol": 0}})
    net = udf2.saving.sum()
    add("rightsize_05", "rightsizing", U_NOTE + RULE + "If every instance is moved to its recommended type, what is the net total monthly saving in USD "
        "(positive = saving; upsizes count as negative savings)? Round to 2 decimals.",
        '{"monthly_saving": <number>}', ws("rightsize_05", ["utilization.csv", "instance_catalog.csv"]),
        {"monthly_saving": r2(net)}, {"monthly_saving": NUM()})
    under = udf2[udf2.saving < -1e-9]
    add("rightsize_06", "rightsizing", U_NOTE + RULE + "List the instance_id of every instance whose recommended type is MORE expensive than its current type (under-provisioned).",
        '{"instance_ids": ["..."]}', ws("rightsize_06", ["utilization.csv", "instance_catalog.csv"]),
        {"instance_ids": sorted(under.instance_id)}, {"instance_ids": {"type": "set"}})

    # ---------------- F. RI / Savings Plan math (6), no files ----------------
    od, up, rh = 0.192, 1000.0, 0.075
    be = up / ((od - rh) * 730)
    add("commit_01", "commitment_math",
        f"An on-demand instance costs ${od}/hour. A 1-year partial-upfront Reserved Instance costs ${up:.0f} upfront plus ${rh}/hour. "
        "Assuming the instance runs 730 hours per month, after how many months (fractional, 2 decimals) does the RI become cheaper than on-demand (break-even point)?",
        '{"months": <number>}', [], {"months": r2(be)}, {"months": {"type": "number", "rel_tol": 0.005, "abs_tol": 0.01}})
    od2, up2 = 0.384, 2150.0
    sav = (od2 * 8760 - up2) / (od2 * 8760) * 100
    add("commit_02", "commitment_math",
        f"An instance costs ${od2}/hour on-demand. A 1-year all-upfront RI costs ${up2:.0f} with no hourly charge. Assuming 8760 hours/year of usage, "
        "what is the percentage saving of the RI versus on-demand over the year? Round to 2 decimals.",
        '{"saving_pct": <number>}', [], {"saving_pct": r2(sav)}, {"saving_pct": {"type": "number", "rel_tol": 0, "abs_tol": 0.05}})
    util_th = (up2 / 8760) / od2 * 100
    add("commit_03", "commitment_math",
        f"Same RI as before: ${up2:.0f} all-upfront for 1 year versus ${od2}/hour on-demand. What is the minimum utilization (percentage of the 8760 hours "
        "the instance must actually run) for the RI to be no more expensive than paying on-demand only for the hours used? Round to 2 decimals.",
        '{"min_utilization_pct": <number>}', [], {"min_utilization_pct": r2(util_th)},
        {"min_utilization_pct": {"type": "number", "rel_tol": 0, "abs_tol": 0.05}})
    commit, disc = 10.0, 0.28
    hourly_od = [12, 8, 15, 6, 10, 14, 9, 11]
    # SP covers up to `commit` $/h of discounted usage; usage priced at OD*(1-disc) under the plan
    total_sp = 0.0
    for h in hourly_od:
        disc_cost = h * (1 - disc)
        covered = min(disc_cost, commit)
        uncovered_od = (disc_cost - covered) / (1 - disc)
        total_sp += commit + uncovered_od
    total_od = sum(hourly_od)
    add("commit_04", "commitment_math",
        f"A Compute Savings Plan has a commitment of ${commit}/hour and gives a {int(disc*100)}% discount off on-demand. In each hour, eligible usage is first "
        "priced at the discounted rate and charged against the commitment (unused commitment is still paid); any discounted cost above the commitment is "
        f"instead billed at full on-demand rates for the uncovered portion. Over 8 hours the on-demand cost of usage per hour was {hourly_od} dollars. "
        "What is the total cost with the Savings Plan over these 8 hours, and the saving versus pure on-demand? Round to 2 decimals.",
        '{"total_cost": <number>, "saving": <number>}', [],
        {"total_cost": r2(total_sp), "saving": r2(total_od - total_sp)},
        {"total_cost": NUM(0.002), "saving": NUM(0.002)})
    up3, mh3, years = 3600.0, 0.05, 3
    eff = (up3 + mh3 * 8760 * years) / (8760 * years)
    add("commit_05", "commitment_math",
        f"A 3-year partial-upfront RI costs ${up3:.0f} upfront plus ${mh3}/hour for every hour of the term (8760 hours/year). What is its effective "
        "(amortized) hourly rate? Round to 4 decimals.",
        '{"effective_hourly": <number>}', [], {"effective_hourly": float(round(eff, 4))},
        {"effective_hourly": {"type": "number", "rel_tol": 0.002, "abs_tol": 0.0001}})
    n, od4, ri4 = 12, 0.10, 0.062
    base_inst = [14, 12, 15, 13, 12, 18, 12, 16, 12, 13, 20, 12]  # running instances each month
    best = None
    for k_ in range(0, 21):
        cost = sum(k_ * ri4 * 730 + max(0, c - k_) * od4 * 730 for c in base_inst)
        if best is None or cost < best[1] - 1e-9:
            best = (k_, cost)
    add("commit_06", "commitment_math",
        f"Over 12 months, the number of identical instances running continuously each month was {base_inst}. Each instance costs ${od4}/hour on-demand, or "
        f"${ri4}/hour as a 1-year RI that is paid for every hour of the year whether used or not (730 hours/month). How many RIs should be purchased to "
        "minimize total yearly cost, and what is that minimum total cost? Round cost to 2 decimals.",
        '{"num_ris": <integer>, "total_cost": <number>}', [],
        {"num_ris": best[0], "total_cost": r2(best[1])},
        {"num_ris": {"type": "number", "rel_tol": 0, "abs_tol": 0}, "total_cost": NUM()})

    # ---------------- G. anomaly detection (6) ----------------
    adays = pd.date_range("2026-08-01", periods=60, freq="D")
    for q in range(1, 7):
        rows = []
        svcs = [s[0] for s in SERVICES[:5]]
        accts = [a[1] for a in ACCOUNTS[:3]]
        spike_day = adays[int(rng.integers(20, 58))]
        spike_svc = str(rng.choice(svcs))
        spike_acct = str(rng.choice(accts))
        for a_ in accts:
            for s_ in svcs:
                base = rng.uniform(40, 400)
                weekly = rng.uniform(0, 0.15)
                for d in adays:
                    c = base * (1 + weekly * (d.day_of_week >= 5) * -1) * rng.normal(1, 0.04)
                    if d == spike_day and s_ == spike_svc and a_ == spike_acct:
                        c *= rng.uniform(2.2, 3.5)
                    rows.append(dict(ChargePeriodStart=d.strftime("%Y-%m-%d"), SubAccountName=a_, ServiceName=s_,
                                     BilledCost=round(float(c), 2)))
        fn = f"daily_cost_{q}.csv"
        pd.DataFrame(rows).to_csv(shared / fn, index=False)
        add(f"anomaly_{q:02d}", "anomaly_detection",
            f"The file {fn} contains daily BilledCost per SubAccountName and ServiceName. Exactly one (date, account, service) combination has an anomalous cost spike. "
            "Identify it.",
            '{"date": "YYYY-MM-DD", "account": "<SubAccountName>", "service": "<ServiceName>"}', ws(f"anomaly_{q:02d}", [fn]),
            {"date": spike_day.strftime("%Y-%m-%d"), "account": spike_acct, "service": spike_svc},
            {"date": {"type": "string"}, "account": {"type": "string"}, "service": {"type": "string"}})

    with open(ROOT / "tasks.jsonl", "w") as f:
        for t in TASKS:
            f.write(json.dumps(t) + "\n")
    with open(ROOT / "ground_truth.jsonl", "w") as f:
        for g in GT:
            f.write(json.dumps(g) + "\n")
    h = hashlib.sha256((ROOT / "ground_truth.jsonl").read_bytes()).hexdigest()[:16]
    print(f"wrote {len(TASKS)} tasks; ground_truth sha256[:16]={h}")
    print(pd.Series([t["category"] for t in TASKS]).value_counts().to_string())


if __name__ == "__main__":
    main()
