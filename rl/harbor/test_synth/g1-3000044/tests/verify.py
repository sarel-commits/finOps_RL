import json, re, sys
def nn(s): return str(s or "").strip().lower()
def nt(s): return re.sub(r"[\s_\-]+", " ", str(s or "")).strip().lower()
gt = {(nn(x["name"]), nt(x["type"])) for x in json.load(open("/tests/label.json"))}
out = {"reward": 0.0, "exact": 0.0, "f1": 0.0, "official_root_cause_resource": 0.0, "format_ok": 0.0}
try:
    a = json.load(open("/app/answer.json"))
    p = {(nn(r.get("name")), nt(r.get("type"))) for r in a["resource"]}
    tp = len(gt & p); pr = tp / len(p) if p else 0.0; rc = tp / len(gt) if gt else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    out.update(format_ok=1.0, exact=float(p == gt), f1=f1, official_root_cause_resource=float(gt <= p))
    out["reward"] = 0.5 * f1 + 0.5 * out["exact"]
except Exception as e:
    print("unparseable /app/answer.json:", e, file=sys.stderr)
json.dump(out, open("/logs/verifier/reward.json", "w"))
print(out)
