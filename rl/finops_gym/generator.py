"""Procedural generator of ITBench-Lite-style FinOps cost-anomaly RCA tasks.

Output per task (identical schema to ITBench-Lite FinOps, so the same prompt/agent/eval apply):
  anomaly.json  {"date": "YYYY-MM-DD", "account_id": "..."}
  data.csv      date,account_id,csp,cloud_service,application,instance_family,price,num_instances,unblended_cost
  label         [{"name", "type"}]  (kept OUT of the agent's sandbox)

Entity hierarchy (as in ITBench-Lite): csp > cloud_service > application > instance_family,
cost = price * num_instances * 24, one row per (date, family) while the family is active.

Two regimes, mirroring what ITBench-Lite actually contains (see rl/README.md, "Eval data findings"):
  clean : root cause(s) shift to a new persistent level ON the anomaly date. Distractors are built to beat
          the naive "largest delta" shortcut: one/two-day bursts, volatile families at a peak, drops,
          old step changes. Label is accepted only if the reference oracle (oracle.detect) reproduces it.
  noisy : volatile baselines; the root cause is the entity with the largest increase vs. the previous week,
          onset possibly 0-3 days before the reported date (detector lag). Label is accepted only if it is the
          top-ranked delta with a margin over the runner-up.
Every task is fully determined by (seed, GenConfig); rejected draws are resampled deterministically.
"""
from __future__ import annotations

import hashlib
import io
import json
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import oracle as orc

ALNUM = string.ascii_letters + string.digits
CSPS = ["AWS", "GCP", "Azure"]
COLUMNS = ["date", "account_id", "csp", "cloud_service", "application", "instance_family", "price",
           "num_instances", "unblended_cost"]
GEN_VERSION = "1"


@dataclass
class GenConfig:
    regime_weights: dict = field(default_factory=lambda: {"clean": 0.7, "noisy": 0.3})
    mechanism_weights: dict = field(default_factory=lambda: {
        "count_shift": 0.30, "price_shift": 0.15, "multi_family": 0.15, "new_app": 0.15,
        "app_wide": 0.10, "new_family": 0.07, "multi_new_apps": 0.08})
    n_days: tuple = (21, 70)
    days_after: tuple = (3, 20)          # days in the data from the anomaly date onwards (inclusive)
    n_services: tuple = (8, 19)
    apps_per_service: tuple = (2, 7)
    fams_per_app: tuple = (3, 10)
    max_rows: int = 15000                # ITBench-Lite files are ~12-16k rows
    n_distractors: tuple = (2, 8)
    p_dominant_distractor: float = 0.6   # clean: P(some distractor's raw delta on the date > root cause's)
    noisy_margin: float = 1.5            # noisy: root-cause delta >= margin * runner-up delta
    max_tries: int = 40


def _rid(rng, n, used):
    while True:
        s = "".join(rng.choice(list(ALNUM), n))
        if s.lower() not in used:
            used.add(s.lower())
            return s


def _pick(rng, weights: dict):
    k = list(weights)
    p = np.array([weights[x] for x in k], float)
    return k[rng.choice(len(k), p=p / p.sum())]


class _Draft:
    """Mutable daily matrices (days x families) for price and instance counts."""

    def __init__(self, rng, cfg: GenConfig, regime: str):
        self.rng, self.cfg, self.regime = rng, cfg, regime
        used: set[str] = set()
        self.account = _rid(rng, 7, used)
        self.T = int(rng.integers(cfg.n_days[0], cfg.n_days[1] + 1))
        after = int(rng.integers(cfg.days_after[0], min(cfg.days_after[1], self.T - 16) + 1))
        self.D = self.T - after                                  # anomaly day index
        start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 330)))
        self.dates = pd.date_range(start, periods=self.T, freq="D")
        fam_budget = max(40, cfg.max_rows // self.T)
        self.fam, self.app_of, self.svc_of, self.csp_of = [], {}, {}, {}
        self.apps: dict[str, list[str]] = {}
        for _ in range(int(rng.integers(*cfg.n_services)) + 1):
            svc, csp = _rid(rng, 5, used), CSPS[int(rng.integers(3))]
            for _ in range(int(rng.integers(*cfg.apps_per_service)) + 1):
                app = _rid(rng, 3, used)
                self.apps[app] = []
                self.svc_of[app], self.csp_of[svc] = svc, csp
                for _ in range(int(rng.integers(*cfg.fams_per_app)) + 1):
                    if len(self.fam) >= fam_budget:
                        break
                    f = _rid(rng, 10, used)
                    self.fam.append(f); self.app_of[f] = app; self.apps[app].append(f)
        self.apps = {a: fs for a, fs in self.apps.items() if fs}
        self.used = used
        F = len(self.fam)
        self.price = np.repeat(np.round(rng.lognormal(-1.5, 0.8, F).clip(0.02, 6.0), 4)[None, :], self.T, 0)
        self.n = np.zeros((self.T, F))
        self.active = np.ones((self.T, F), bool)
        self.behavior = {}
        for j in range(F):
            self._baseline(j)

    # ---------- baselines ----------
    def _baseline(self, j):
        rng, T, D = self.rng, self.T, self.D
        b = _pick(rng, {"stable": .55, "noisy": .2, "weekly": .1, "volatile": .1, "sparse": .05}
                  if self.regime == "clean" else
                  {"stable": .15, "noisy": .35, "weekly": .1, "volatile": .3, "steppy": .1})
        base = float(rng.integers(1, 40))
        t = np.arange(T)
        if b == "stable":
            x = np.full(T, base)
            if rng.random() < 0.15 and D > 20:                    # old step change, far before the window
                x[: int(rng.integers(1, D - 16))] = max(1, round(base * rng.uniform(0.4, 1.6)))
        elif b == "noisy":
            x = base * (1 + rng.normal(0, 0.08, T))
        elif b == "weekly":
            x = base * np.where(self.dates.day_of_week < 5, 1.0, rng.uniform(0.4, 0.8))
        elif b == "volatile":
            lo, hi = max(1, base / 4), base * rng.uniform(2, 5)
            x = rng.uniform(lo, hi, T)
        elif b == "steppy":                                       # noisy regime: random level changes
            x = np.full(T, base)
            for c in np.sort(rng.integers(1, T, int(rng.integers(1, 4)))):
                x[c:] = max(1, round(x[c - 1] * rng.uniform(0.5, 1.5)))
            x = x * (1 + rng.normal(0, 0.05, T))
        else:  # sparse: on/off
            x = np.where(rng.random(T) < 0.6, base, 0.0)
        if self.regime == "noisy" and rng.random() < 0.15:        # occasional zero days
            x[rng.integers(0, T, int(rng.integers(1, 4)))] = 0
        # some families start or end mid-period, never inside [D-15, D+2]
        if rng.random() < 0.08 and D > 20:
            self.active[: int(rng.integers(1, D - 16)), j] = False
        self.n[:, j] = np.maximum(0, np.round(x))
        self.behavior[self.fam[j]] = b
        _ = t

    def j(self, f):
        return self.fam.index(f)

    def daily_cost(self, js):
        return (self.price[:, js] * self.n[:, js] * 24 * self.active[:, js]).sum(axis=-1) if isinstance(js, list) \
            else self.price[:, js] * self.n[:, js] * 24 * self.active[:, js]

    def raw_delta(self, js):
        c = self.daily_cost(js)
        return float(c[self.D] - c[max(0, self.D - 7): self.D].mean())

    # ---------- new entities ----------
    def add_app(self, n_fams, start):
        rng = self.rng
        svc = self.svc_of[list(self.apps)[int(rng.integers(len(self.apps)))]]
        app = _rid(rng, 3, self.used)
        self.apps[app] = []
        self.svc_of[app] = svc
        for _ in range(n_fams):
            f = _rid(rng, 10, self.used)
            self.fam.append(f); self.app_of[f] = app; self.apps[app].append(f)
            self.price = np.concatenate([self.price, np.full((self.T, 1), round(float(rng.lognormal(-1.5, 0.8)), 4))], 1)
            col = np.full(self.T, float(rng.integers(2, 40)))
            self.n = np.concatenate([self.n, col[:, None]], 1)
            act = np.zeros(self.T, bool); act[start:] = True
            self.active = np.concatenate([self.active, act[:, None]], 1)
            self.behavior[f] = "new"
        return app

    def to_frame(self) -> pd.DataFrame:
        r, c = np.nonzero(self.active)
        df = pd.DataFrame({
            "date": [f"{d.month}/{d.day}/{d.year}" for d in self.dates[r]],
            "account_id": self.account,
            "csp": [self.csp_of[self.svc_of[self.app_of[self.fam[j]]]] for j in c],
            "cloud_service": [self.svc_of[self.app_of[self.fam[j]]] for j in c],
            "application": [self.app_of[self.fam[j]] for j in c],
            "instance_family": [self.fam[j] for j in c],
            "price": self.price[r, c],
            "num_instances": self.n[r, c].astype(int),
        })
        df["unblended_cost"] = np.round(df["price"] * df["num_instances"] * 24, 4)
        df["_d"], df["_s"] = r, df["cloud_service"] + df["application"] + df["instance_family"]
        return df.sort_values(["_d", "_s"], kind="stable").drop(columns=["_d", "_s"]).reset_index(drop=True)


def _inject_clean(dr: _Draft, mech: str):
    rng, D = dr.rng, dr.D
    stable = [f for f in dr.fam if dr.behavior[f] in ("stable", "noisy") and dr.active[D - 15:, dr.j(f)].all()]
    apps_ok = [a for a, fs in dr.apps.items() if len(fs) >= 2 and all(f in stable for f in fs)]
    gt = []

    # background noise level: the largest natural day-over-week cost move of any family before injection
    noise = max(abs(dr.raw_delta(j)) for j in range(len(dr.fam)))
    floor = max(30.0, 1.5 * noise)

    def shift(f, kind):
        j = dr.j(f)
        if kind == "price":
            dr.price[D:, j] = np.round(dr.price[D:, j] * rng.uniform(1.8, 4.0), 4)
            need = floor - dr.raw_delta(j)
            if need > 0:                                   # price alone too small to stand out: raise it further
                dr.price[D:, j] = np.round(dr.price[D:, j] + need / (24 * max(dr.n[D, j], 1)), 4)
        else:
            dr.n[D:, j] = np.round(dr.n[D:, j] * rng.uniform(2.0, 6.0) + rng.integers(3, 15))
            need = floor - dr.raw_delta(j)
            if need > 0:
                dr.n[D:, j] += np.ceil(need / (24 * dr.price[D, j]))

    if mech in ("count_shift", "price_shift"):
        f = stable[int(rng.integers(len(stable)))]
        shift(f, "price" if mech == "price_shift" else "count"); gt = [(f, orc.TYPE_FAMILY)]
    elif mech == "multi_family":
        k = int(rng.integers(2, 4))
        for f in rng.choice(stable, k, replace=False):
            shift(str(f), "count" if rng.random() < .7 else "price"); gt.append((str(f), orc.TYPE_FAMILY))
    elif mech == "app_wide":
        if not apps_ok:
            return None
        a = apps_ok[int(rng.integers(len(apps_ok)))]
        fac = rng.uniform(2.0, 5.0)
        for f in dr.apps[a]:
            dr.n[D:, dr.j(f)] = np.round(dr.n[D:, dr.j(f)] * fac + 2)
        gt = [(a, orc.TYPE_APP)]
    elif mech == "new_app":
        gt = [(dr.add_app(int(rng.integers(2, 12)), D), orc.TYPE_APP)]
    elif mech == "multi_new_apps":
        gt = [(dr.add_app(int(rng.integers(1, 6)), D), orc.TYPE_APP) for _ in range(int(rng.integers(3, 21)))]
    elif mech == "new_family":
        a = list(dr.apps)[int(rng.integers(len(dr.apps)))]
        app_new = dr.add_app(1, D)                       # create, then re-home the family into an existing app
        f = dr.apps.pop(app_new)[0]; dr.app_of[f] = a; dr.apps[a].append(f)
        dr.n[D:, dr.j(f)] = rng.integers(10, 60)
        gt = [(f, orc.TYPE_FAMILY)]
    true_delta = sum(max(dr.raw_delta([dr.j(f) for f in (dr.apps[n] if t == orc.TYPE_APP else [n])]), 1.0)
                     for n, t in gt)
    return gt, true_delta


def _distractors(dr: _Draft, gt, true_delta):
    rng, D, cfg = dr.rng, dr.D, dr.cfg
    gt_fams = {f for n, t in gt for f in (dr.apps.get(n, []) if t == orc.TYPE_APP else [n])}
    pool = [f for f in dr.fam if f not in gt_fams and dr.behavior[f] != "new" and dr.active[D - 15:, dr.j(f)].all()]
    dominant = rng.random() < cfg.p_dominant_distractor
    kinds = []
    for i in range(int(rng.integers(*cfg.n_distractors)) if pool else 0):
        f = pool[int(rng.integers(len(pool)))]
        j = dr.j(f)
        kind = _pick(rng, {"burst": .45, "volatile_peak": .25, "drop": .3})
        target = true_delta * (rng.uniform(1.2, 3.0) if (dominant and i == 0) else rng.uniform(0.2, 0.9))
        if kind == "burst":                                   # 1-2 day spike that reverts
            L = int(rng.integers(1, 3))
            add = max(1, round(target / (dr.price[D, j] * 24)))
            dr.n[D:D + L, j] += add
        elif kind == "volatile_peak":                          # high-variance family, peak on the date, seen before
            base = dr.n[D - 14:D, j].mean() + 1
            peak = base + max(1, round(target / (dr.price[D, j] * 24)))
            hist = rng.uniform(base * 0.3, peak, dr.T)
            hist[rng.choice(np.arange(max(0, D - 14), D), 2, replace=False)] = peak * rng.uniform(1.0, 1.1, 2)
            dr.n[:, j] = np.round(hist); dr.n[D, j] = round(peak)
            dr.behavior[f] = "volatile"
        else:                                                 # drop
            dr.n[D:, j] = np.round(dr.n[D:, j] * rng.uniform(0.0, 0.5))
        kinds.append(kind)
    return kinds, dominant


def _inject_noisy(dr: _Draft):
    rng, D = dr.rng, dr.D
    k = 1 if rng.random() < 0.85 else 2
    cands = [f for f in dr.fam if dr.active[D - 15:, dr.j(f)].all()]
    gt = []
    for f in rng.choice(cands, k, replace=False):
        j, lag = dr.j(str(f)), int(rng.choice([0, 0, 1, 2, 3]))
        lvl = dr.n[D - 14:D, j].mean() * rng.uniform(2.5, 8.0) + rng.integers(10, 40)
        x = lvl * (1 + rng.normal(0, 0.05, dr.T - (D - lag)))
        dips = rng.random(x.size) < 0.12
        x[dips & (np.arange(x.size) > lag + 1)] *= rng.uniform(0, 0.4)   # occasional dips after the date
        dr.n[D - lag:, j] = np.round(x)
        gt.append((str(f), orc.TYPE_FAMILY))
    return gt


@dataclass
class Task:
    task_id: str
    seed: int
    regime: str
    mechanism: str
    anomaly: dict
    label: list
    csv_sha256: str
    n_rows: int
    meta: dict
    source_dir: str | None = None        # set for file-backed tasks (ITBench-Lite eval); None = regenerate from seed
    split: str = ""

    def to_json(self):
        return json.dumps(asdict(self))


def task_from_dir(d: Path, split: str = "itbench_lite") -> Task:
    """File-backed task (e.g. ITBench-Lite). The label is read here, on the host, and never copied to the sandbox."""
    import yaml
    d = Path(d)
    y = yaml.safe_load((d / "ground_truth.yaml").read_text())
    label = [dict(name=r["resource"]["name"], type=r["resource"]["type"]) for r in y["resource"]
             if r["resource"].get("root_cause", True)]
    text = (d / "data.csv").read_bytes()
    return Task(task_id=f"{split}-{d.name}", seed=-1, regime="external", mechanism="unknown",
                anomaly=json.loads((d / "anomaly.json").read_text()), label=label,
                csv_sha256=hashlib.sha256(text).hexdigest(), n_rows=text.count(b"\n") - 1, meta={},
                source_dir=str(d.resolve()), split=split)


def generate(seed: int, cfg: GenConfig | None = None, want_frame: bool = False):
    """Deterministic: same (seed, cfg) -> same task. Returns Task (and the DataFrame if want_frame)."""
    cfg = cfg or GenConfig()
    root = np.random.SeedSequence([seed, int(GEN_VERSION)])
    top, *children = root.spawn(cfg.max_tries + 1)
    top_rng = np.random.default_rng(top)                 # regime + mechanism fixed per seed (retries keep them)
    regime = _pick(top_rng, cfg.regime_weights)
    mech = _pick(top_rng, cfg.mechanism_weights) if regime == "clean" else "noisy_shift"
    for attempt, child in enumerate(children):
        rng = np.random.default_rng(child)
        dr = _Draft(rng, cfg, regime)
        meta = dict(attempt=attempt, n_days=dr.T, days_after=dr.T - dr.D, n_families=len(dr.fam))
        if regime == "clean":
            res = _inject_clean(dr, mech)
            if res is None:
                continue
            gt, true_delta = res
            kinds, dominant = _distractors(dr, gt, true_delta)
            meta.update(distractors=kinds, dominant_distractor=dominant, true_delta=round(true_delta, 2))
        else:
            gt = _inject_noisy(dr)
        df = dr.to_frame()
        anomaly = {"date": str(dr.dates[dr.D].date()), "account_id": dr.account}
        if not _validate(df, anomaly, gt, regime, cfg, meta):
            continue
        buf = io.StringIO(); df.to_csv(buf, index=False)
        sha = hashlib.sha256(buf.getvalue().encode()).hexdigest()
        label = [dict(name=n, type=t) for n, t in sorted(gt)]
        task = Task(task_id=f"g{GEN_VERSION}-{seed}", seed=seed, regime=regime, mechanism=mech, anomaly=anomaly,
                    label=label, csv_sha256=sha, n_rows=len(df), meta=meta)
        return (task, df) if want_frame else task
    raise RuntimeError(f"seed {seed}: no valid scenario in {cfg.max_tries} tries")


def _validate(df, anomaly, gt, regime, cfg, meta) -> bool:
    gt_set = {(n.lower(), t) for n, t in gt}
    naive = orc.naive_top_delta(df, anomaly)
    meta["naive_top_delta_exact"] = int({(n.lower(), t) for n, t in naive} == gt_set)
    if regime == "clean":
        pred = orc.detect(df, anomaly)
        ok = {(n.lower(), t) for n, t in pred} == gt_set
        meta["oracle_exact"] = int(ok)
        return ok
    d = orc.top_deltas(df, anomaly)
    k = len(gt)
    top = {(str(n).lower(), orc.TYPE_FAMILY) for n in d.index[:k]}
    margin = d.iloc[k - 1] / max(d.iloc[k], 1e-9)
    meta["rank_margin"] = round(float(margin), 3)
    meta["oracle_exact"] = int({(n.lower(), t) for n, t in orc.detect(df, anomaly)} == gt_set)
    return top == gt_set and d.iloc[k - 1] > 0 and margin >= cfg.noisy_margin


def materialize(task: Task | dict, out_dir: Path, cfg: GenConfig | None = None, verify: bool = True) -> Path:
    """Write ONLY the agent-visible files (anomaly.json, data.csv) for a task. Never writes the label."""
    t = task if isinstance(task, Task) else Task(**task)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if t.source_dir:                                   # file-backed: copy ONLY the two agent-visible files
        src = Path(t.source_dir)
        data = (src / "data.csv").read_bytes()
        if verify and hashlib.sha256(data).hexdigest() != t.csv_sha256:
            raise RuntimeError(f"{t.task_id}: data.csv changed on disk")
        (out_dir / "data.csv").write_bytes(data)
        (out_dir / "anomaly.json").write_bytes((src / "anomaly.json").read_bytes())
        return out_dir
    _, df = generate(t.seed, cfg, want_frame=True)
    buf = io.StringIO(); df.to_csv(buf, index=False)
    text = buf.getvalue()
    if verify and hashlib.sha256(text.encode()).hexdigest() != t.csv_sha256:
        raise RuntimeError(f"{t.task_id}: regenerated data.csv hash mismatch (generator/config/library drift)")
    (out_dir / "data.csv").write_text(text)
    (out_dir / "anomaly.json").write_text(json.dumps(t.anomaly, indent=4))
    return out_dir
