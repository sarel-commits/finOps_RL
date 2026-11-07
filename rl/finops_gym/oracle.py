"""Reference root-cause detector ("oracle") for ITBench-Lite-style FinOps cost anomalies.

It is independent of the generator's internals: it sees only anomaly.json + data.csv, exactly
like the agent. Its job is to make the task *semantics* explicit and testable:

  A root cause is an entity whose daily cost shifts to a new, persistent level starting on the
  anomaly date, beyond what its own recent history explains.
  - An Instance Family is reported when it shifted on its own.
  - An Application is reported instead of its families when the application is new on the
    anomaly date, or when all of its families shifted together.

Uses:
  * validates the generator: every generated scenario must be solved exactly by the oracle
    (otherwise the label is ambiguous and the scenario is rejected);
  * validates the semantics against the real ITBench-Lite ground truth (scripts/validate_oracle.py);
  * strong non-LLM baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TYPE_FAMILY = "Instance Family"
TYPE_APP = "Application"


@dataclass
class OracleParams:
    pre_days: int = 14          # history window before the anomaly date
    post_days: int = 3          # persistence window starting on the anomaly date (inclusive)
    z: float = 4.0              # shift must exceed z * pre std
    rel: float = 0.25           # ...and rel * pre mean
    abs_min: float = 5.0        # ...and an absolute floor in $/day
    persist_frac: float = 1.0   # fraction of post days that must sit above the pre-window max
    onset_lag: int = 0          # allow the shift to start up to this many days BEFORE the reported date.
                                # Tried 3 on ITBench-Lite: fixes nothing and adds a false positive on S9 -> default 0.


def load_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def _daily(df: pd.DataFrame, col: str) -> pd.DataFrame:
    p = df.groupby(["date", col])["unblended_cost"].sum().unstack(fill_value=0.0)
    full = pd.date_range(p.index.min(), p.index.max(), freq="D")
    return p.reindex(full, fill_value=0.0)


def _shift_at(a: np.ndarray, b: np.ndarray, prm: OracleParams):
    if b.size == 0 or b.sum() <= 0:
        return None
    if a.size and a.sum() == 0:
        return dict(kind="new", delta=float(b.mean()))
    if a.size == 0:
        return None
    delta = b.mean() - a.mean()
    thr = max(prm.z * a.std(), prm.rel * a.mean(), prm.abs_min)
    persist = (b > a.max()).mean() >= prm.persist_frac
    return dict(kind="shift", delta=float(delta)) if delta > thr and persist else None


def shifted_entities(df: pd.DataFrame, col: str, date: pd.Timestamp, prm: OracleParams) -> dict[str, dict]:
    """Entities with a persistent upward level shift whose onset t is in [date - onset_lag, date].
    pre = [t - pre_days, t - 1], post = [t, max(date, t + post_days - 1)] (always covers the reported date)."""
    p = _daily(df, col)
    out = {}
    for lag in range(0, prm.onset_lag + 1):
        t = date - pd.Timedelta(days=lag)
        pre = p.loc[t - pd.Timedelta(days=prm.pre_days): t - pd.Timedelta(days=1)]
        end = max(date, t + pd.Timedelta(days=prm.post_days - 1))
        post = p.loc[t: end]
        if pre.empty or post.empty:
            continue
        for e in p.columns:
            if e in out:
                continue
            r = _shift_at(pre[e].to_numpy(), post[e].to_numpy(), prm)
            if r:
                out[e] = dict(r, onset_lag=lag)
    return out


def detect(df: pd.DataFrame, anomaly: dict, prm: OracleParams | None = None) -> list[tuple[str, str]]:
    prm = prm or OracleParams()
    date = pd.Timestamp(anomaly["date"])
    df = _ensure_dt(df)
    df = df[df["account_id"] == anomaly["account_id"]] if "account_id" in anomaly else df
    fam = shifted_entities(df, "instance_family", date, prm)
    app_of = df.groupby("instance_family")["application"].first()
    fams_of_app = df[df["date"] >= date - pd.Timedelta(days=prm.pre_days)].groupby("application")["instance_family"].unique()
    app_first = df.groupby("application")["date"].min()

    result, covered = [], set()
    for app, fams in fams_of_app.items():
        fams = list(fams)
        if app_first[app] == date:                      # brand-new application
            result.append((app, TYPE_APP)); covered.update(fams)
        elif len(fams) >= 2 and all(f in fam for f in fams):   # all families moved together
            result.append((app, TYPE_APP)); covered.update(fams)
    for f in fam:
        if f not in covered:
            result.append((f, TYPE_FAMILY))
    return sorted(result)


def detect_dir(scenario_dir, prm: OracleParams | None = None) -> list[tuple[str, str]]:
    import json
    from pathlib import Path
    d = Path(scenario_dir)
    return detect(load_csv(d / "data.csv"), json.loads((d / "anomaly.json").read_text()), prm)


def top_deltas(df: pd.DataFrame, anomaly: dict, col: str = "instance_family", lookback: int = 7) -> pd.Series:
    """Cost on the reported date minus the mean of the previous `lookback` days, per entity, descending."""
    date = pd.Timestamp(anomaly["date"])
    df = _ensure_dt(df)
    df = df[df["account_id"] == anomaly["account_id"]] if "account_id" in anomaly else df
    p = _daily(df, col)
    prev = p.loc[date - pd.Timedelta(days=lookback): date - pd.Timedelta(days=1)].mean()
    return (p.loc[date] - prev).sort_values(ascending=False)


def naive_top_delta(df: pd.DataFrame, anomaly: dict, col: str = "instance_family", lookback: int = 7) -> list[tuple[str, str]]:
    """Largest raw cost delta vs. a short lookback mean. In the *clean* regime this is the shortcut we must
    defeat (volatile distractors); in the *noisy* regime it is the labelling rule (ITBench-Lite S11-S15)."""
    d = top_deltas(df, anomaly, col, lookback)
    t = TYPE_FAMILY if col == "instance_family" else TYPE_APP
    return [(str(d.index[0]), t)]


_ = np  # numpy used via pandas arrays
