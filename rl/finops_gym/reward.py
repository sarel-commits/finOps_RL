"""Deterministic verifier / reward for FinOps root-cause tasks. No LLM judge.

Matching follows the eval scorer (scripts/score.py) so that train reward and eval metric agree:
name = exact after trim, case-insensitive; type = case-insensitive with spaces/_/- normalized.

Unlike the official `root_cause_resource` metric (passes a *superset* of the ground truth), the
training reward penalizes extra resources: otherwise RL learns to list many resources.

Reward (RewardConfig.mode):
  "f1"        : F1 over (name, type) pairs
  "exact"     : 1 iff predicted set == GT set
  "f1+exact"  : 0.5 * F1 + 0.5 * exact          (default: dense signal + bonus for the clean answer)
Format gate: no parseable <answer>{"resource": [...]}</answer> in the final message -> format_penalty (default 0).
Optional step cost (off by default): reward -= step_cost * steps_used / max_steps, applied only when exact == 1,
so it can never make a wrong answer look better than a slower right one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class RewardConfig:
    mode: str = "f1+exact"
    format_penalty: float = 0.0
    step_cost: float = 0.0
    require_tags: bool = True       # training: the answer must be inside <answer></answer>


def norm_name(s) -> str:
    return str(s or "").strip().lower()


def norm_type(s) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s or "")).strip().lower()


def extract_answer(text: str, require_tags: bool = True):
    """Return the parsed JSON answer or None. With require_tags, only the last <answer>...</answer> counts."""
    if not text:
        return None
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    cands = [m[-1]] if m else []
    if not require_tags:
        cands += re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)[::-1]
    for c in cands:
        c = re.sub(r"^\s*```(?:json)?|```\s*$", "", c.strip()).strip()
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def _pairs(answer) -> set[tuple[str, str]] | None:
    if not (isinstance(answer, dict) and isinstance(answer.get("resource"), list)):
        return None
    out = set()
    for r in answer["resource"]:
        if not isinstance(r, dict):
            return None
        out.add((norm_name(r.get("name")), norm_type(r.get("type"))))
    return out


def score_answer(answer, gt: list[dict]) -> dict:
    """All metrics for one parsed answer vs GT [{'name','type'}]. Includes the official metric for reference."""
    g = {(norm_name(x["name"]), norm_type(x["type"])) for x in gt}
    p = _pairs(answer)
    if p is None:
        return dict(format_ok=0, exact=0, f1=0.0, precision=0.0, recall=0.0, official=0, n_pred=0, n_gt=len(g))
    tp = len(g & p)
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else (1.0 if not p else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(format_ok=1, exact=int(p == g), f1=f1, precision=prec, recall=rec,
                official=int(g <= p), n_pred=len(p), n_gt=len(g))


def compute_reward(final_text: str, gt: list[dict], steps_used: int = 0, max_steps: int = 20,
                   cfg: RewardConfig | None = None) -> dict:
    cfg = cfg or RewardConfig()
    ans = extract_answer(final_text, require_tags=cfg.require_tags)
    s = score_answer(ans, gt)
    if not s["format_ok"]:
        return dict(s, reward=cfg.format_penalty)
    if cfg.mode == "f1":
        r = s["f1"]
    elif cfg.mode == "exact":
        r = float(s["exact"])
    elif cfg.mode == "f1+exact":
        r = 0.5 * s["f1"] + 0.5 * s["exact"]
    else:
        raise ValueError(cfg.mode)
    if cfg.step_cost and s["exact"]:
        r -= cfg.step_cost * min(steps_used, max_steps) / max_steps
    return dict(s, reward=float(r))
