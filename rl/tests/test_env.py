"""Core invariants of the RL environment. Run: rl/.venv-vf/bin/python -m pytest rl/tests -q"""
import json
import sys
from pathlib import Path

import pytest

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL))

from finops_gym import oracle, tasks as T  # noqa: E402
from finops_gym.generator import GenConfig, generate, materialize  # noqa: E402
from finops_gym.prompts import SYSTEM, TASK  # noqa: E402
from finops_gym.reward import RewardConfig, compute_reward  # noqa: E402
from finops_gym.sandbox import Sandbox, audit_code  # noqa: E402

GT = [{"name": "AbC", "type": "Instance Family"}, {"name": "x1", "type": "Application"}]


def ans(res):
    return "<answer>" + json.dumps({"resource": res}) + "</answer>"


# ---------- reward ----------
def test_reward_exact_and_normalization():
    r = compute_reward(ans([{"name": " abc ", "type": "instance_family"}, {"name": "X1", "type": "application"}]), GT)
    assert r["exact"] == 1 and r["reward"] == 1.0


def test_reward_superset_is_penalized_but_official_passes():
    r = compute_reward(ans(GT + [{"name": "extra", "type": "Application"}]), GT)
    assert r["official"] == 1 and r["exact"] == 0 and r["reward"] < 1.0


def test_reward_format_gate():
    assert compute_reward(json.dumps({"resource": GT}), GT)["reward"] == 0.0          # no tags
    assert compute_reward("<answer>{not json}</answer>", GT)["reward"] == 0.0
    assert compute_reward("", GT)["format_ok"] == 0


def test_step_cost_only_applies_to_correct_answers():
    cfg = RewardConfig(step_cost=0.2)
    right = compute_reward(ans(GT), GT, steps_used=20, cfg=cfg)["reward"]
    wrong = compute_reward(ans(GT[:1]), GT, steps_used=1, cfg=cfg)["reward"]
    assert right == pytest.approx(0.8) and wrong == pytest.approx(0.5 * (2 / 3))


# ---------- generator ----------
@pytest.mark.parametrize("seed", [1, 2, 3, 1_000_007])
def test_generator_deterministic(seed):
    a, b = generate(seed), generate(seed)
    assert a.csv_sha256 == b.csv_sha256 and a.label == b.label


@pytest.mark.parametrize("seed", range(10, 30))
def test_labels_valid_for_regime(seed):
    t, df = generate(seed, want_frame=True)
    gt = {(x["name"].lower(), x["type"]) for x in t.label}
    if t.regime == "clean":
        assert {(n.lower(), ty) for n, ty in oracle.detect(df, t.anomaly)} == gt
    else:
        d = oracle.top_deltas(df, t.anomaly)
        assert {str(n).lower() for n in d.index[: len(gt)]} == {n for n, _ in gt}


def test_names_unique_case_insensitive():
    _, df = generate(5, want_frame=True)
    names = set(df.instance_family) | set(df.application) | set(df.cloud_service)
    assert len({n.lower() for n in names}) == len(names)


def test_materialize_writes_only_agent_files(tmp_path):
    t = generate(11)
    materialize(t, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["anomaly.json", "data.csv"]
    text = (tmp_path / "data.csv").read_text()
    assert "root_cause" not in text and "ground_truth" not in text


# ---------- manifests / contamination ----------
def test_training_loader_rejects_eval_tasks(tmp_path):
    itb = T.manifest_path("itbench_lite")
    if not itb.exists():
        pytest.skip("run build_dataset.py first")
    bad = tmp_path / "train_bad.jsonl"
    bad.write_text(itb.read_text())
    with pytest.raises(ValueError):
        T.load(bad, for_training=True)


def test_split_seeds_disjoint():
    if not T.manifest_path("train").exists():
        pytest.skip("run build_dataset.py first")
    s = {k: {t.seed for t in T.load(T.manifest_path(k))} for k in ("train", "val", "test_synth")}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test_synth"]) and not (s["val"] & s["test_synth"])


# ---------- sandbox ----------
def test_sandbox_isolation():
    with Sandbox() as sb:
        materialize(generate(3), sb.workdir)
        assert "rows" in sb.run_python("import pandas as pd;print('rows', len(pd.read_csv('data.csv')))").output
        secret = RL / "data" / "train.jsonl"
        r = sb.run_python(f"print(open({str(secret)!r}).read()[:20])")
        assert r.error and "seed" not in r.output
        r = sb.run_python("import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080',timeout=2)")
        assert r.error


def test_audit_flags():
    assert audit_code("open('../../data/train.jsonl')")
    assert not audit_code("import pandas as pd\ndf=pd.read_csv('data.csv')")


# ---------- prompt parity with the eval harness ----------
def test_prompts_match_eval_harness():
    pytest.importorskip("openai")
    sys.path.insert(0, str(RL.parent / "scripts"))
    import agent  # the baseline eval harness
    import run_bench
    assert SYSTEM == agent.SYSTEM_NATIVE
    assert TASK == run_bench.ITB_PROMPT
    from finops_gym.prompts import RUN_PYTHON_TOOL
    assert RUN_PYTHON_TOOL == agent.RUN_PYTHON_TOOL
