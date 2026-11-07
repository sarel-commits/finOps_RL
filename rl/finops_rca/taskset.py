"""finops-rca: ITBench-Lite-style FinOps cost-anomaly root-cause analysis as a verifiers v1 taskset.

  splits: train | val | test_synth (procedural, regenerated from seed) | itbench_lite (file-backed, eval only)
  agent:  null harness (plain chat loop) + one MCP tool `run_python` (sandboxed, see servers/run_python.py)
  reward: deterministic, finops_gym.reward (F1 + exact-set bonus, format gate). No LLM judge.

The label lives on FinOpsData (evaluator side only; verifiers never injects extra TaskData fields into the
prompt) and is never written into the sandbox.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import verifiers.v1 as vf

from finops_gym import tasks as T
from finops_gym.prompts import SYSTEM, task_prompt
from finops_gym.reward import RewardConfig, compute_reward
from finops_rca.servers.run_python import FinOpsState, RunPythonToolset, RunPythonToolsetConfig


class FinOpsData(vf.TaskData):
    task_json: str                   # agent-visible spec: seed, anomaly, csv hash, source_dir (no label)
    label: list[dict]
    regime: str = ""
    mechanism: str = ""


class FinOpsTaskConfig(vf.TaskConfig):
    max_steps: int = 20
    reward_mode: str = "f1+exact"    # f1 | exact | f1+exact
    step_cost: float = 0.0           # >0 rewards shorter correct solutions (applied only when exact)


class FinOpsTask(vf.Task[FinOpsData, FinOpsState, FinOpsTaskConfig]):
    async def setup(self, trace: vf.Trace) -> None:
        trace.state.task_json = self.data.task_json
        trace.state.max_steps = self.config.max_steps

    async def finalize(self, trace: vf.Trace) -> None:
        import shutil
        if trace.state.workdir:
            shutil.rmtree(trace.state.workdir, ignore_errors=True)
        trace.info["steps"] = trace.state.steps
        trace.info["audit_flags"] = list(trace.state.audit_flags)

    @vf.stop
    async def step_budget(self, trace: vf.Trace) -> bool:
        # max_steps tool turns + 1 answer turn
        return trace.num_turns >= self.config.max_steps + 1

    def _score(self, trace: vf.Trace) -> dict:
        cfg = RewardConfig(mode=self.config.reward_mode, step_cost=self.config.step_cost)
        return compute_reward(trace.last_reply or "", self.data.label, steps_used=trace.state.steps,
                              max_steps=self.config.max_steps, cfg=cfg)

    @vf.reward(weight=1.0)
    async def root_cause(self, trace: vf.Trace) -> float:
        r = self._score(trace)
        trace.info["score"] = r
        return r["reward"]

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Gold check for vf-validate: the label, formatted as the agent must answer, scores exactly 1.0;
        and the materialized data reproduces the manifest hash (generator determinism)."""
        import tempfile
        from finops_gym.generator import Task as GTask, materialize
        gold = "<answer>" + json.dumps({"resource": self.data.label}) + "</answer>"
        ok_reward = compute_reward(gold, self.data.label)["reward"] == 1.0
        with tempfile.TemporaryDirectory() as d:
            materialize(GTask(**json.loads(self.data.task_json)), Path(d), verify=True)
        return ok_reward

    @vf.metric
    async def details(self, trace: vf.Trace) -> dict[str, float]:
        r = self._score(trace)
        return {"exact": float(r["exact"]), "f1": r["f1"], "official_root_cause_resource": float(r["official"]),
                "format_ok": float(r["format_ok"]), "steps": float(trace.state.steps),
                "audit_flagged": float(bool(trace.state.audit_flags))}


class FinOpsConfig(vf.TasksetConfig):
    split: str = "train"
    manifest: str | None = None      # explicit manifest path, e.g. rl/data/train_calibrated.jsonl
    task: FinOpsTaskConfig = FinOpsTaskConfig()
    tools: RunPythonToolsetConfig = RunPythonToolsetConfig()


class FinOpsRCATaskset(vf.Taskset[FinOpsTask, FinOpsConfig]):
    @classmethod
    def toolsets(cls, config: FinOpsConfig) -> list[vf.Toolset]:
        return [RunPythonToolset(config.tools)]

    def load(self):
        path = Path(self.config.manifest) if self.config.manifest else T.manifest_path(self.config.split)
        training = self.config.split in ("train",) or "train" in path.name
        for t in T.load(path, for_training=training):
            spec = {k: v for k, v in asdict(t).items() if k != "label"}
            spec["label"] = []       # the tool server only ever sees an empty label
            yield FinOpsTask(
                FinOpsData(id=t.task_id, name=t.task_id, prompt=task_prompt(t.anomaly), system_prompt=SYSTEM,
                           task_json=json.dumps(spec), label=t.label, regime=t.regime, mechanism=t.mechanism),
                self.config.task)
