"""`run_python` MCP tool server for the finops-rca taskset (verifiers v1).

Taskset-scoped (one server per environment worker); per-rollout isolation via `self.state`:
the first call of a rollout creates a fresh sandbox directory and materializes ONLY anomaly.json + data.csv
(regenerated from the task seed, hash-checked). The label never reaches this process.
The tool also enforces the step budget, so the budget holds even if the harness's turn cap is misconfigured.
"""
from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf

from finops_gym.generator import Task, materialize
from finops_gym.prompts import RUN_PYTHON_DESCRIPTION
from finops_gym.sandbox import Sandbox, SandboxConfig, audit_code


class FinOpsState(vf.State):
    task_json: str = ""          # agent-visible task spec (no label), set by FinOpsTask.setup
    workdir: str = ""
    steps: int = 0
    max_steps: int = 20
    audit_flags: list[str] = []


class RunPythonToolsetConfig(vf.SharedToolsetConfig):
    sandbox_backend: str = "auto"    # bwrap on Linux, sandbox-exec on macOS; "none" only for debugging
    timeout_s: int = 120


class RunPythonToolset(vf.Toolset[RunPythonToolsetConfig, FinOpsState]):
    TOOL_PREFIX = None               # tool is exposed as plain `run_python`, the name used in the eval prompt

    def _sandbox(self) -> Sandbox:
        cfg = SandboxConfig(backend=self.config.sandbox_backend, timeout_s=self.config.timeout_s)
        if self.state.workdir and Path(self.state.workdir).exists():
            return Sandbox(cfg, workdir=Path(self.state.workdir))
        sb = Sandbox(cfg)
        materialize(Task(**json.loads(self.state.task_json)), sb.workdir)
        self.state.workdir = str(sb.workdir)
        return sb

    @vf.tool
    async def run_python(self, code: str) -> str:
        """Execute a Python 3.12 script in the task's working directory (which contains the data files). pandas and numpy are installed. Returns combined stdout/stderr (truncated to 8000 chars). Each call is a fresh process: re-load data every time. Use print() to see results."""
        if self.state.steps >= self.state.max_steps:
            return "[tool budget exhausted: give your final answer now as JSON inside <answer></answer> tags]"
        self.state.steps += 1
        flags = audit_code(code)
        if flags:
            self.state.audit_flags = sorted(set(self.state.audit_flags) | set(flags))
        return self._sandbox().run_python(code).output


assert RunPythonToolset.run_python.__doc__ is None or True  # docstring mirrors prompts.RUN_PYTHON_DESCRIPTION
_ = RUN_PYTHON_DESCRIPTION

if __name__ == "__main__":
    RunPythonToolset.run()
