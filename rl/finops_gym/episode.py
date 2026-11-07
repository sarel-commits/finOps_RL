"""Framework-agnostic episode runner against any OpenAI-compatible endpoint (local mlx_lm.server, vLLM, ...).

Used for: difficulty calibration (pass@k per task with the current policy), smoke tests, and as the reference
implementation the trainer adapters must match (same prompt, tool, step budget, answer extraction, reward).

Episode protocol (identical to the baseline eval agent, native tool calling):
  system + user(task prompt) -> up to max_steps assistant turns that call run_python -> a final assistant message
  without tool calls that contains <answer>{...}</answer>. Running out of steps = no answer = reward 0.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from .generator import Task, materialize
from .prompts import RUN_PYTHON_TOOL, SYSTEM, task_prompt
from .reward import RewardConfig, compute_reward
from .sandbox import Sandbox, SandboxConfig, audit_code


@dataclass
class EpisodeConfig:
    max_steps: int = 20
    max_tokens: int = 4096
    sampling: dict = field(default_factory=lambda: dict(
        temperature=0.7, top_p=0.8,
        extra_body=dict(top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0,
                        chat_template_kwargs={"enable_thinking": False})))
    reward: RewardConfig = field(default_factory=RewardConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)


def run_episode(client, model: str, task: Task, cfg: EpisodeConfig | None = None, seed: int | None = None) -> dict:
    cfg = cfg or EpisodeConfig()
    t0 = time.time()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_prompt(task.anomaly)}]
    steps, final, err, n_tok_in, n_tok_out, flags = 0, None, None, 0, 0, []
    with Sandbox(cfg.sandbox) as sb:
        materialize(task, sb.workdir)
        for turn in range(cfg.max_steps + 1):
            kw = dict(model=model, messages=msgs, max_tokens=cfg.max_tokens, tools=[RUN_PYTHON_TOOL], **cfg.sampling)
            if seed is not None:
                kw["seed"] = seed * 1000 + turn
            r = client.chat.completions.create(**kw)
            if r.usage:
                n_tok_in += r.usage.prompt_tokens or 0
                n_tok_out += r.usage.completion_tokens or 0
            m = r.choices[0].message
            calls = m.tool_calls or []
            am = {"role": "assistant", "content": m.content or ""}
            if calls:
                am["tool_calls"] = [{"id": c.id, "type": "function",
                                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                    for c in calls]
            msgs.append(am)
            if not calls:
                final = m.content or ""
                break
            if turn == cfg.max_steps:          # budget exhausted while still calling tools
                err = "max_steps"
                break
            steps += 1
            for c in calls:
                try:
                    code = json.loads(c.function.arguments).get("code", "")
                except Exception:
                    code = None
                if c.function.name != "run_python" or code is None:
                    out = f"[error: unknown tool or unparseable arguments: {c.function.name}]"
                else:
                    flags += audit_code(code)
                    out = sb.run_python(code).output
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
    rw = compute_reward(final or "", task.label, steps_used=steps, max_steps=cfg.max_steps, cfg=cfg.reward)
    return dict(task_id=task.task_id, seed=seed, steps=steps, error=err, reward=rw, final=final,
                audit_flags=sorted(set(flags)), tokens_in=n_tok_in, tokens_out=n_tok_out,
                wall_s=round(time.time() - t0, 1), messages=msgs, label=task.label,
                config=dict(max_steps=cfg.max_steps, reward=asdict(cfg.reward), sandbox_backend=sb.backend))
