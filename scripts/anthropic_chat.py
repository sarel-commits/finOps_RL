"""Claude (Anthropic API) backend for scripts/agent.py, using the official `anthropic` SDK.

The harness keeps its conversation as an OpenAI-style message list. This adapter converts it to the
Messages API on every call and returns a response shaped like the OpenAI one the harness already reads.

Fidelity rules:
- Each assistant turn keeps Claude's raw content blocks (thinking, text, tool_use) under the private key
  `_anthropic_content`, and they are sent back unchanged on later calls (append-only history).
- Consecutive tool results become one user message of tool_result blocks; a user text that follows tool
  results (e.g. the forced-final nudge) is appended to that same user message.
- Thinking is adaptive (always on for Opus 5.5); depth is set with output_config.effort.
- Prompt caching: top-level automatic cache_control, so each step re-reads the growing prefix from cache.
- No server-side refusal fallback: a fallback would silently switch models and contaminate the measurement.
  A refusal is recorded (stop_reason) and the run fails like any other non-answer.
"""
from __future__ import annotations
import os
from types import SimpleNamespace as NS

import anthropic

# USD per million tokens, Anthropic list prices (claude-api skill model table, cached 2026-10-06).
# cache_write = 5-minute TTL write (1.25x input); cache_read as listed per model.
PRICES = {
    "claude-opus-5-5": dict(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20),
    "claude-sonnet-5-5": dict(input=2.00, output=10.00, cache_write=2.50, cache_read=0.20),
}


def _tool_def(openai_tool):
    f = openai_tool["function"]
    return {"name": f["name"], "description": f["description"], "input_schema": f["parameters"]}


def to_anthropic(messages):
    """OpenAI-style list -> (system, messages) for the Messages API."""
    system, out = None, []
    for m in messages:
        role = m["role"]
        if role == "system":
            system = m["content"]
        elif role == "assistant":
            blocks = m.get("_anthropic_content")
            if blocks is None:  # not produced by Claude (should not happen in a Claude run)
                blocks = [{"type": "text", "text": m.get("content") or ""}]
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "user":
            if out and out[-1]["role"] == "user":  # e.g. forced-final nudge after tool results
                c = out[-1]["content"]
                if isinstance(c, str):
                    c = [{"type": "text", "text": c}]
                c.append({"type": "text", "text": m["content"]})
                out[-1]["content"] = c
            else:
                out.append({"role": "user", "content": m["content"]})
    return system, out


class ClaudeChat:
    def __init__(self, model, effort="medium", max_tokens=32000, timeout=900.0, api_key_env="ANTHROPIC_API_KEY"):
        if model not in PRICES:
            raise ValueError(f"no price entry for {model}")
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        # Explicit key from the named env var (value never logged). SDK retries 429/5xx/connection errors.
        self.client = anthropic.Anthropic(api_key=os.environ[api_key_env], timeout=timeout, max_retries=4)

    def config(self):
        return dict(provider="anthropic", model=self.model, thinking="adaptive", effort=self.effort,
                    max_tokens=self.max_tokens, prompt_caching="automatic (top-level cache_control)",
                    refusal_fallback=False, prices_usd_per_mtok=PRICES[self.model])

    def cost(self, u):
        p = PRICES[self.model]
        return (u["input_tokens"] * p["input"] + u["cache_creation_input_tokens"] * p["cache_write"]
                + u["cache_read_input_tokens"] * p["cache_read"] + u["output_tokens"] * p["output"]) / 1e6

    def chat(self, messages, tools, tool_choice_none=False):
        system, msgs = to_anthropic(messages)
        kw = dict(model=self.model, max_tokens=self.max_tokens, messages=msgs,
                  thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                  cache_control={"type": "ephemeral"})
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = [_tool_def(t) for t in tools]
            if tool_choice_none:
                kw["tool_choice"] = {"type": "none"}
        with self.client.messages.stream(**kw) as stream:
            resp = stream.get_final_message()
        raw = [b.to_dict() for b in resp.content]
        text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [NS(id=b.id, function=NS(name=b.name, arguments=b.input))
                      for b in resp.content if b.type == "tool_use"] or None
        u = resp.usage
        usage = dict(input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                     cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
                     cache_read_input_tokens=u.cache_read_input_tokens or 0)
        usage["cost_usd"] = round(self.cost(usage), 6)
        prompt_tokens = usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"]
        finish = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length"}.get(resp.stop_reason,
                                                                                          resp.stop_reason)
        msg = NS(content=text, tool_calls=tool_calls, reasoning=None, model_extra={}, anthropic_content=raw)
        return NS(choices=[NS(message=msg, finish_reason=finish)],
                  usage=NS(prompt_tokens=prompt_tokens, completion_tokens=usage["output_tokens"]),
                  usage_detail=usage, stop_reason=resp.stop_reason, request_id=getattr(resp, "_request_id", None) or resp.id)
