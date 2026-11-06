#!/usr/bin/env python
"""Health check: chat completion + native tool calling round-trip. Saves results/sanity/healthcheck.json"""
import json, sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
import openai
from agent import RUN_PYTHON_TOOL, SAMPLING
c = openai.OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local", timeout=600)
S = SAMPLING["nothink"]; out = {}
def call(msgs, tools=None, samp=S, max_tokens=512):
    t = time.time()
    r = c.chat.completions.create(model="default_model", messages=msgs, max_tokens=max_tokens, temperature=samp["temperature"],
                                  top_p=samp["top_p"], extra_body=samp["extra_body"], seed=1, **({"tools": tools} if tools else {}))
    return r, time.time() - t
r, dt = call([{"role": "user", "content": "Say OK and nothing else."}])
out["chat"] = dict(content=r.choices[0].message.content, usage=r.usage.model_dump(), latency_s=round(dt, 2))
print("chat:", repr(r.choices[0].message.content), r.usage, f"{dt:.1f}s")
r, dt = call([{"role": "user", "content": "What is 17*23? Answer briefly."}], samp=SAMPLING["think"], max_tokens=2048)
m = r.choices[0].message
out["thinking"] = dict(content=m.content, reasoning=(m.model_extra or {}).get("reasoning") or (m.model_extra or {}).get("reasoning_content"),
                       extra_keys=list((m.model_extra or {}).keys()), usage=r.usage.model_dump())
print("thinking: content=", repr(m.content)[:200], "| extra keys:", list((m.model_extra or {}).keys()))
msgs = [{"role": "system", "content": "Use the run_python tool to compute answers."},
        {"role": "user", "content": "Use Python to compute the sum of squares of 1..100, then tell me the result."}]
r, dt = call(msgs, tools=[RUN_PYTHON_TOOL])
m = r.choices[0].message
tcs = m.tool_calls or []
out["tool_call_1"] = dict(content=m.content, finish=r.choices[0].finish_reason,
                          tool_calls=[dict(id=t.id, name=t.function.name, arguments=t.function.arguments) for t in tcs])
print("tool call:", r.choices[0].finish_reason, [(t.function.name, t.function.arguments) for t in tcs], "| content:", repr(m.content)[:200])
if tcs:
    msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [dict(id=t.id, type="function", function=dict(name=t.function.name, arguments=t.function.arguments)) for t in tcs]})
    msgs.append({"role": "tool", "tool_call_id": tcs[0].id, "content": "338350"})
    r, dt = call(msgs, tools=[RUN_PYTHON_TOOL])
    out["tool_call_2"] = dict(content=r.choices[0].message.content, tool_calls=bool(r.choices[0].message.tool_calls))
    print("after tool result:", repr(r.choices[0].message.content)[:200])
out["native_tool_calls_ok"] = bool(tcs) and json.loads(tcs[0].function.arguments).get("code") is not None
json.dump(out, open("results/sanity/healthcheck.json", "w"), indent=1, default=str)
print("NATIVE_TOOL_CALLS_OK" if out["native_tool_calls_ok"] else "NATIVE_TOOL_CALLS_BROKEN")
