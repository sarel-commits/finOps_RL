"""Minimal tool-using agent harness against a local OpenAI-compatible server.

Modes:
  native : OpenAI `tools` / `tool_calls` (server parses Qwen XML tool calls)
  react  : text protocol fallback — model emits one ```python block per turn,
           harness executes it and returns "Observation:". Results produced in
           this mode are marked protocol=react in every trajectory/score file.

Each item writes <out_dir>/trajectory.json (checkpoint: skipped if status=done).
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, tempfile, time, traceback, uuid
from pathlib import Path
import urllib.request
from types import SimpleNamespace as NS

import openai

ROOT = Path.home() / "finops-eval"
PY = str(ROOT / ".venv" / "bin" / "python")
TOOL_TIMEOUT_S = 120
TOOL_OUT_MAX = 8000

SAMPLING = {
    "nothink": dict(temperature=0.7, top_p=0.8, extra_body=dict(top_k=20, min_p=0.0, presence_penalty=1.5,
                                                                repetition_penalty=1.0,
                                                                chat_template_kwargs={"enable_thinking": False})),
    "think": dict(temperature=1.0, top_p=0.95, extra_body=dict(top_k=20, min_p=0.0, presence_penalty=0.0,
                                                               repetition_penalty=1.0,
                                                               chat_template_kwargs={"enable_thinking": True})),
    # other vendors' models: provider-default sampling (Qwen model-card values are Qwen-specific), thinking off
    "provider_nothink": dict(temperature=None, top_p=None,
                             extra_body=dict(chat_template_kwargs={"enable_thinking": False})),
}

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": ("Execute a Python 3.12 script in the task's working directory (which contains the data files). "
                        "pandas and numpy are installed. Returns combined stdout/stderr (truncated to 8000 chars). "
                        "Each call is a fresh process: re-load data every time. Use print() to see results."),
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python source code to run"}},
                       "required": ["code"]},
    },
}

SYSTEM_NATIVE = (
    "You are an expert FinOps (cloud financial operations) analyst agent. You can execute Python with the run_python tool "
    "to inspect the data files in your working directory. Always compute answers from the data with code; never guess or "
    "invent numbers or identifiers. When you are done, give your final answer as a single JSON object wrapped in "
    "<answer></answer> tags, exactly matching the required format, and do not call any tool in that final message."
)
SYSTEM_NATIVE_V2 = """You are an expert FinOps (cloud financial operations) analyst agent. You can execute Python with the run_python tool to inspect the data files in your working directory. Always compute answers from the data with code; never guess or invent numbers or identifiers.

You have a strict budget of {max_steps} tool calls. Work efficiently:
- Plan your analysis before writing code, and do several computations in one tool call when you can.
- Do not re-run an analysis that you have already done with small variations. If a result already answers the question, stop and answer.
- An empty result is a valid answer. If the data shows that nothing qualifies, say so; do not keep searching.
- It is always better to give your best-supported answer than to run out of steps without one.

When you are done, give your final answer as a single JSON object wrapped in <answer></answer> tags, exactly matching the required format, and do not call any tool in that final message."""

NUDGE_STEP = "[Step {k} of {max_steps} used.]"
NUDGE_WARN = ("[Budget warning: only 5 tool calls remain. Finish the analysis you need and give your final answer soon. "
              "Do not start new exploratory analyses.]")
NUDGE_FINAL = ("[The tool budget is exhausted. You cannot run more code. Based on the analysis you have already done, "
               "give your best final answer now as JSON inside <answer></answer> tags.]")

SYSTEM_REACT = (
    "You are an expert FinOps (cloud financial operations) analyst agent. You can execute Python to inspect the data files "
    "in your working directory (pandas and numpy are installed). To run code, reply with exactly ONE fenced ```python code "
    "block and nothing after it; you will receive its stdout/stderr as an Observation. Each run is a fresh process. "
    "Always compute answers from the data; never guess or invent numbers or identifiers. When you are done, reply with your "
    "final answer as a single JSON object wrapped in <answer></answer> tags, exactly matching the required format, and no code block."
)


def run_python(code: str, cwd: Path) -> tuple[str, float, bool]:
    t0 = time.time()
    fn = cwd / f"_tool_{uuid.uuid4().hex[:8]}.py"
    fn.write_text(code)
    timed_out = False
    try:
        p = subprocess.run([PY, "-I", fn.name], cwd=cwd, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S,
                           env={"PATH": "/usr/bin:/bin", "HOME": str(cwd), "MPLBACKEND": "Agg"})
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        if p.returncode != 0:
            out += f"\n[exit code {p.returncode}]"
    except subprocess.TimeoutExpired:
        out, timed_out = f"[tool timeout after {TOOL_TIMEOUT_S}s]", True
    finally:
        fn.unlink(missing_ok=True)
    if len(out) > TOOL_OUT_MAX:
        out = out[:TOOL_OUT_MAX // 2] + f"\n...[truncated {len(out) - TOOL_OUT_MAX} chars]...\n" + out[-TOOL_OUT_MAX // 2:]
    return out if out.strip() else "[no output — use print()]", time.time() - t0, timed_out


def extract_answer(text: str):
    """Return (parsed_json_or_None, raw_answer_text_or_None)."""
    if not text:
        return None, None
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    cands = []
    if m:
        cands.append(m[-1])
    cands += re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)[::-1]
    # last balanced {...}
    depth, start, objs = 0, None, []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                objs.append(text[start:i + 1])
    cands += objs[::-1]
    for c in cands:
        c = c.strip()
        c2 = re.sub(r"^```(?:json)?|```$", "", c).strip()
        for attempt in (c2, c2.replace("'", '"')):
            try:
                return json.loads(attempt), c
            except Exception:
                pass
    return None, (cands[0] if cands else None)


def parse_raw_tool_call(text: str):
    """Fallback: parse Qwen XML tool call if the server left it in content."""
    m = re.search(r"<function=run_python>\s*<parameter=code>\n?(.*?)\n?</parameter>", text or "", re.S)
    return m.group(1) if m else None


def server_alive(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_server(base_url: str):
    if server_alive(base_url):
        return
    print("[harness] server down — restarting", flush=True)
    subprocess.run(["bash", str(ROOT / "scripts" / "server.sh"), "restart"], check=False)
    for _ in range(120):
        if server_alive(base_url):
            return
        time.sleep(5)
    raise RuntimeError("server did not come back")


class Agent:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", model="default_model", protocol="native",
                 thinking=False, max_steps=20, max_tokens=None, request_timeout=1200, retries=3,
                 task_time_budget=3600, budget_nudges=False, prompt_version="v1", api_key_env=None, stream=False, sampling_profile=None):
        self.base_url, self.model, self.protocol = base_url, model, protocol
        self.thinking = thinking
        self.sampling_profile = sampling_profile or ("think" if thinking else "nothink")
        self.samp = SAMPLING[self.sampling_profile]
        self.max_steps = max_steps
        self.max_tokens = max_tokens or (16384 if thinking else 4096)
        api_key = os.environ[api_key_env] if api_key_env else "local"
        self.api_key_env, self.stream = api_key_env, stream
        self.local = any(h in base_url for h in ("127.0.0.1", "localhost"))
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout, max_retries=0)
        self.retries = retries
        self.task_time_budget = task_time_budget
        self.budget_nudges = budget_nudges
        self.prompt_version = prompt_version
        if prompt_version == "v2" and protocol != "native":
            raise ValueError("prompt v2 is defined for the native protocol only")

    def config(self):
        return dict(base_url=self.base_url, model=self.model, protocol=self.protocol, thinking=self.thinking,
                    sampling={k: v for k, v in self.samp.items()}, max_steps=self.max_steps,
                    max_tokens_per_turn=self.max_tokens, tool_timeout_s=TOOL_TIMEOUT_S,
                    task_time_budget_s=self.task_time_budget, budget_nudges=self.budget_nudges,
                    prompt_version=self.prompt_version, api_key_env=self.api_key_env, stream=self.stream,
                    sampling_profile=self.sampling_profile)

    def _chat(self, messages, seed, tools=True):
        kw = dict(model=self.model, messages=messages, max_tokens=self.max_tokens, seed=seed,
                  extra_body=self.samp["extra_body"])
        for k in ("temperature", "top_p"):
            if self.samp[k] is not None:
                kw[k] = self.samp[k]
        if self.protocol == "native" and tools:
            kw["tools"] = [RUN_PYTHON_TOOL]
        last = None
        for a in range(self.retries):
            try:
                if self.local:
                    ensure_server(self.base_url)
                t0 = time.time()
                r = self._stream(kw) if self.stream else self.client.chat.completions.create(**kw)
                return r, time.time() - t0
            except Exception as e:  # timeout / connection / 5xx
                last = e
                print(f"[harness] request error (attempt {a+1}): {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(5 * (a + 1))
        raise last

    def _stream(self, kw):
        """Streaming request, reassembled into the shape of a non-streaming response."""
        st = self.client.chat.completions.create(**kw, stream=True, stream_options={"include_usage": True})
        content, reas, tcs, usage, fin = "", "", {}, None, None
        for ch in st:
            if getattr(ch, "usage", None):
                usage = ch.usage
            if not ch.choices:
                continue
            d = ch.choices[0].delta
            fin = ch.choices[0].finish_reason or fin
            content += d.content or ""
            ex = d.model_extra or {}
            reas += ex.get("reasoning") or ex.get("reasoning_content") or ""
            for tc in d.tool_calls or []:
                e = tcs.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                e["id"] = tc.id or e["id"]
                if tc.function:
                    e["name"] += tc.function.name or ""
                    e["args"] += tc.function.arguments or ""
        tool_calls = [NS(id=e["id"] or f"call_{i}", function=NS(name=e["name"], arguments=e["args"]))
                      for i, e in sorted(tcs.items())] or None
        msg = NS(content=content, tool_calls=tool_calls, reasoning=reas or None, model_extra={})
        return NS(choices=[NS(message=msg, finish_reason=fin)], usage=usage)

    def run(self, prompt: str, files: list[Path], out_dir: Path, seed: int, meta: dict | None = None):
        out_dir.mkdir(parents=True, exist_ok=True)
        tf = out_dir / "trajectory.json"
        if tf.exists():
            try:
                old = json.loads(tf.read_text())
                if old.get("status") == "done":
                    return old
            except Exception:
                pass
        work = Path(tempfile.mkdtemp(prefix="finops_ws_", dir=str(ROOT / "tmp")))
        for f in files:
            shutil.copy(f, work / Path(f).name)
        if self.protocol == "native":
            sysmsg = SYSTEM_NATIVE_V2.format(max_steps=self.max_steps) if self.prompt_version == "v2" else SYSTEM_NATIVE
        else:
            sysmsg = SYSTEM_REACT
        messages = [{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}]
        traj = dict(meta=meta or {}, seed=seed, config=self.config(), prompt=prompt, steps=[], status="running",
                    started=time.time())
        tok_in = tok_out = 0
        gen_time = 0.0
        final_text, err = None, None
        forced_final = False
        n_tool_calls = n_tool_errors = n_tool_timeouts = n_raw_parsed = 0
        t_start = time.time()
        try:
            for step in range(self.max_steps):
                if time.time() - t_start > self.task_time_budget:
                    err = "timeout_budget"
                    break
                r, dt = self._chat(messages, seed + step)
                gen_time += dt
                msg = r.choices[0].message
                u = r.usage
                pin, pout = (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)
                tok_in += pin
                tok_out += pout
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) \
                    or (msg.model_extra or {}).get("reasoning") or (msg.model_extra or {}).get("reasoning_content")
                srec = dict(step=step, latency_s=round(dt, 2), prompt_tokens=pin, completion_tokens=pout,
                            finish_reason=r.choices[0].finish_reason, content=content, reasoning=reasoning,
                            tool_calls=[], observations=[])
                traj["steps"].append(srec)
                code_list = []
                if self.protocol == "native":
                    tcs = msg.tool_calls or []
                    for tc in tcs:
                        try:
                            args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                            code_list.append((tc.id, tc.function.name, args.get("code", "")))
                        except Exception as e:
                            code_list.append((tc.id, tc.function.name, None))
                    if not tcs and "<tool_call>" in content:
                        c = parse_raw_tool_call(content)
                        if c is not None:
                            n_raw_parsed += 1
                            code_list.append(("raw_" + uuid.uuid4().hex[:6], "run_python", c))
                            srec["raw_tool_call_parsed_by_harness"] = True
                    am = {"role": "assistant", "content": content}
                    if reasoning:
                        am["reasoning_content"] = reasoning
                    if code_list:
                        am["tool_calls"] = [{"id": cid, "type": "function",
                                             "function": {"name": nm, "arguments": json.dumps({"code": c or ""})}}
                                            for cid, nm, c in code_list]
                    messages.append(am)
                else:
                    blocks = re.findall(r"```python\s*\n(.*?)```", content, re.S)
                    if blocks and "<answer>" not in content:
                        code_list.append((None, "run_python", blocks[0]))
                    am = {"role": "assistant", "content": content}
                    if reasoning:
                        am["reasoning_content"] = reasoning
                    messages.append(am)

                if not code_list:
                    final_text = content
                    if r.choices[0].finish_reason == "length" and "<answer>" not in content:
                        err = "max_tokens_no_answer"
                    break
                for cid, nm, code in code_list:
                    n_tool_calls += 1
                    if nm != "run_python" or code is None:
                        obs, tdt, to = f"[error: unknown tool or unparseable arguments: {nm}]", 0.0, False
                        n_tool_errors += 1
                    else:
                        obs, tdt, to = run_python(code, work)
                        if "Traceback" in obs or "[exit code" in obs:
                            n_tool_errors += 1
                        n_tool_timeouts += int(to)
                    srec["tool_calls"].append(dict(id=cid, name=nm, code=code))
                    srec["observations"].append(dict(output=obs, seconds=round(tdt, 2), timed_out=to))
                    if self.budget_nudges and cid == code_list[-1][0]:
                        nudge = "\n\n" + NUDGE_STEP.format(k=step + 1, max_steps=self.max_steps)
                        if self.max_steps - (step + 1) == 5:
                            nudge += "\n" + NUDGE_WARN
                        obs = obs + nudge
                        srec["nudge"] = nudge.strip()
                    if self.protocol == "native":
                        messages.append({"role": "tool", "tool_call_id": cid, "content": obs})
                    else:
                        messages.append({"role": "user", "content": "Observation:\n" + obs})
            else:
                err = "max_steps"
                if self.budget_nudges:
                    # one extra turn with tools disabled to force a final answer
                    messages.append({"role": "user", "content": NUDGE_FINAL})
                    r, dt = self._chat(messages, seed + self.max_steps, tools=False)
                    gen_time += dt
                    msg = r.choices[0].message
                    u = r.usage
                    pin, pout = (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)
                    tok_in += pin
                    tok_out += pout
                    content = msg.content or ""
                    reasoning = getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning")
                    traj["forced_final_step"] = dict(latency_s=round(dt, 2), prompt_tokens=pin, completion_tokens=pout,
                                                     finish_reason=r.choices[0].finish_reason, content=content,
                                                     reasoning=reasoning, tool_calls_ignored=len(msg.tool_calls or []))
                    am = {"role": "assistant", "content": content}
                    if reasoning:
                        am["reasoning_content"] = reasoning
                    messages.append(am)
                    final_text, forced_final = content, True
                    err = None if extract_answer(content)[0] is not None else "forced_final_no_answer"
        except Exception as e:
            err = f"exception: {type(e).__name__}: {str(e)[:300]}"
            traj["traceback"] = traceback.format_exc()
        parsed, raw = extract_answer(final_text or "")
        if final_text is None and err is None:
            err = "no_final"
        traj.update(status="done", error=err, forced_final=forced_final, final_text=final_text, answer=parsed, answer_raw=raw,
                    parse_ok=parsed is not None, tokens_in=tok_in, tokens_out=tok_out,
                    wall_s=round(time.time() - t_start, 2), gen_s=round(gen_time, 2),
                    decode_tps_approx=round(tok_out / gen_time, 2) if gen_time else None,
                    n_steps=len(traj["steps"]), n_tool_calls=n_tool_calls, n_tool_errors=n_tool_errors,
                    n_tool_timeouts=n_tool_timeouts, n_raw_tool_calls_parsed_by_harness=n_raw_parsed,
                    leak_check_ground_truth=any("ground_truth" in (tc.get("code") or "")
                                                for s in traj["steps"] for tc in s["tool_calls"]),
                    messages=messages)
        if err and err.startswith("exception"):
            traj["status"] = "error"  # not checkpointed as done -> retried on resume
        tf.write_text(json.dumps(traj, indent=1, default=str))
        shutil.rmtree(work, ignore_errors=True)
        return traj
