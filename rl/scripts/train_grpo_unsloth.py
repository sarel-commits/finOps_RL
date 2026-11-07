#!/usr/bin/env python
"""Single-GPU GRPO for finops-rca: LoRA on Qwen3.8-27B-FP8 with Unsloth (vLLM and the trainer share ONE FP8 copy of
the weights on one H100; vLLM frees its KV cache while the trainer runs - "standby").

The environment is the same one prime-rl uses (finops_gym: generator, sandbox, reward, prompts) and the episode
protocol is finops_gym/episode.py's (system + task prompt, native run_python tool calls, <=20 tool steps, 21 model
turns, answer in <answer>...</answer>, reward 0.5*F1 + 0.5*exact). Only the trainer differs (prime-rl needs >=2 GPUs).

One step:
  1. rollouts   tasks_per_step x group episodes, multi-turn, all episodes batched per turn through vLLM with the
                current LoRA. Token-in/token-out: the sequence the trainer sees is exactly what vLLM sampled, plus the
                chat-template tokens of the tool results (masked out of the loss).
  2. reward     finops_gym.reward; advantage = reward - group mean (Dr. GRPO, no std division); groups whose rewards
                are all equal carry no signal and are skipped (no forward/backward spent on them).
  3. update     token-level policy gradient over the model's own tokens, normalised by the batch's token count, with a
                truncated importance weight min(exp(logp_train - logp_vllm), cap): vLLM's FP8 kernels and the
                training kernels give slightly different log-probs. AdamW on the LoRA parameters only.
  4. sync       the new LoRA is handed to vLLM in memory (model.load_lora(load_tensors=True)).

  ~/.venv-unsloth/bin/python rl/scripts/train_grpo_unsloth.py --smoke      # 2 tasks x 2 samples x 2 steps
  ~/.venv-unsloth/bin/python rl/scripts/train_grpo_unsloth.py              # defaults below
Outputs (rl/results/train/): metrics.jsonl (per step), rollouts.jsonl (per episode), eval.jsonl (ITBench-Lite),
samples/step_N.jsonl (two full transcripts per step), lora/step_N (adapter checkpoints).
"""
from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")   # must be set before `import unsloth`

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL))

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj"]          # last three: the 48 GatedDeltaNet layers


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--revision", default="c3a462cc3d38cf351f3b26cbd763469eeb74ef81")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-targets", default=",".join(LORA_TARGETS))
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--tasks-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20, help="tool-step budget per episode (eval protocol)")
    ap.add_argument("--max-turn-tokens", type=int, default=4096)
    ap.add_argument("--tis-cap", type=float, default=2.0)
    ap.add_argument("--logp-chunk", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=10, help="ITBench-Lite eval every N steps (0 = off)")
    ap.add_argument("--eval-samples", type=int, default=1)
    ap.add_argument("--eval-at-start", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--tool-workers", type=int, default=16)
    ap.add_argument("--itbench-root", default=str(Path.home() / "itbench"))
    ap.add_argument("--out", default=str(RL / "results" / "train"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.tasks_per_step, a.group, a.ckpt_every, a.eval_at_start, a.eval_every = 2, 2, 2, 0, 0, 0
        a.out = str(Path(a.out).parent / "train_smoke")
    return a


# ---------------------------------------------------------------- tool-call parsing (qwen3_coder XML format)
TC_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
PR_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def _strip1(v: str) -> str:
    """vLLM's qwen3_coder parser drops one leading and one trailing newline of a parameter value."""
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def parse_tool_calls(text: str):
    """-> (content, [(name, params|None)]). content = text before the first complete tool call (all text if none)."""
    calls = []
    for m in TC_RE.finditer(text):
        f = FN_RE.search(m.group(1))
        if not f:
            calls.append(("<unparseable>", None))
            continue
        calls.append((f.group(1).strip(), {k.strip(): _strip1(v) for k, v in PR_RE.findall(f.group(2))}))
    i = text.find("<tool_call>") if calls else -1
    return (text[:i] if i >= 0 else text).strip(), calls


# ---------------------------------------------------------------- episodes
@dataclass
class Ep:
    task: object
    sample: int
    group: int
    ids: list = field(default_factory=list)      # full token sequence (prompt + model turns + tool results)
    mask: list = field(default_factory=list)     # 1 = sampled by the policy (trained on), 0 = prompt/tool tokens
    vlp: list = field(default_factory=list)      # vLLM log-prob of each sampled token (0 elsewhere)
    n_prompt: int = 0
    steps: int = 0
    turns: int = 0
    done: bool = False
    final: str | None = None
    err: str | None = None
    flags: list = field(default_factory=list)
    sb: object = None
    rw: dict | None = None
    adv: float = 0.0


def main():
    a = parse_args()
    from unsloth import FastModel                 # before transformers / peft
    import torch
    from vllm import SamplingParams

    from finops_gym import tasks as T
    from finops_gym.generator import materialize
    from finops_gym.prompts import RUN_PYTHON_TOOL, SYSTEM, task_prompt
    from finops_gym.reward import RewardConfig, compute_reward
    from finops_gym.sandbox import Sandbox, SandboxConfig, audit_code

    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "lora").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def gpu_used_gb():
        free, total = torch.cuda.mem_get_info()
        return round((total - free) / 2**30, 1)

    # ------------------------------------------------------------ model
    t0 = time.time()
    model, proc = FastModel.from_pretrained(
        model_name=a.model, revision=a.revision, max_seq_length=a.max_seq_len,
        # The checkpoint is already block-FP8: load it as-is. load_in_fp8=True would add vLLM ONLINE fp8 quantization
        # on top, which vLLM rejects ("fp8 does not match fp8_per_channel").
        load_in_4bit=False, load_in_fp8=False, fast_inference=True, gpu_memory_utilization=a.gpu_mem,
        max_lora_rank=a.lora_rank, max_num_seqs=a.max_num_seqs, text_only=True, random_state=a.seed)
    tok = getattr(proc, "tokenizer", proc)
    model = FastModel.get_peft_model(
        model, r=a.lora_rank, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        target_modules=a.lora_targets.split(","), finetune_vision_layers=False, finetune_language_layers=True,
        use_gradient_checkpointing="unsloth", random_state=a.seed)
    llm = model.vllm_engine
    sleep_mode = bool(getattr(llm.llm_engine.vllm_config.model_config, "enable_sleep_mode", False))
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"loaded in {time.time() - t0:.0f}s | trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params "
        f"({trainable[0].dtype}) | sleep_mode={sleep_mode} | GPU used {gpu_used_gb()} GB")
    if trainable[0].dtype != torch.float32:      # bf16 master weights lose small AdamW updates
        for p in trainable:
            p.data = p.data.float()
        log("LoRA master weights upcast to float32")

    backbone = next(m for _, m in model.named_modules()
                    if hasattr(m, "embed_tokens") and hasattr(m, "norm")
                    and isinstance(getattr(m, "layers", None), torch.nn.ModuleList) and len(m.layers) > 8)
    lm_head_w = model.get_output_embeddings().weight
    log(f"backbone {type(backbone).__name__} ({len(backbone.layers)} layers), lm_head {tuple(lm_head_w.shape)} "
        f"{lm_head_w.dtype}")

    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    start_step = 0
    if a.resume:
        ck = sorted((out / "lora").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
        if ck:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(str(ck[-1] / "adapter_model.safetensors")))
            start_step = int(ck[-1].name.split("_")[1])
            log(f"resumed LoRA from {ck[-1]} (optimizer state not restored)")

    # ------------------------------------------------------------ chat template plumbing
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    STOP_IDS = sorted({i for i in (IM_END, tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0})

    def render(msgs, gen_prompt):
        return tok.apply_chat_template(msgs, tools=[RUN_PYTHON_TOOL], tokenize=False,
                                       add_generation_prompt=gen_prompt, enable_thinking=False)

    def prompt_ids(task):
        return tok(render([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": task_prompt(task.anomaly)}], True),
                   add_special_tokens=False).input_ids

    def tool_delta(outputs):
        """Template tokens that follow an assistant turn ending in <|im_end|>: the tool results + next turn header."""
        base = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"type": "function", "function": {"name": "run_python", "arguments": {"code": "x"}}}
                                for _ in outputs]}]
        a_ = render(base, False)
        b_ = render(base + [{"role": "tool", "content": o} for o in outputs], True)
        if not b_.startswith(a_):
            raise RuntimeError("chat template re-renders earlier turns; token deltas would be wrong")
        end = a_.rindex("<|im_end|>") + len("<|im_end|>")
        return tok(a_[end:] + b_[len(a_):], add_special_tokens=False).input_ids

    d = tok.decode(tool_delta(["OUT1", "OUT2"]))
    assert "OUT1" in d and "OUT2" in d and d.rstrip().endswith(("assistant", "</think>")), d
    c_, k_ = parse_tool_calls("look\n<tool_call>\n<function=run_python>\n<parameter=code>\nprint(1)\n</parameter>\n"
                              "</function>\n</tool_call>")
    assert c_ == "look" and k_ == [("run_python", {"code": "print(1)"})], (c_, k_)
    log(f"template ok | stop ids {STOP_IDS} | tool delta sample: {d[:120]!r}")

    # ------------------------------------------------------------ data
    train = T.load(T.manifest_path("train"), for_training=True)
    random.Random(a.seed).shuffle(train)
    itb = T.load(T.manifest_path("itbench_lite"))
    for t in itb:                                      # manifest paths point at the machine that built it
        if t.source_dir and not Path(t.source_dir).exists():
            t.source_dir = str(Path(a.itbench_root) / Path(t.source_dir).name)
    log(f"train tasks {len(train)} | itbench_lite {len(itb)}")

    pool = ThreadPoolExecutor(a.tool_workers)
    rcfg = RewardConfig()

    def run_tools(job):
        e, calls = job
        outs = []
        for name, params in calls:
            code = params.get("code") if isinstance(params, dict) else None
            if name != "run_python" or code is None:
                outs.append(f"[error: unknown tool or unparseable arguments: {name}]")
            else:
                e.flags += audit_code(code)
                outs.append(e.sb.run_python(code).output)
        return outs

    def rollout(eps, sp_for, lora_req):
        for e in eps:
            e.sb = Sandbox(SandboxConfig())
            materialize(e.task, e.sb.workdir)
            e.ids = prompt_ids(e.task)
            e.n_prompt = len(e.ids)
            e.mask, e.vlp = [0] * e.n_prompt, [0.0] * e.n_prompt
        if sleep_mode:
            torch.cuda.empty_cache()                   # hand the trainer's cached blocks back before vLLM re-maps KV
            llm.wake_up()
        try:
            for turn in range(a.max_steps + 1):
                live, reqs, sps = [], [], []
                for e in eps:
                    if e.done:
                        continue
                    room = a.max_seq_len - len(e.ids)
                    if room < 256:
                        e.done, e.err = True, "context"
                        continue
                    live.append(e)
                    reqs.append({"prompt_token_ids": list(e.ids)})
                    sps.append(sp_for(min(a.max_turn_tokens, room), e, turn))
                if not live:
                    break
                outs = llm.generate(reqs, sps, lora_request=lora_req, use_tqdm=False)
                jobs = []
                for e, o in zip(live, outs):
                    c = o.outputs[0]
                    gen = list(c.token_ids)
                    lps = [c.logprobs[i][t].logprob for i, t in enumerate(gen)] if c.logprobs else [0.0] * len(gen)
                    e.ids += gen
                    e.mask += [1] * len(gen)
                    e.vlp += lps
                    e.turns += 1
                    stopped = bool(gen) and gen[-1] in STOP_IDS
                    content, calls = parse_tool_calls(tok.decode(gen[:-1] if stopped else gen,
                                                                 skip_special_tokens=False))
                    if not calls:
                        e.final, e.done = content, True
                        continue
                    if turn == a.max_steps:            # budget exhausted while still calling tools
                        e.err, e.done = "max_steps", True
                        continue
                    if not stopped:                    # complete call(s) but the turn ran to the length cap
                        e.ids.append(IM_END)
                        e.mask.append(0)
                        e.vlp.append(0.0)
                    e.steps += 1
                    jobs.append((e, calls))
                for (e, _), outs_ in zip(jobs, pool.map(run_tools, jobs)):
                    dlt = tool_delta(outs_)
                    e.ids += dlt
                    e.mask += [0] * len(dlt)
                    e.vlp += [0.0] * len(dlt)
        finally:
            if sleep_mode:
                llm.sleep(1)
            for e in eps:
                e.rw = compute_reward(e.final or "", e.task.label, steps_used=e.steps, max_steps=a.max_steps, cfg=rcfg)
                if e.sb is not None:
                    e.sb.close()
                    e.sb = None
        return eps

    def train_sp(max_tokens, e, turn):
        return SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_tokens, logprobs=0,
                              stop_token_ids=STOP_IDS, skip_special_tokens=False)

    def eval_sp(max_tokens, e, turn):                  # model-card sampling, as in the baseline eval
        return SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5,
                              repetition_penalty=1.0, max_tokens=max_tokens, stop_token_ids=STOP_IDS,
                              skip_special_tokens=False, seed=(e.sample + 1) * 1000 + turn)

    def ep_record(e, step, kind):
        r = e.rw
        return dict(kind=kind, step=step, task_id=e.task.task_id, sample=e.sample, regime=e.task.regime,
                    mechanism=e.task.mechanism, reward=r["reward"], exact=r["exact"], f1=round(r["f1"], 3),
                    format_ok=r["format_ok"], steps=e.steps, turns=e.turns, err=e.err, adv=round(e.adv, 4),
                    gen_tokens=sum(e.mask), seq_tokens=len(e.ids), audit=sorted(set(e.flags)))

    def append(name, rows):
        with open(out / name, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def chunked_logp(h, tgt):
        """log p(tgt) for hidden states h [n, d] without materialising [n, vocab] logits for all n at once."""
        def blk(hc, tc):
            lg = (hc @ lm_head_w.t()).float()
            return lg.gather(-1, tc[:, None]).squeeze(-1) - torch.logsumexp(lg, -1)
        from torch.utils.checkpoint import checkpoint
        return torch.cat([checkpoint(blk, h[i:i + a.logp_chunk], tgt[i:i + a.logp_chunk], use_reentrant=False)
                          for i in range(0, h.shape[0], a.logp_chunk)])

    def train_on(eps):
        batch = [e for e in eps if abs(e.adv) > 1e-8]
        n_tok = sum(sum(e.mask) for e in batch)
        if not batch:
            return dict(loss=0.0, grad_norm=0.0, mismatch=None, tis_clipped=None, trained_eps=0, trained_tokens=0)
        model.train()
        loss_sum, mm_sum, clip_n = 0.0, 0.0, 0
        for e in batch:
            ids = torch.tensor([e.ids], device="cuda")
            m = torch.tensor(e.mask[1:], device="cuda", dtype=torch.bool)
            vlp = torch.tensor(e.vlp[1:], device="cuda", dtype=torch.float32)[m]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = backbone(input_ids=ids, use_cache=False).last_hidden_state[0, :-1][m]
                logp = chunked_logp(h, ids[0, 1:][m])
            with torch.no_grad():
                diff = logp.detach() - vlp
                w = torch.exp(diff).clamp(max=a.tis_cap)
                mm_sum += diff.abs().sum().item()
                clip_n += (torch.exp(diff) > a.tis_cap).sum().item()
            loss = -(e.adv * w * logp).sum() / n_tok
            loss.backward()
            loss_sum += loss.item()
            del ids, h, logp, loss
        gn = torch.nn.utils.clip_grad_norm_(trainable, a.max_grad_norm).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return dict(loss=loss_sum, grad_norm=gn, mismatch=mm_sum / n_tok, tis_clipped=clip_n / n_tok,
                    trained_eps=len(batch), trained_tokens=n_tok)

    def lora_diagnostics():
        """What vLLM wraps with LoRA vs what the trainer's adapter keys resolve to (printed when hand-off fails)."""
        from unsloth_zoo import vllm_utils as vu
        runner = vu._get_vllm_model_runner(model)
        vm = vu._get_vllm_lora_model(model, runner)
        targets = vu._vllm_lora_target_names(vm, vu._get_vllm_lora_manager(model, runner),
                                             vu._vllm_worker_lora_config(runner, vm))
        mods = sorted(targets[0]) if targets else []
        mapper = getattr(vm, "hf_to_vllm_mapper", None)
        log(f"vLLM model {type(vm).__name__}: {len(mods)} LoRA-wrapped modules; layer-0/3 examples: "
            f"{[m for m in mods if '.layers.0.' in m or '.layers.3.' in m]}")
        log(f"vLLM packed_modules_mapping: {getattr(vm, 'packed_modules_mapping', None)}")
        log(f"vLLM hf_to_vllm_mapper: {mapper}")
        keys = [k for k in vu._lora_state_dict(model) if ".layers.0." in k or ".layers.3." in k]
        for k in keys[:12]:
            log(f"  train key {k} -> {vu._resolve_lora_key_to_module(k, mapper)}")

    def lora_request(step):
        """The live LoRA as an in-memory vLLM request. The trainer is the text-only model (keys model.layers.N...),
        vLLM serves the VLM class whose mapper expects checkpoint keys model.language_model.layers.N... -> rename,
        then run Unsloth's own servability check so a silently ignored adapter is impossible."""
        from unsloth_zoo import vllm_utils as vu
        from vllm.lora.request import LoRARequest
        path = str(out / "lora_live")
        try:
            peft_config = vu._live_lora_config(model, path)
            sd = {k.replace(".default", "").replace("base_model.model.model.layers.",
                                                    "base_model.model.model.language_model.layers.", 1): v
                  for k, v in vu._lora_state_dict(model).items()}
            vu._check_lora_is_servable(model, list(sd), "the training model (VLM key layout)", peft_config)
        except RuntimeError:
            lora_diagnostics()
            raise
        return LoRARequest(str(step + 1), step + 1, lora_tensors=sd, lora_config=peft_config)

    def evaluate(step, lora_req):
        t = time.time()
        eps = rollout([Ep(task, s, gi) for gi, task in enumerate(itb) for s in range(a.eval_samples)], eval_sp, lora_req)
        recs = [ep_record(e, step, "eval_itbench") for e in eps]
        append("eval.jsonl", recs)
        n = len(recs)
        res = dict(step=step, itbench_pass=sum(r["exact"] for r in recs) / n,
                   itbench_reward=sum(r["reward"] for r in recs) / n,
                   itbench_answered=sum(r["format_ok"] for r in recs) / n, eval_s=round(time.time() - t, 1))
        log(f"EVAL step {step}: {json.dumps(res)}")
        return res

    # ------------------------------------------------------------ loop
    lr_at = lambda s: a.lr * min(1.0, (s + 1) / max(1, a.warmup))
    if a.eval_at_start and a.eval_every and start_step == 0:
        evaluate(0, lora_request(0))
    for step in range(start_step, a.steps):
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        k = a.tasks_per_step
        batch_tasks = [train[(step * k + i) % len(train)] for i in range(k)]
        eps = [Ep(task, s, gi) for gi, task in enumerate(batch_tasks) for s in range(a.group)]
        lreq = lora_request(step)
        rollout(eps, train_sp, lreq)
        t_roll = time.time() - ts
        gpu_after_roll = gpu_used_gb()
        informative = 0
        for gi in range(k):
            grp = [e for e in eps if e.group == gi]
            mean = sum(e.rw["reward"] for e in grp) / len(grp)
            for e in grp:
                e.adv = e.rw["reward"] - mean
            informative += any(abs(e.adv) > 1e-8 for e in grp)
        if a.smoke and not informative:                # exercise backward + memory even if all rewards tie
            for e in eps:
                e.adv = 0.1 if e.sample == 0 else -0.1
        tt = time.time()
        torch.cuda.reset_peak_memory_stats()
        tr = train_on(eps)
        t_train = time.time() - tt
        recs = [ep_record(e, step, "train") for e in eps]
        append("rollouts.jsonl", recs)
        n = len(recs)
        met = dict(step=step, lr=lr_at(step),
                   reward=sum(r["reward"] for r in recs) / n, pass_rate=sum(r["exact"] for r in recs) / n,
                   answered=sum(r["format_ok"] for r in recs) / n,
                   hit_budget=sum(r["err"] == "max_steps" for r in recs) / n,
                   hit_context=sum(r["err"] == "context" for r in recs) / n,
                   mean_steps=sum(r["steps"] for r in recs) / n,
                   mean_gen_tokens=sum(r["gen_tokens"] for r in recs) / n,
                   mean_seq_tokens=sum(r["seq_tokens"] for r in recs) / n,
                   max_seq_tokens=max(r["seq_tokens"] for r in recs),
                   informative_groups=informative, groups=k, audit_flagged=sum(bool(r["audit"]) for r in recs),
                   **tr, t_rollout=round(t_roll, 1), t_train=round(t_train, 1), t_step=round(time.time() - ts, 1),
                   gpu_gb_after_rollout=gpu_after_roll, train_peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                   time=time.time())
        append("metrics.jsonl", [met])
        log(f"STEP {step}: " + json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in met.items()
                                            if k_ != "time"}))
        with open(out / "samples" / f"step_{step}.jsonl", "w") as f:
            for e in sorted(eps, key=lambda e: -e.rw["reward"])[:1] + sorted(eps, key=lambda e: e.rw["reward"])[:1]:
                f.write(json.dumps(dict(ep_record(e, step, "train"), label=e.task.label,
                                        transcript=tok.decode(e.ids[e.n_prompt:]))) + "\n")
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0:
            model.save_lora(str(out / "lora" / f"step_{step + 1}"))
        if a.eval_every and (step + 1) % a.eval_every == 0:
            evaluate(step + 1, lora_request(step + 1))
    if not a.smoke:
        model.save_lora(str(out / "lora" / "final"))
        if a.eval_every and a.steps % a.eval_every:
            evaluate(a.steps, lora_request(a.steps))
    log("done")


if __name__ == "__main__":
    main()
