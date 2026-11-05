#!/usr/bin/env python
"""Phase 2 sanity check + performance probe (in-process mlx_lm, no server).

  python scripts/sanity_perf.py <model_path> <out_json>

Runs 3 sanity prompts (arithmetic, FinOps definition, strict JSON) in non-thinking
mode with model-card sampling, plus prefill-length probes. Records prompt tok/s,
generation tok/s and peak memory reported by mlx_lm.
"""
import json, sys, time, re
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

model_path, out = sys.argv[1], sys.argv[2]
t0 = time.time()
model, tok = load(model_path)
load_s = time.time() - t0
mx.random.seed(0)
sampler = make_sampler(temp=0.7, top_p=0.8, top_k=20, min_p=0.0)
lp = make_logits_processors(presence_penalty=1.5) if "presence_penalty" in make_logits_processors.__code__.co_varnames else None


def chat(user, max_tokens=512, enable_thinking=False):
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                     enable_thinking=enable_thinking, tokenize=False)
    text, last = "", None
    t = time.time()
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=sampler, logits_processors=lp):
        text += r.text
        last = r
    return dict(prompt=user, output=text, prompt_tokens=last.prompt_tokens, prompt_tps=round(last.prompt_tps, 2),
                generation_tokens=last.generation_tokens, generation_tps=round(last.generation_tps, 2),
                peak_memory_gb=round(last.peak_memory, 3), wall_s=round(time.time() - t, 2),
                finish_reason=last.finish_reason)


res = dict(model=model_path, load_s=round(load_s, 1), sanity=[], prefill_probe=[])
s = chat("What is 1234 * 56 - 789? Reply with just the number.", 64)
s["expected"] = 1234 * 56 - 789
s["pass"] = str(s["expected"]) in re.sub(r"[,\s]", "", s["output"])
res["sanity"].append(s)
s = chat("In two or three sentences: what is a reserved instance in cloud FinOps?", 256)
s["pass"] = bool(re.search(r"(?i)discount|commit|term|1.year|3.year|reserv", s["output"]))
res["sanity"].append(s)
s = chat('Return ONLY a JSON object (no markdown, no prose) with keys "service" (string "EC2"), '
         '"monthly_cost" (number 1234.5) and "tags" (array with "prod" and "web").', 128)
try:
    j = json.loads(s["output"].strip())
    s["pass"] = j == {"service": "EC2", "monthly_cost": 1234.5, "tags": ["prod", "web"]}
except Exception as e:
    s["pass"], s["json_error"] = False, str(e)
res["sanity"].append(s)
# prefill probes with a repeated CSV-like block
row = "2025-10-13,49tAGZz,GCP,Sj2f9,9fC,o4d6MEfKqF,0.1577,10,37.848\n"
for n_rows in [100, 1000, 4000]:
    p = chat("Here is some data:\n" + row * n_rows + "\nHow many rows are above? Answer with a number only.", 16)
    res["prefill_probe"].append({k: p[k] for k in ["prompt_tokens", "prompt_tps", "generation_tokens",
                                                   "generation_tps", "peak_memory_gb", "wall_s"]})
res["peak_memory_gb_overall"] = round(mx.get_peak_memory() / 1e9, 3)
json.dump(res, open(out, "w"), indent=1)
for s in res["sanity"]:
    print(f"PASS={s['pass']} | {s['prompt'][:50]!r} -> {s['output'][:150]!r} | prefill {s['prompt_tps']} tok/s, "
          f"decode {s['generation_tps']} tok/s, peak {s['peak_memory_gb']} GB")
for p in res["prefill_probe"]:
    print("prefill probe", p)
print("load_s", res["load_s"], "peak overall GB", res["peak_memory_gb_overall"])
