import time, json, sys
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
m, tok = load(sys.argv[1]); out = []
print("wired limit / recommended:", mx.device_info().get("max_recommended_working_set_size"), flush=True)
for run, pp in [(1, 1.5), (2, 0.0), (3, 1.5)]:
    p = tok.apply_chat_template([{"role": "user", "content": "Write a detailed 600-word essay on cloud cost optimization."}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    t0 = time.time(); last = t0; gaps = []
    for i, r in enumerate(stream_generate(m, tok, p, max_tokens=600, sampler=make_sampler(temp=0.7, top_p=0.8, top_k=20), logits_processors=make_logits_processors(presence_penalty=pp))):
        now = time.time(); gaps.append(now - last); last = now
        if (i + 1) % 100 == 0: print(run, pp, i + 1, round(now - t0, 1), "max gap so far", round(max(gaps), 2), flush=True)
    out.append(dict(run=run, pp=pp, n=i + 1, total_s=round(time.time() - t0, 1), max_gap_s=round(max(gaps), 2), tps=round(r.generation_tps, 2), peak_gb=round(r.peak_memory, 2)))
    print(out[-1], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
