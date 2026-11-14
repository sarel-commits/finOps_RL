"""Merge a PEFT LoRA adapter (from train_grpo_unsloth.py) into the bf16 HF checkpoint, shard by shard.

    W_merged = W + (lora_alpha / r) * B @ A        (computed in float32, stored as bf16)

Output is a full HF checkpoint (same shards, index and config) that mlx_lm.convert can quantize exactly like the
baseline (affine 8-bit, group size 64), so eval differences come from the adapter, not from a different pipeline.

    python rl/scripts/merge_lora.py --base models/qwen3.8-27b-bf16 \
        --adapter rl/results/aws_train/lora/final --out models/qwen3.8-27b-rl-step32-bf16
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, adp, out = Path(a.base), Path(a.adapter), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)

    cfg = json.loads((adp / "adapter_config.json").read_text())
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "only plain LoRA is supported"
    scale = cfg["lora_alpha"] / cfg["r"]

    # PEFT key  base_model.model.model.layers.N.<mod>.lora_{A,B}.weight  ->  base  model.language_model.layers.N.<mod>.weight
    pairs = {}
    with safe_open(str(adp / "adapter_model.safetensors"), "np") as f:
        for k in f.keys():
            m = re.fullmatch(r"base_model\.model\.model\.(layers\.\d+\..+)\.lora_([AB])\.weight", k)
            assert m, f"unexpected adapter key {k}"
            pairs.setdefault(f"model.language_model.{m.group(1)}.weight", {})[m.group(2)] = k
    lora = mx.load(str(adp / "adapter_model.safetensors"))

    idx = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = [k for k in pairs if k not in idx]
    assert not missing, f"adapter targets not in base: {missing[:5]}"

    merged, rel = 0, []
    for shard in sorted(set(idx.values())):
        w = mx.load(str(base / shard))
        for k in [k for k in w if k in pairs]:
            A = lora[pairs[k]["A"]].astype(mx.float32)
            B = lora[pairs[k]["B"]].astype(mx.float32)
            W0 = w[k].astype(mx.float32)
            assert (B.shape[0], A.shape[1]) == tuple(W0.shape), (k, W0.shape, A.shape, B.shape)
            d = scale * (B @ A)
            w[k] = (W0 + d).astype(mx.bfloat16)
            rel.append((mx.linalg.norm(d) / mx.linalg.norm(W0)).item())
            merged += 1
        mx.save_safetensors(str(out / shard), w, metadata={"format": "pt"})
        del w
        mx.clear_cache()
        print(f"{shard}: done ({merged}/{len(pairs)} merged)", flush=True)

    assert merged == len(pairs), f"merged {merged} of {len(pairs)}"
    for p in base.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors"):
            shutil.copy2(p, out / p.name)
    rel.sort()
    print(f"merged {merged} matrices, scale {scale}; relative change |dW|/|W|: "
          f"median {rel[len(rel) // 2]:.2e}, max {rel[-1]:.2e}")


if __name__ == "__main__":
    main()
