"""Convert a PEFT LoRA adapter (train_grpo_unsloth.py) to an mlx-lm adapter for the local qwen3_5 MLX model.

The adapter stays UNMERGED (base 8-bit + LoRA on top), like training (FP8 base + LoRA). Merging into bf16 and
re-quantizing to 8-bit would round away most of the update (|dW|/|W| ~ 3e-4, below the 8-bit step).

mlx-lm LoRALinear: y = base(x) + scale * (x @ lora_a) @ lora_b,  lora_a: (in, r), lora_b: (r, out)
PEFT:              y = base(x) + (alpha/r) * B @ A @ x,           A: (r, in),     B: (out, r)
=> lora_a = A.T, lora_b = B.T, scale = alpha / r. mlx qwen3_5 sanitize() does not permute these linear weights.

    python rl/scripts/peft_to_mlx_adapter.py rl/results/aws_train/lora/final models/adapters/rl-step32
"""
import json
import re
import sys
from pathlib import Path

import mlx.core as mx


def main(src: str, dst: str):
    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "adapter_config.json").read_text())
    assert not cfg.get("use_rslora") and not cfg.get("use_dora")
    w = mx.load(str(src / "adapter_model.safetensors"))
    out, mods, layers = {}, set(), set()
    for k, v in w.items():
        m = re.fullmatch(r"base_model\.model\.model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight", k)
        assert m, k
        n, mod, ab = m.groups()
        layers.add(int(n))
        mods.add(mod)
        out[f"language_model.model.layers.{n}.{mod}.lora_{ab.lower()}"] = v.T
    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst / "adapters.safetensors"), out)
    (dst / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": max(layers) + 1,
        "lora_parameters": {"rank": cfg["r"], "scale": cfg["lora_alpha"] / cfg["r"], "dropout": 0.0,
                            "keys": sorted(mods)},
        "source": str(src.resolve()),
    }, indent=2))
    print(f"{len(out)} tensors, {len(layers)} layers, modules {sorted(mods)} -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
