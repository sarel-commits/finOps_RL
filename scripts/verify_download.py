#!/usr/bin/env python
"""Verify BF16 download: 18 shards present; crc32.txt entries match; every LFS file
matches the Hub's size + SHA-256 (crc32.txt does not list the shards)."""
import json, zlib, hashlib, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
M = Path.home() / "finops-eval/models/qwen3.8-27b-bf16"
exp = json.load(open(Path.home() / "finops-eval/logs/bf16_expected_files.json"))
ok = True
shards = sorted(M.glob("model-*-of-00018.safetensors"))
print(f"shards present: {len(shards)}/18"); ok &= len(shards) == 18
for line in (M / "crc32.txt").read_text().split("\n"):
    if not line.strip(): continue
    want, name = line.split()
    got = format(zlib.crc32((M / name).read_bytes()) & 0xFFFFFFFF, "08x")
    print(f"crc32 {name}: {'OK' if got == want else 'MISMATCH ' + got}"); ok &= got == want
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
lfs = {k: v for k, v in exp.items() if v["sha256"]}
with ThreadPoolExecutor(4) as ex:
    res = dict(zip(lfs, ex.map(lambda k: sha(M / k) if (M / k).exists() else None, lfs)))
for k, v in lfs.items():
    p = M / k
    good = p.exists() and p.stat().st_size == v["size"] and res[k] == v["sha256"]
    print(f"sha256 {k}: {'OK' if good else 'BAD'}"); ok &= good
print("ALL_OK" if ok else "VERIFY_FAILED"); sys.exit(0 if ok else 1)
