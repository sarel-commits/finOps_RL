#!/usr/bin/env python
"""Build task manifests (seed + label + data hash per task; data.csv is regenerated at rollout time).

  python rl/scripts/build_dataset.py --train 2000 --val 200 --test-synth 300
Writes rl/data/{train,val,test_synth,itbench_lite}.jsonl and rl/data/stats.json.
Splits use disjoint seed ranges; itbench_lite is file-backed and eval-only.
"""
import argparse, collections, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finops_gym.generator import GEN_VERSION, GenConfig, generate, task_from_dir  # noqa: E402
from finops_gym import tasks as T  # noqa: E402

ITB = Path(__file__).resolve().parents[2] / "data/itbench-lite/snapshots/finops/v0.1-finops-anomaly"


def _gen(args):
    seed, split = args
    t = generate(seed, GenConfig())
    t.split = split
    return t


def stats(ts):
    c = collections.Counter
    st = dict(n=len(ts), regime=c(t.regime for t in ts), mechanism=c(t.mechanism for t in ts),
              n_root_causes=c(len(t.label) for t in ts), label_type=c(t.label[0]["type"] for t in ts),
              rows_mean=round(sum(t.n_rows for t in ts) / len(ts)),
              attempts_mean=round(sum(t.meta.get("attempt", 0) for t in ts) / len(ts), 2))
    for reg in ("clean", "noisy"):
        sub = [t for t in ts if t.regime == reg]
        if sub:
            st[f"naive_top_delta_exact_{reg}"] = round(sum(t.meta["naive_top_delta_exact"] for t in sub) / len(sub), 3)
            st[f"oracle_exact_{reg}"] = round(sum(t.meta["oracle_exact"] for t in sub) / len(sub), 3)
    return {k: dict(v) if isinstance(v, collections.Counter) else v for k, v in st.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test-synth", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = {"generator_version": GEN_VERSION, "gen_config": GenConfig().__dict__, "splits": {}}
    for split, n in (("train", a.train), ("val", a.val), ("test_synth", a.test_synth)):
        t0 = time.time()
        base = T.SPLIT_SEED_BASE[split]
        with ProcessPoolExecutor(a.workers) as ex:
            ts = list(ex.map(_gen, [(base + i, split) for i in range(n)], chunksize=16))
        T.save(ts, T.manifest_path(split))
        out["splits"][split] = dict(stats(ts), seconds=round(time.time() - t0, 1))
        print(split, json.dumps(out["splits"][split]))
    itb = [task_from_dir(d) for d in sorted(ITB.iterdir(), key=lambda p: int(p.name.split("-")[1]))]
    T.save(itb, T.manifest_path("itbench_lite"))
    out["splits"]["itbench_lite"] = dict(n=len(itb), eval_only=True)
    # sanity: no seed overlap, no eval tasks in train
    seeds = {s: {t.seed for t in T.load(T.manifest_path(s))} for s in ("train", "val", "test_synth")}
    assert not (seeds["train"] & seeds["val"]) and not (seeds["train"] & seeds["test_synth"])
    T.load(T.manifest_path("train"), for_training=True)
    (T.DATA / "stats.json").write_text(json.dumps(out, indent=1, default=str))
    print("wrote", T.DATA)


if __name__ == "__main__":
    main()
