"""Task manifests: load/save, split hygiene."""
from __future__ import annotations

import json
from pathlib import Path

from .generator import Task

RL_ROOT = Path(__file__).resolve().parents[1]
DATA = RL_ROOT / "data"
SPLIT_SEED_BASE = {"train": 1_000_000, "val": 2_000_000, "test_synth": 3_000_000}
EVAL_ONLY = {"itbench_lite"}            # never allowed in a training manifest


def manifest_path(split: str, name: str | None = None) -> Path:
    return DATA / f"{name or split}.jsonl"


def save(tasks: list[Task], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(t.to_json() + "\n" for t in tasks))
    tmp.replace(path)


def load(path: Path, for_training: bool = False) -> list[Task]:
    tasks = [Task(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if for_training:
        bad = [t.task_id for t in tasks if t.split in EVAL_ONLY or t.source_dir]
        if bad:
            raise ValueError(f"{path}: eval-only tasks in a training manifest (contamination): {bad[:5]}")
    return tasks
