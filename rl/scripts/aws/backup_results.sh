#!/bin/bash
# Snapshot the synced training outputs (rl/results/<src>) into rl/results/backups/<run>/, which the 5-minute sync
# never writes to. Small files: timestamped copies. LoRA checkpoints and samples: one accumulating copy.
#   bash rl/scripts/aws/backup_results.sh run2              # src defaults to aws_train
#   bash rl/scripts/aws/backup_results.sh run1_lockstep aws_train_run1_lockstep
set -u
cd ~/finops-eval/rl/results || exit 1
RUN=${1:-run2}
SRC=${2:-aws_train}
DST=backups/$RUN
[ -d "$SRC" ] || { echo "backup: no $SRC yet"; exit 0; }
ts=$(date -u +%Y%m%dT%H%MZ)
mkdir -p "$DST/snapshots/$ts"
for f in metrics.jsonl eval.jsonl rollouts.jsonl config.json train.log; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/snapshots/$ts/"
done
[ -d "$SRC/samples" ] && rsync -a "$SRC/samples/" "$DST/samples/"
[ -d "$SRC/lora" ] && rsync -a "$SRC/lora/" "$DST/lora/"
n_ck=$(ls "$DST/lora" 2>/dev/null | wc -l | tr -d ' ')
echo "backup $RUN @ $ts: $(ls "$DST/snapshots/$ts" | tr '\n' ' ')| lora checkpoints: $n_ck | total $(du -sh "$DST" | cut -f1)"
