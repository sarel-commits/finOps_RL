#!/bin/bash
# Resumable download loop: hf download resumes partial files automatically.
source ~/finops-eval/.venv/bin/activate
for i in 1 2 3 4 5 6 7 8; do
  echo "=== attempt $i $(date) ==="
  hf download Qwen/Qwen3.8-27B --local-dir ~/finops-eval/models/qwen3.8-27b-bf16 && { echo "DOWNLOAD_OK $(date)"; exit 0; }
  sleep 15
done
echo "DOWNLOAD_FAILED"; exit 1
