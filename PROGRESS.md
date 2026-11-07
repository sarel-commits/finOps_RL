# PROGRESS

## Phase 0: Preflight — DONE (2026-11-06)
- Chip: Apple M5 Max
- Unified memory: 68719476736 bytes (64 GiB) -> >=48GB, so 8-bit quantization (not the 4-bit fallback)
- macOS 26.5.1 (25F80)
- Free disk: 1.7 TiB on /System/Volumes/Data, enough for BF16 + 8-bit
- Docker: NOT installed
- uv 0.13.7 installed user-local at ~/finops-eval/.tools/bin (no shell profile changes)
- venv: ~/finops-eval/.venv (CPython 3.12.16); versions pinned in requirements.lock
  - mlx 0.32.4, mlx-lm 0.32.2, mlx-vlm 0.7.12, transformers 5.22.1, huggingface-hub 2.4.0, datasets 5.1.0

## Phase 1: Download — DONE + model card notes
- Finished 01:57 (DOWNLOAD_OK); 55 GB on disk. Verification (logs/verify_download.log): 18/18 shards present; all LFS files match Hub SHA-256 + size; crc32.txt: 5/8 match, 3 mismatches traced to a stale upstream crc32.txt (local files byte-identical to Hub blobs, see DECISIONS #14).
- Started 00:11 via scripts/download.sh (resumable loop, log logs/download.log). Link ~34 Mbps; ~6.8 MB/s effective.
- Model card / config facts (from config.json, generation_config.json, chat_template.jinja, README.md):
  - Architecture: `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`, pipeline `image-text-to-text`; 27.78B params BF16; text decoder 64 layers (hybrid: 3x linear_attention : 1x full_attention), hidden 5120, 24 heads / 4 KV heads, vocab 248,320.
  - Max context: 262,144 native (max_position_embeddings); extensible to ~1M via YaRN (factor 4).
  - Sampling: thinking: temp 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, rep 1.0. Non-thinking/instruct: temp 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5, rep 1.0. (generation_config.json defaults: temp 1.0, top_p 0.95, top_k 20.)
  - Thinking toggle: thinking is ON by default; disable per request via `chat_template_kwargs: {"enable_thinking": false}` (template then emits an empty `<think>\n\n</think>`). `reasoning_effort` in {xhigh (default), medium, low} injects a system instruction; `preserve_thinking` (default true) keeps past reasoning in context.
  - Tool-calling format: tools are listed as JSON inside `<tools></tools>` in the system prompt; the model calls tools with XML: `<tool_call>\n<function=NAME>\n<parameter=ARG>\nVALUE\n</parameter>\n</function>\n</tool_call>`; tool results go back as `<tool_response>...</tool_response>` inside a user turn. (Same as the Qwen3-Coder format; mlx-lm parser `qwen3_coder`.)

## Prep work done while downloading
- ITBench-Lite FinOps (15 scenarios, 14 MB) downloaded to data/itbench-lite. GT types: 12x single Instance Family, 1x two Instance Families (S9), 2x Application (S3: 1 app; S6: 20 apps).
- Official evaluator located: itbench-hub/ITBench-Evaluations@14f026f (LLM judge; rubric = all GT resources matched by name+type → 1/0). Implemented deterministically in scripts/score.py; oracle test = 15/15.
- Full ITBench FinOps: skipped (Kubernetes/containers needed). aws-bench: reviewed (REPORT §8), not run.
- Synthetic eval: scripts/gen_synthetic.py → 43 tasks / 7 categories, deterministic (GT sha256[:16] 6f909dfab764aed0 on two runs); oracle test 43/43.
- Harness: scripts/agent.py (native tool calls + ReAct fallback, checkpointing, retries, server auto-restart), scripts/run_bench.py, scripts/server.sh, scripts/sanity_perf.py, scripts/verify_download.py.
- Non-LLM heuristic reference on ITBench-Lite: top-instance-family-by-delta 9/15 = 60.0%; auto-dimension variant 7/15 = 46.7% (results/itbench_lite/_heuristic_baseline/).

## Phase 2: Quantize — DONE (01:58)
- `python -m mlx_lm convert --hf-path models/qwen3.8-27b-bf16 -q --q-bits 8 --q-group-size 64 --q-mode affine --mlx-path models/qwen3.8-27b-mlx-8bit` → 27 GB, 6 shards, 8.501 bits/weight, quantization {group_size 64, bits 8, mode affine}. Log: logs/convert.log.
- Sanity (results/sanity/sanity_perf_8bit.json): arithmetic FAIL (answered 67542, correct 68315), RI definition PASS, strict JSON PASS. Decode 18–22 tok/s; prefill 841 tok/s @5.5k, 597 @55k, 263 @220k tokens; peak 28.9 GB (short) → 54.8 GB @220k context.

## Phase 3: Server — DONE
- scripts/server.sh: mlx_lm.server on 127.0.0.1:8080, logs/server.log, prompt cache 10, + caffeinate tied to the server pid.
- scripts/healthcheck.py: chat OK, thinking OK (reasoning field), native tool calls OK → no ReAct fallback needed.
- Debugged ~16-min generation stalls → macOS idle system sleep (DECISIONS #15). Fixed with caffeinate; verified (400 tokens in 23 s, twice).
- Smoke test revealed prompt priming on resource type → prompt fixed (DECISIONS #16).

## Phase 4: Benchmarks — RUNNING (started ~05:00 via scripts/run_all.sh, log logs/run_all.log)

### 4.1 ITBench-Lite non-thinking — DONE (08:1x)
- 45/45 runs completed (scores: results/itbench_lite/nothink_native/summary.json).
- root_cause_resource 48.9% ± 3.8% (seeds 46.7/46.7/53.3); 22 pass / 23 fail.
- 17 failures = 20-step cap with no answer; 6 = wrong entity (wrong column/dimension). 0 harness-parsed tool calls, 0 leak flags.
- Mean 261 s/run, 114.7k prompt tokens/run (cumulative over steps), 3.5k output tokens/run.
- analyze.py run; REPORT §2.1 written. Synthetic eval started automatically (~17 s/task).

### 4.4 Synthetic — DONE (09:42)
- 129/129 runs; 93.0% ± 2.3% (120 pass). Failures: 6 step-cap, 2 wrong_computation, 1 wrong_entity.
- REPORT §2.2 written; analyze.py re-run. ITBench thinking run (think_native) started 09:42.

### Follow-up experiment (user request, 11:00)
- 11:00 Stopped the thinking chain (DECISIONS #18). Kept 5 complete thinking runs (seed 1, S1–S5); server is still up under caffeinate.
- Added `--budget-nudges` and `--prompt-version v2` (DECISIONS #19). Defaults verified to reproduce the old prompts exactly.
- Checked F1 ± 0.000: computed per seed, coincidence (DECISIONS #20).
- Smoke tests running: `_smoke_nudges_v2` (S4, S6, seed 1) and `_smoke_forced` (S3, max-steps 6).
- 11:16 Smoke tests passed (logs/smoke_nudges.log; results in results/itbench_lite/_smoke_*; excluded from analysis). S4 v2: forced final answered correctly (+1 extra family); S6 v2: chose Application dimension (baseline never did); S3 max-steps 6: warning at step 1 and forced final work.
- 11:16 Launched scripts/run_followup.sh under caffeinate (log logs/run_followup.log): A nothink_nudges → B nothink_nudges_v2 → C synthetic nothink_nudges_v2.
- 12:07 Stopped the follow-up chain (A: 10/45 runs done). User asked for the failed runs on Qwen3.8-2.4T-A95B-FP8 via Together: not servable on Together (no serverless access, no dedicated-endpoint hardware). Nothing run (DECISIONS #21–22).
- 12:14 Launched scripts/run_flash_failed.sh (Together Qwen3.8-Flash, 32 failed runs; log logs/run_flash_failed.log). DECISIONS #23.
- 12:46 Flash re-run DONE: ITBench 1/23 rescued (21 step-cap), synthetic 7/9; ~$0.74 est. Fixed score.py per-category crash on partial sets (baseline re-score unchanged, 93.0%). REPORT §2.4 written.
- 13:28 Launched GLM-5.3 and Kimi-K3.1-Pro re-runs of the 32 failed runs in parallel (logs/run_glm53_failed.log, logs/run_kimik31_failed.log). Mistral 4 and Opus 5.5 skipped (DECISIONS #24).
- 13:30 STOPPED GLM-5.3 and Kimi-K3.1-Pro re-runs on user instruction; partial trajectories kept, nothing deleted. mlx_lm.server still running.
- 13:32 RESUMED GLM-5.3 and Kimi-K3.1-Pro failed-run re-runs (checkpointed) on user instruction.
- 13:43 Launched harness-improvement experiment (DECISIONS #26): local chain (logs/run_improve_local.log) + Flash grid x3 (logs/flash_*.log). GLM/Kimi failed-run re-runs still running.
- 13:46 Flash effort probe: only top-level reasoning_effort honoured; harness fixed; launched flash_think_medium_nudges_v2_guard (DECISIONS #27).
- 14:31 GLM-5.3 (12/23 ITB, 9/9 syn, ~$5.95) and Kimi-K3.1-Pro (21/23, 9/9, ~$4.79) failed-run re-runs DONE. Flash full baseline 20.0%; Flash nudges+v2 82.2% (post-tuning). REPORT §2.3/§2.5 written.
- 14:40 Flash grid DONE: v1 20.0%, nudges+v2 82.2%, +guard 82.2%, think-medium+nudges+v2+guard 95.6% ($0.66). Local chain re-ordered: thinking next (DECISIONS #28). REPORT §2.5 updated.

## 2026-11-06 18:30
- Local 27B `nothink_nudges_v2` finished: **84.4% ± 7.7%** (38/45, Wilson 71.2–92.3%), baseline 48.9%. Pre-registered decision rule: ≥70% → passes (beats 60% heuristic). REPORT §2.5 updated.
- Chain moved to `think_medium_nudges_v2_guard` (one transient APIConnectionError at the start, retried; server healthy).
- 18:45 REPORT §4 Error analysis, §6 quantization caveat, §7 skipped list, §9 Recommendations written (analyze.py re-run; error_categories.png refreshed). §3 heuristic sentence updated for post-tuning result.
- 18:33 27B thinking-medium run: first result S1 seed 1 correct in 10 steps, 383 s. Estimated 45 runs ≈ 4.8 h.
- 00:27 27B think-medium + nudges + v2 + guard finished: 100% official (45/45, Wilson 92.1–100%), strict set match 60%, F1 0.868, 8.5 steps, 480 s/run. REPORT §2.5/§5/§9 updated. Chain now on synthetic nothink_nudges_v2.
- 02:56 Synthetic nothink_nudges_v2 finished: 99.2% ± 1.3% (128/129); only commit_06 s3 fails. 15 forced finals, all correct. REPORT §2.2 updated; charts regenerated. Chain now on itbench nothink_nudges_v2_guard (last).
- 03:05 Per-scenario chart redrawn as a configs×scenarios heatmap (grouped bars had become unreadable with 7 configs); partial runs excluded from charts (analyze.py; backup analyze_v1_backup.py).
- 06:21 Local chain complete (LOCAL_IMPROVE2_DONE). nothink_nudges_v2_guard: 84.4% ± 7.7% (same as without guard; different failing runs). REPORT §2.1/§2.5 status updated; charts regenerated.
