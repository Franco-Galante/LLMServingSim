# 02 · Per-step batch log (`--step-log`)

| | |
|---|---|
| Status | **implemented and validated**, not committed yet |
| Base | `develop` (v1.2.0 + [01](01_pd_deferred_handoff.md)) |
| Files | `serving/core/step_log.py` (new), `serving/__main__.py`, `serving/core/scheduler.py`, `serving/core/request.py`; docs: `docs/docs/reference/cli-flags.md`, `docs/docs/simulator/reading-output.md`, `docs/docs/examples/disaggregated/prefill-decode-split.mdx`, `docs/docs/contributor/codebase-tour.md`, `serving/README.md`, `AGENTS.md`, `CHANGELOG.md` (+ synced `docs/src/pages/changelog.md`) |
| Affects | nothing unless the flag is given |

## Why

`experiments/pd_ttft/gpu_report.py` reconstructs MFU and MBU from 1-second log
lines (see `experiments/pd_ttft/assets/compute_and_memory.md`). That
reconstruction:

- **works only for P/D runs:** colocated GPUs mix prefill and decode, and the
  log cannot split them;
- **underestimates MBU by about 15–20 points** when the decode GPU preempts,
  because recompute steps produce no tokens and so are not counted as steps;
- **is noisy per second,** with single seconds above 100%.

All three come from one missing piece of information: what each GPU step did,
and how long it took.

## Change

- **New opt-in flag `--step-log <path>.csv`** (supports `{run_id}`; a relative
  path is relative to where you run from). One row per completed batch,
  written by `StepLog.record` (`serving/core/step_log.py`), called at the end
  of `Scheduler.add_done`.
- **Almost everything comes from the existing `Batch`.** Two additions:
  - `Batch.recompute_tokens`, snapshotted in `_build_batch` from a new
    `Request.max_computed_tokens` high-water mark (a speculative-decode rollback
    lowers it, since rejected slots get new tokens);
  - `Batch.npu_end`, each NPU's completion clock.

| Column | Source | Used for |
|---|---|---|
| `instance`, `batch_id` | batch | which GPU |
| `start_ns`, `end_ns` | `batch.batch_time`, `finish` | exact step duration |
| `npu_end_ns` | each NPU's completion | per-stage timing, kept for future PP support |
| `num_prefill`, `num_decode` | batch | composition |
| `prefill_tokens` | Σ prefill chunk tokens | linear-layer FLOPs |
| `recompute_tokens` | the part of those from preempted requests | work that makes no tokens |
| `decode_tokens` | Σ real decode tokens | linear-layer FLOPs |
| `forward_tokens` | `batch.total_len`, including CUDA-graph / DP padding | padding diagnostics |
| `sequences` | requests in the step | `lm_head` FLOPs |
| `attn_pairs` | Σ causal query×key pairs | attention FLOPs |
| `kv_read_tokens` | Σ KV length of the step's requests | KV bytes read |
| `kv_used_bytes` | `batch.kv_size` | memory check |
| `gen_tokens` | tokens produced by the step | throughput check |

Per step, `gpu_report.py` will then compute (FLOP terms as in `compute_and_memory.md`):

```
MFU = (2·P_blocks·tokens + 4·L·q_dim·attn_pairs + 2·h·V·sequences) ÷ (duration · TP · peak FLOP/s)
MBU = (weights + kv_read_tokens · KV bytes/token) ÷ (duration · peak bandwidth)
```

- **Run values** come in two forms: **busy** (Σ work ÷ Σ step time) and
  **wall** (Σ work ÷ run span, including idle time).
- **Colocated runs are covered too,** since every row belongs to one instance.
- **Runs without a step log** fall back to the per-second reconstruction.

## Parallel runs: no race conditions

| Risk | Guard |
|---|---|
| Two runs writing one file | one path per run (`run.sh`: `$base.steps.csv`; elsewhere `{run_id}`), and no shared file |
| Two `run.sh` on one output folder | the existing `flock` on `$OUT/.run.lock` |
| Relative path broken by the `chdir` into `astra-sim/` | made absolute at startup, like `--output` |
| A killed run leaving a log that looks complete | written to `<path>.tmp`, renamed with `os.replace` (atomic) only on normal exit |
| Stale files from a failed run | `run.sh` deletes `.steps.csv` / `.tmp` before each run and on failure |
| Concurrent writes inside one run | none: the simulator is single-threaded |

## Validation (done)

1. **Flag off:** the full `./serving/validate.sh` passes: 70/70 scenarios and
   all 10 `bench/examples` outputs byte-identical.
2. **Flag on:** in the 40-run `experiments/pd_ttft` sweep, every per-request CSV
   is byte-identical to the run without it.
3. **Consistency, all 40 runs:**
   - Σ `gen_tokens` = "Total generated tokens";
   - Σ (prefill + decode − recompute) tokens = input + output − requests;
   - no two steps of one GPU overlap;
   - no `.tmp` file left.
4. **Bug found and fixed during validation:** `batch.total_len` is the
   *padded* forward size by the time a batch completes (`_pad_dp_round` /
   CUDA-graph padding), so `decode_tokens` first counted padding rows. They now
   come from `q_list`, and the padded size is the separate `forward_tokens`
   column.

## Findings

- **The old reconstruction was right about MFU, and mostly right about MBU.**
  The prefill MFU wall value is identical. Decode MBU was up to 7 points low on
  llama8b P/D at ≥ 10 req/s.
- **The overloaded qwen32b decode GPU really does use about 65% MBU.** It spends
  about 40% of its time on recompute steps (121 ms, 47% MBU, 26% MFU) beside
  plain decode steps (71 ms, 78% MBU). The earlier explanation, that the 65%
  was an undercount, was wrong.
- **Colocated GPUs now get values too:** MFU 4% → 29%, MBU 87% → 57% as the load
  rises.

## Limits

- **`pp_size > 1`:** a batch's `start→end` spans every pipeline stage, while each
  GPU spends only one stage on it and meanwhile works on other batches. Stage
  start times are not tracked, so MFU/MBU are not computed for PP (the
  experiments use TP=1, PP=1).
- **Peak FLOP/s is still the datasheet figure** (500 TFLOPS bf16 dense).
- **"Model FLOPs" only:** CUDA-graph padding rows, norms and activation traffic
  are not counted.
