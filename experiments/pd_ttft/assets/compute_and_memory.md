# Compute (MFU) and memory bandwidth (MBU) from the simulator outputs

`gpu_report.py` computes two utilization metrics for every GPU. LLMServingSim
does not model FLOPs or bytes: it replays profiled kernel latencies. The
metrics are rebuilt from what each step did, plus the model's dimensions.

| Metric | Definition | Limits |
|---|---|---|
| **MFU**, model FLOPs utilization | model FLOPs done ÷ (time × peak FLOP/s) | prefill (compute-bound) |
| **MBU**, memory bandwidth utilization | bytes read from GPU memory ÷ (time × peak bandwidth) | decode (bandwidth-bound) |

## Source: the step log

`run.sh` runs the simulator with `--step-log <run>.steps.csv`, a simulator
feature added for this purpose (see `added_features/02_step_log.md`). It
writes one row per GPU step (one completed batch):

| Column | Used for |
|---|---|
| `start_ns`, `end_ns` | the step's duration |
| `prefill_tokens`, `decode_tokens` | real tokens: linear-layer FLOPs |
| `attn_pairs` = Σ q·k + q(q+1)/2 | attention FLOPs (q = tokens computed this step, k = tokens already in KV) |
| `sequences` | `lm_head` FLOPs (one row per sequence) |
| `kv_read_tokens` = Σ k + q | KV bytes read |
| `recompute_tokens`, `forward_tokens`, `gen_tokens` | checks and diagnostics |

**Every row belongs to one instance,** and here an instance is one GPU
(TP=1, PP=1). So colocated and P/D runs are both covered.

## Formulas

Per step, with `L` layers, hidden size `h`, vocabulary `V`:

```
FLOPs = 2 · P_blocks · (prefill_tokens + decode_tokens)    linear layers
      + 4 · L · q_dim · attn_pairs                         attention (QKᵀ and AV)
      + 2 · h · V · sequences                              lm_head
bytes = W + kv_read_tokens · 2 · L · kv_dim · 2            weights + K and V, bf16

P_blocks = L · ( h·(q_dim + 2·kv_dim) + q_dim·h + 3·h·intermediate )
W        = every weight (blocks, norms, embedding, lm_head) · 2 bytes
```

`W` matches the simulator's own weight figure (15,316.5 MiB for Llama-3.1-8B,
62,488.8 MiB for Qwen3-32B).

**Run values, per GPU:**

| Value | = | Answers |
|---|---|---|
| `busy %` | Σ step time ÷ run span | how much of the time the GPU is working |
| `mfu_busy_pct`, `mbu_busy_pct` | Σ work ÷ (Σ step time × peak) | how efficiently it works while working |
| `mfu_wall_pct`, `mbu_wall_pct` | Σ work ÷ (run span × peak) = busy × busy % | its use over the whole run, idle time included |

**Per second** (timelines): each step's work is spread over its own interval.
Steps last 12–350 ms, so the values are close to exact.

**Peaks:**
- **Bandwidth:** 1597.6 GB/s, the value the simulator itself uses
  (`profiler/perf/RTXPRO6000/hardware.yaml`).
- **FLOP/s:** 500 TFLOPS bf16 dense (`PEAK_TFLOPS`). This is the datasheet's
  1000 TFLOPS with 2:4 sparsity, halved; it's not in the simulator. Every MFU
  scales with it.

## Checks (all 40 runs pass)

- **Same results:** the per-request CSVs are byte-identical with and without
  `--step-log`.
- **Token identity:** Σ `gen_tokens` = the run's "Total generated tokens", and
  Σ (prefill + decode − recompute) = input + output − requests (each request's
  first token comes from its prefill step).
- **Timing:** no two steps of one GPU overlap.
- **MBU against token gaps:** at llama8b P/D 2 req/s, (16.1 GB + ≈2.8 GB KV) ÷
  13.6 ms step ≈ 1.39 TB/s ≈ 87%, against 86.5% from the log.

## Results (RTXPRO6000, 1000 requests per run)

| | MFU busy | MBU busy | Note |
|---|---|---|---|
| P/D prefill GPU | 61–68% (llama), 65–74% (qwen) | 12–26% | compute-bound. Busy % rises with rate, from 8% to 100% |
| P/D decode GPU | 4–19% | 81–87% (llama); 85% → 65% (qwen) | bandwidth-bound and always busy |
| Colocated GPU | 4% → 23% (llama), 5% → 29% (qwen) | 87% → 65% (llama), 84% → 57% (qwen) | mixes both; always busy |

**Why the overloaded qwen decode GPU drops to about 65% MBU:** it preempts
constantly, and recomputing a preempted request is prefill-like work. At
3 req/s:

| Decode GPU steps | Share of time | Mean step | MBU | MFU |
|---|---|---|---|---|
| plain decode | 61% | 71 ms | 78% | 14% |
| containing recompute | 39% | 121 ms | 47% | 26% |

The drop is real: the GPU spends 40% of its time on compute-heavier steps.

## Fallback without a step log

For a run without `<run>.steps.csv`, `gpu_report.py` reconstructs MFU and MBU
from the per-second log and the per-request CSV, and prints a warning naming
those runs. **This works only for P/D runs:** a colocated GPU mixes prefill and
decode, and those outputs don't say which tokens ran where.

- **Prefill GPU MFU:** each request's prefill FLOPs, spread over first-scheduled
  → first-token (`arrival + queuing_delay` → `arrival + TTFT`), in 1-s bins.
  Identical to the step log's wall value.
- **Decode GPU MBU:** steps/s ≈ generated tokens/s ÷ running requests, and bytes
  per step = W + KV in use. This undercounts by up to 7 points on llama8b P/D at
  ≥ 10 req/s (73–75% vs 81%), and single seconds can exceed 100%.

## How reliable: scores (1 = unreasonable, 10 = matches a real GPU)

Overall about **7/10** with the step log.

| Value | Score | Why |
|---|---|---|
| MFU, run values | 7 | Exact token and attention counts per step. The main doubt is the 500 TFLOPS peak, which is not measured |
| MBU, run values | 8 | Exact per-step bytes and durations, and they match the token-gap check |
| Per-second values | 7 | Exact steps spread over 1-s bins |
| Fallback (no step log) | 4–7 | MFU 7 (matches the step log); decode MBU up to 7 points low; P/D only |

**Against real hardware:** the step durations come from latencies profiled on
a real RTX PRO 6000 with vLLM, so kernel efficiency is realistic. Not modelled:
CPU overheads between steps (a real vLLM run would likely show somewhat lower
MFU/MBU) and anything outside the profiled range (extrapolated linearly).

**What would raise the scores further:**
1. **A measured peak FLOP/s:** a large bf16 GEMM benchmark on the GPU.
2. **Ground truth:** a real vLLM run with DCGM counters
   (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`).

## Not counted

- **CUDA-graph padding:** a decode batch is padded up to a capture size
  (`forward_tokens` − real tokens; up to about a third extra rows at low load).
  It is real GPU work but not model FLOPs, so MFU leaves it out.
- **Activations, norms, rotary embeddings:** as in the standard definitions.
- **`pp_size > 1`:** a step's `start→end` spans every pipeline stage, so it is
  not one GPU's busy time. The experiments use TP=1 and PP=1.

## Outputs, under the experiment's output folder (`outputs/pd_ttft_long/`)

- `gpu_report.csv` / `.txt`: `busy_pct`, `step_ms_mean`, `mfu_busy_pct`,
  `mfu_wall_pct`, `mbu_busy_pct`, `mbu_wall_pct`, and `util_source`
  (`step_log` or `reconstructed`).
- `plots/<model>_mfu_mbu_vs_rate.png`: busy (solid) and wall (dashed) values vs
  rate.
- `plots/mfu_mbu_timelines/<model>_rate<R>.png`: per-second MFU and MBU per GPU.
