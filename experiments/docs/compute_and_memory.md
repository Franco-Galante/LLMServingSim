# Compute (MFU) and memory bandwidth (MBU) from the simulator outputs

The experiments' report scripts (`experiments/pd_balance/report.py`) compute
two utilization metrics for every GPU. LLMServingSim does not model FLOPs or
bytes: it replays profiled kernel latencies. The metrics are rebuilt from what
each step did, plus the model's dimensions.

| Metric | Definition | Limits |
|---|---|---|
| **MFU**, model FLOPs utilization | model FLOPs done ÷ (time × peak FLOP/s) | prefill (compute-bound) |
| **MBU**, memory bandwidth utilization | bytes read from GPU memory ÷ (time × peak bandwidth) | decode (bandwidth-bound) |

## Source: the step log

The experiment's `run.sh` runs the simulator with `--step-log <run>.steps.csv`
(see `added_features/02_step_log.md`). It writes one row per GPU step (one
completed batch):

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

**Run values, per GPU,** over the experiment's measurement window (in
`pd_balance`, the steady-state window 120-600 s; a step cut by the window's
edges counts pro rata):

| Value | = | Answers |
|---|---|---|
| `busy %` | Σ step time ÷ window length | how much of the time the GPU is working |
| `mfu_busy_pct`, `mbu_busy_pct` | Σ work ÷ (Σ step time × peak) | how efficiently it works while working |
| `mfu_wall_pct`, `mbu_wall_pct` | Σ work ÷ (window length × peak) = busy × busy % | its use over the whole window, idle time included |

A prefill GPU's "while busy" MFU is high (about 65%) even at low load, because
every prefill step is a large matrix multiply; load changes its busy %, and so
its wall value.

**Per second** (timelines): each step's work is spread over its own interval.
Steps last 12–350 ms, so the values are close to exact.

**Peaks:**
- **Bandwidth:** 1597.6 GB/s, the value the simulator itself uses
  (`profiler/perf/RTXPRO6000/hardware.yaml`).
- **FLOP/s:** 500 TFLOPS bf16 dense (`PEAK_TFLOPS`). This is the datasheet's
  1000 TFLOPS with 2:4 sparsity, halved; it's not in the simulator. Every MFU
  scales with it.

## Checks

Done when the step log was introduced, on 40 runs of an earlier sweep
(Llama-3.1-8B and Qwen3-32B, P/D and colocated):

- **Same results:** the per-request CSVs are byte-identical with and without
  `--step-log`.
- **Token identity:** Σ `gen_tokens` = the run's "Total generated tokens", and
  Σ (prefill + decode − recompute) = input + output − requests (each request's
  first token comes from its prefill step).
- **Timing:** no two steps of one GPU overlap.
- **MBU against token gaps:** at Llama-3.1-8B P/D 2 req/s, (16.1 GB + ≈2.8 GB
  KV) ÷ 13.6 ms step ≈ 1.39 TB/s ≈ 87%, against 86.5% from the step log.

## Why a preempting decode GPU shows lower MBU

Recomputing a preempted request is prefill-like work on the decode GPU, so
steps that contain recompute are more compute-heavy and read relatively fewer
bytes per second. On an overloaded Qwen3-32B decode GPU (1P+1D at 3 req/s):

| Decode GPU steps | Share of time | Mean step | MBU | MFU |
|---|---|---|---|---|
| plain decode | 61% | 71 ms | 78% | 14% |
| containing recompute | 39% | 121 ms | 47% | 26% |

The drop in MBU under overload is real, not an artifact of the calculation.

## How reliable: scores (1 = unreasonable, 10 = matches a real GPU)

Overall about **7/10**.

| Value | Score | Why |
|---|---|---|
| MFU, run values | 7 | Exact token and attention counts per step. The main doubt is the 500 TFLOPS peak, which is not measured |
| MBU, run values | 8 | Exact per-step bytes and durations, and they match the token-gap check |
| Per-second values | 7 | Exact steps spread over 1-s bins |

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

## Outputs (`pd_balance`, under `outputs/pd_balance/`)

- `gpu_report.csv` / `.txt`: per GPU and rate, `busy_pct`, `step_ms_mean`,
  `mfu_busy_pct`, `mfu_wall_pct`, `mbu_busy_pct`, `mbu_wall_pct`, plus running,
  waiting and memory.
- `plots/G<G>_*/gpu_load.png`: busy %, MFU and MBU while busy per GPU role vs rate.
- `plots/G<G>_*/timelines/<layout>_rate<R>.png`: per-second MFU and MBU per GPU.
