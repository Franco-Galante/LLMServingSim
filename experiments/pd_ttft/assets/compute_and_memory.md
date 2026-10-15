# Compute (MFU) and memory bandwidth (MBU) from the simulator outputs

`gpu_report.py` reconstructs two utilization metrics per GPU. LLMServingSim does
not model FLOPs or bytes: it replays profiled kernel latencies. The metrics are
therefore rebuilt from what a run writes (`<run>.csv`, `<run>.log`) plus the
model's dimensions.

| Metric | Definition | Computed for |
|---|---|---|
| **MFU**, model FLOPs utilization | model FLOPs done per second ÷ peak FLOP/s | the P/D **prefill** GPU (compute-bound) |
| **MBU**, memory bandwidth utilization | bytes read from GPU memory per second ÷ peak bandwidth | the P/D **decode** GPU (bandwidth-bound) |

## Only for disaggregated (P/D) runs

Both metrics need to know **which GPU ran which tokens**. The outputs give that
only under disaggregation:

- **Prefill GPU (instance 0):** runs every prompt token and nothing else.
- **Decode GPU (instance 1):** generates every output token and nothing else.

In a **colocated** run each GPU mixes prefill chunks and decode tokens in the
same steps. The log's throughput lines are for the whole cluster, and the CSV
does not say which steps a request shared with which others. The split per GPU
cannot be recovered, so colocated runs get **no MFU/MBU**. `gpu_report.py`
prints a warning saying so on every run.

## Inputs

| From | Values used |
|---|---|
| `<run>.csv`, one row per request | `input` (prompt tokens), `arrival`, `queuing_delay`, `TTFT` (ns) |
| `<run>.log`, one tick per simulated second | generated tokens/s (whole cluster); per instance: requests running and memory used (MiB) |
| `configs/model/<model>.json` | layers `L`, hidden size `h`, `q_dim`, `kv_dim`, intermediate size, vocabulary `V` |
| `profiler/perf/RTXPRO6000/hardware.yaml` (or the cluster config's `npu_mem.mem_bw`, if set) | peak bandwidth: 1597.6 GB/s (RTX PRO 6000, vendor figure); the value the simulator itself uses |
| constant `PEAK_TFLOPS` in `gpu_report.py` | 500 TFLOPS bf16 dense (datasheet: 1000 with 2:4 sparsity, halved). Not in the simulator |

## MFU, prefill GPU

FLOPs of one request's whole prefill, `n` = prompt tokens:

```
FLOPs(n) = 2 · P_blocks · n          linear layers, every prompt token
         + 2 · L · q_dim · n²        attention (QKᵀ and AV, causal: half the n×n matrix)
         + 2 · h · V                 lm_head, last token only

P_blocks = L · ( h·(q_dim + 2·kv_dim) + q_dim·h + 3·h·intermediate )
```

1. **Spread each request's FLOPs** evenly over its time on the prefill GPU.
   - It starts when the request is first scheduled: `arrival + queuing_delay`.
   - It ends with its first token: `arrival + TTFT`.
2. **Sum per 1-second bin:** `MFU(t) = FLOPs in second t ÷ PEAK_TFLOPS`.
3. **Run value:** total FLOPs ÷ (busy span × peak), where the busy span runs
   from the first request's start to the last request's first token.

**Why the CSV and not the log's prompt throughput:** the log counts every
prefill-like step in the cluster, including the decode GPU's recomputation
after a preemption. Without preemptions the two agree exactly (llama8b, all
rates: 852,138 tokens each). Once the decode GPU preempts, the log overcounts
(qwen32b at 3 req/s: 1,281,012 logged vs 865,226 real prompt tokens).

## MBU, decode GPU

Each decode step reads all the weights plus the KV cache of every running
request, and makes one token per running request:

```
steps/s(t)   = generated tokens/s(t) ÷ running(t)
bytes/step   = W + KV(t)
KV(t)        = decode GPU memory used(t) − W
W            = weights = the prefill GPU's lowest memory reading (idle: weights only)
MBU(t)       = steps/s(t) · bytes/step ÷ peak bandwidth
```

- **`running(t)`** is the mean of the samples at `t−1` and `t`, because the
  token count covers that whole second.
- **The decode GPU is the only one generating tokens**, so the cluster-wide
  generated tokens/s belongs to it alone.
- **Run value:** the mean of `MBU(t)` from the first to the last second with a
  running request.

**Check:** llama8b at 2 req/s has a mean token gap of 13.8 ms. Then
(16.1 GB weights + about 2.8 GB KV) ÷ 13.8 ms ≈ 1.36 TB/s ≈ 85% of 1597.6 GB/s,
which matches the reconstructed 86.2%.

## Limits

- **1-second resolution.** The simulator does not log per-step token counts or
  durations.
  - Spreading FLOPs evenly and sampling `running` once per second make
    individual seconds noisy, and a single second can exceed 100%.
  - Read the run means: the `mfu_mean_pct` / `mbu_mean_pct` columns and
    `plots/<model>_mfu_mbu_vs_rate.png`.
- **The decode GPU's recomputation is not in MBU.** After a preemption it
  recomputes prompts (compute-heavy steps that make no tokens).
  - Its steps/s is then underestimated.
  - This is why the overloaded qwen32b decode GPU, which preempts constantly,
    shows about 65% MBU against 83–86% for runs that don't preempt (about 80%
    in its own seconds without recompute, about 60% in those with).
- **"Model FLOPs" only.** Activation traffic, norms, rotary embeddings and
  padding are not counted, as in the standard MFU/MBU definitions.
- **The peak FLOP/s is a datasheet figure,** not measured.
  - If `PEAK_TFLOPS` changes, every MFU scales by the inverse ratio.

## How reliable: scores (1 = unreasonable, 10 = matches a real GPU)

Overall about **6/10**: good as run means without preemption, weak per second
and on the overloaded runs.

| Value | Score | Why |
|---|---|---|
| MFU, run mean | 7 | Standard FLOP formula; inputs (prompt lengths, timings) exact from the CSV. Main doubt: the 500 TFLOPS peak is the datasheet's sparse figure halved, not a stated or measured value, and every MFU scales with it |
| MBU, run mean, no preemption | 7 | Agrees with an independent check (token gap → 85% vs 86.2%). Small approximations: KV from allocated blocks (the last block is partly empty), activations ignored |
| MBU with preemptions (qwen32b ≥ 1.5 req/s) | 4 | Biased low by about 15–20 points: recompute steps make no tokens, so the token-based step count misses them (about 80% vs 60% in seconds without vs with recompute) |
| Per-second values (timelines) | 3–4 | Even spreading, one `running` sample per second and 1-s bins make single seconds noisy; some exceed 100%. Good for shape, not for values |

**Against real hardware:** the simulator's step times come from latencies
profiled on a real RTX PRO 6000 with vLLM, so kernel efficiency is realistic.
It leaves out CPU overheads between steps (scheduling, kernel launches), so a
real vLLM run would likely show somewhat lower MFU/MBU. Lookups outside the
profiled range are linear extrapolations.

**What would raise the scores:**
1. **A per-step log from the simulator** (tokens, recompute tokens, KV in use,
   step duration per instance): exact steps/s, removes the preemption bias and
   the per-second noise.
2. **A measured peak FLOP/s** (a large bf16 GEMM benchmark on the GPU) in place
   of the datasheet figure.
3. **Ground truth:** a real vLLM run with DCGM counters
   (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`) to compare.

## Outputs, under the experiment's output folder (`outputs/pd_ttft_long/`)

- `gpu_report.csv`: `mfu_mean_pct` on P/D prefill rows, `mbu_mean_pct` on P/D
  decode rows, empty elsewhere.
- `gpu_report.txt`: the same, as the `MFU %` / `MBU %` columns.
- `plots/<model>_mfu_mbu_vs_rate.png`: run means vs request rate.
- `plots/mfu_mbu_timelines/<model>_rate<R>.png`: per-second MFU and MBU.
