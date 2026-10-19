# Time horizon: why every workload spans 600 s of arrivals

The runs in `outputs/pd_balance_check/` exist only to choose the simulated
time horizon (`DURATION_S`, `WARMUP_S` in `prepare.py`). They are not part of
the comparison itself.

## The problem with a fixed request count

Every workload used to hold 1000 requests. At rate R they all arrive within
1000 / R seconds, while a Qwen3-32B request lives about 40-60 s (about 665
output tokens at 50-90 ms each). Load builds up over the first minute or so:
running requests ≈ rate × lifetime, and the decode KV cache fills as they
accumulate. A fast run therefore ended before reaching steady state:

| 12 req/s | Requests | Arrival span | TTFT p99 (s) | TPOT p99 (ms) | Preemptions | SLO |
|---|---|---|---|---|---|---|
| 2P+7D | 1000 | 83 s | 2.01 | 81 | 111 | borderline |
| 2P+7D | 2000 | 167 s | 1.73 | 128 | 517 | no |
| colocated ×9 | 1000 | 83 s | 0.72 | 86 | 0 | yes |
| colocated ×9 | 2000 | 167 s | 19.5 | 109 | 481 | no |

Even 2000 requests were not enough: completions were still rising (about
9.3 req/s served in the second half, against 12 offered).

## The fix: a fixed arrival span, metrics over a steady-state window

- Every workload spans the same `DURATION_S` = 600 s of arrivals, so a rate R
  has about 600 × R requests. All rates share one request sequence: only the
  arrival times scale with R.
- Run values use only the window [`WARMUP_S`, `DURATION_S`] = [120 s, 600 s]:
  requests arriving in it (latencies), completions in it (served), samples and
  step time in it (utilization, running, waiting, memory).
- Run totals (preemptions, recomputed tokens, deferred handoffs) are reported
  per 1000 requests, since request counts differ between rates.

## Check: is 600 s long enough?

Each layout ran with 600 s and with 1200 s of arrivals (same request sequence).
If 600 s reaches steady state, the window 120-600 s and the second half of the
1200 s run (600-1200 s) must agree.

**Below capacity (9 req/s): they agree on every metric.**

| 9 req/s | Window (s) | Served (req/s) | TTFT p99 (s) | TPOT mean (ms) | Delay mean (s) |
|---|---|---|---|---|---|
| 2P+7D, 600 s run | 120-600 | 8.99 | 1.15 | 64.0 | 42.8 |
| 2P+7D, 1200 s run | 600-1200 | 8.94 | 1.18 | 63.9 | 42.8 |
| colocated ×9, 600 s run | 120-600 | 8.97 | 0.80 | 72.0 | 48.1 |
| colocated ×9, 1200 s run | 600-1200 | 8.96 | 0.76 | 71.9 | 48.1 |

**Above capacity (11 req/s): served agrees, latencies do not.** There is no
steady state when the arrival rate exceeds capacity: the backlog grows without
limit, so latency depends on how long the overload lasted. The served rate is
stable and is the layout's capacity.

| 11 req/s | Window (s) | Served (req/s) | TTFT p99 (s) | TPOT mean (ms) | Delay mean (s) |
|---|---|---|---|---|---|
| 2P+7D, 600 s run | 120-600 | 9.50 | 1.54 | 140 | 93 |
| 2P+7D, 1200 s run | 600-1200 | 9.40 | 1.91 | 280 | 183 |
| colocated ×9, 600 s run | 120-600 | 10.35 | 30.5 | 102 | 75 |
| colocated ×9, 1200 s run | 600-1200 | 10.21 | 83.9 | 105 | 117 |

The overload shows up in different places: colocated in the queue (TTFT),
P/D in the decode GPUs (deferred handoffs, recompute, TPOT).

**Conclusion:** 600 s of arrivals with a 120 s warm-up gives steady-state
values below capacity. Above capacity only the served rate is meaningful, so
the sweeps go at most one or two rates past each layout's capacity.

## Per-GPU capacity in steady state (from the same runs)

| GPU | Observation | Capacity per GPU |
|---|---|---|
| Prefill | 74% busy at 4.5 req/s, 88% at 5.5 | about 6-6.5 req/s (extrapolated) |
| Decode | 1.29 req/s: 54 running, 82% memory, no preemptions; overloaded: 9.5 / 7 GPUs | about 1.35-1.45 req/s, limited by KV (~80 requests) |
| Colocated | 1 req/s: 48 running, 79% memory; overloaded: 10.3 / 9 GPUs | about 1.15 req/s |

So the balanced prefill:decode ratio is about 1:4.5, and at that ratio P/D and
colocated have about the same capacity per GPU (about 1.13 vs 1.15 req/s).
This is what the layouts in `prepare.py` (1P+3D, 1P+5D, 2P+9D) are chosen
from. The sweep itself then measured (served rate once saturated): one
prefill GPU about 6.65 req/s (1P+5D), a decode GPU about 1.45 req/s before it
starts preempting and about 1.35 once it does (1P+3D: 4.05 / 3), and 2P+9D
about 13.4 req/s against 13.25 for colocated ×11. 3P+9D (decode-bound) peaks
at 13.5 req/s (1.5 per decode GPU) and falls to 12.3 under thrashing, against
14.4 for colocated ×12.

## Files in `outputs/pd_balance_check/`

| Files | Run |
|---|---|
| `n2000_rate12.jsonl`, `*_n2000_rate12.*` | 2000 requests at 12 req/s (fixed-count check) |
| `T600_rate11.jsonl`, `T1200_rate11.jsonl`, `*_T{600,1200}_rate11.*` | 600 s / 1200 s of arrivals at 11 req/s (above capacity) |
| `T600_rate9.jsonl`, `T1200_rate9.jsonl`, `*_T{600,1200}_rate9.*` | 600 s / 1200 s at 9 req/s (below capacity) |

Layouts: `pd_2p7d` and `colocated_9` (Qwen3-32B, TP1, RTXPRO6000), with the
configs `prepare.py` generated at that time (same batching as now). These
workloads came from an inline script with one generator for lengths and
arrivals (seed 42), so their request sequence differs from `prepare.py`'s.
