# 01 · Queue P/D handoffs when the decode KV cache is full

| | |
|---|---|
| Status | **done**: commit `4d68d9de86f77c6747d3e2faa473afcf506e6a63` on `develop` |
| Base | upstream `casys-kaist/LLMServingSim` v1.2.0 (`b09c7aa`) |
| Files | `serving/core/scheduler.py`, `serving/core/request.py`, `serving/__main__.py` (+38 −15) |
| Affects | P/D disaggregated runs only (`pd_type: prefill` / `decode` instances) |

## Problem

In a P/D run, a request is prefilled on the prefill GPU and then handed to the
decode GPU (`Scheduler.add_decode`). The handoff claimed KV blocks on the decode
GPU at once and, when they were not free, **raised**:

```
RuntimeError: [Scheduler] ... decode instance cannot admit request N:
T tokens need more blocks than the pool has free (F of B)
```

So any P/D run whose decode GPU filled up **crashed** instead of slowing down.
Upstream v1.2.0 still has this behaviour.

## Change

`add_decode` now queues the handoff instead of raising, as vLLM's decode side
does: it keeps such a request waiting until blocks free up.

- **The blocks fit and nothing else is waiting:** admitted directly, as before.
- **Otherwise:** the request is put in the decode instance's `waiting` queue
  (arrival order, first come first served) and flagged `awaiting_decode_admission`.
- **Admission:** the normal phase B admits it once blocks are free. Its
  `num_computed_tokens` is already the full prompt, so it is scheduled as a
  one-token decode that claims blocks for the whole sequence.
- **No double counting:** on admission the flag skips `record_prefix_stats`,
  because the prompt was already counted on the prefill instance.
- **Counter:** `Scheduler.num_decode_deferred`. The run summary prints
  `Deferred P/D handoffs (decode KV full): N`, only when N > 0.

**Why queue every later handoff once one waits:** otherwise a small request could
overtake a large one forever, and the large one would never be admitted.

## Effect

- **Runs that never fill the decode KV:** identical results, with no deferred
  handoffs.
- **Runs that do:** they complete. The wait appears as a longer gap between the
  first token (made on the prefill GPU) and the second (decode GPU), and as
  waiting requests on the decode instance.

Deferred handoffs in the `experiments/pd_ttft` sweep (1000 requests per run;
that experiment has since been removed).
Every non-zero row would have crashed without this change:

| Model | Rates with deferrals | Deferred handoffs |
|---|---|---|
| Llama-3.1-8B | 15, 20, 30, 40 req/s | 115 – 413 |
| Qwen3-32B | 1.5 – 10 req/s | 730 – 907 |

## Validation

- `./serving/validate.sh` (full, on v1.2.0 + this change): 70/70 scenarios match
  their baselines, and all 10 `bench/examples` outputs are byte-identical. No
  existing scenario fills a decode KV cache, so none of them changed.
- The 40-run `experiments/pd_ttft` sweep and the 15-run `experiments/pd_kv_sweep`
  (both since removed) complete with no failures.

## Limits

- **No real-vLLM ground truth checks the queueing policy.** vLLM's handling
  depends on its KV connector; first come first served is the simplest faithful
  choice, not a measured one.
- **The KV transfer is still charged at prefill time.** A deferred request's KV
  is assumed to wait at the decode side with no further transfer cost and no
  memory charge while it waits.
