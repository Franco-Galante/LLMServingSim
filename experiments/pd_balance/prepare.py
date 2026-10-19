"""Generate the cluster config and workloads for the P/D balance study.

Run from the repo root (stdlib only, works on the host or in the sim container):

    python3 experiments/pd_balance/prepare.py

Writes, under experiments/pd_balance/, for every (P, N) in PD_LAYOUTS:
    configs/qwen32b/pd_<P>p<N>d.json      P prefill GPUs + N decode GPUs, all TP1
    configs/qwen32b/colocated_<P+N>.json  P+N GPUs, each prefill + decode (vLLM
                                          default batching), TP1: the same GPU count
and workloads/qwen32b/rate<R>.jsonl for every rate in RATES: Poisson arrivals
at R req/s (the arrival rate, i.e. offered load) for DURATION_S seconds, so
about R * DURATION_S requests.

Every rate gets the same arrival time span rather than the same request count.
A request lives about 40-60 s, so a short span never reaches steady state (with
1000 requests, 12 req/s lasts only 83 s and showed 5x fewer preemptions than a
longer run at the same rate). report.py measures only the steady-state window
[WARMUP_S, DURATION_S]. How 600 s was chosen: assets/time_horizon.md.

Why these layouts: in steady state one prefill GPU serves about 6-6.5 req/s
of Qwen3-32B prompts and one decode GPU about 1.35-1.45 req/s before its KV
fills (24.5 GiB, ~80 requests), so the balanced ratio is about 1:4.5. 1P+3D is
decode-bound, 1P+5D prefill-bound, 2P+9D about balanced (slightly
prefill-bound), and 3P+9D decode-bound: its 9 decode GPUs are fed faster than
they can serve. Configs of layouts no longer listed are removed.

All rates share one request sequence (common random numbers): request i has
the same input/output lengths at every rate, and its arrival time is u_i / R
for one fixed unit-rate Poisson sequence u. A rate R keeps the requests with
u_i / R < DURATION_S.
"""

import copy
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# Arrival span of every workload, and the warm-up excluded by report.py.
DURATION_S = 600
WARMUP_S = 120
SEED = 42
MODEL_KEY = "qwen32b"
MODEL_NAME = "Qwen/Qwen3-32B"
SOURCE_WORKLOAD = "workloads/sharegpt-qwen3-32b-300-sps10.jsonl"
# Every rate any layout runs (run.sh picks the rates per layout).
RATES = [1, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 10, 11, 11.5, 12, 12.5,
         13, 13.5, 14, 14.5, 15, 16, 17]
# (prefill GPUs, decode GPUs). Must match SYSTEMS in run.sh.
PD_LAYOUTS = [(1, 3), (1, 5), (2, 9), (3, 9)]

# Network and per-role batching: prefill batches many tokens of few requests,
# decode many requests of one token each.
LINK_BW = 50
LINK_LATENCY = 5000
BATCHING = {
    "prefill": {"max_num_seqs": 32, "max_num_batched_tokens": 2048},
    "decode": {"max_num_seqs": 256, "max_num_batched_tokens": 256},
}


def instance(pd_type):
    # No npu_mem: GPU memory size and bandwidth come from the measured
    # profiler/perf/RTXPRO6000/hardware.yaml. A colocated instance (pd_type None)
    # keeps vLLM's default batching.
    inst = {"model_name": MODEL_NAME, "hardware": "RTXPRO6000", "tp_size": 1,
            "pd_type": pd_type}
    if pd_type:
        inst.update(copy.deepcopy(BATCHING[pd_type]))
    return inst


def cluster(instances):
    return {
        "num_nodes": 1,
        "link_bw": LINK_BW,
        "link_latency": LINK_LATENCY,
        "nodes": [{
            "num_instances": len(instances),
            "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
            "instances": instances,
        }],
    }


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
        f.write("\n")


def main():
    cfg_dir = os.path.join(HERE, "configs", MODEL_KEY)
    wl_dir = os.path.join(HERE, "workloads", MODEL_KEY)
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(wl_dir, exist_ok=True)

    for stale in os.listdir(cfg_dir):            # generated files only
        if stale.endswith(".json"):
            os.remove(os.path.join(cfg_dir, stale))
    for p, n in PD_LAYOUTS:
        pd = [instance("prefill") for _ in range(p)] + [instance("decode") for _ in range(n)]
        write_json(os.path.join(cfg_dir, f"pd_{p}p{n}d.json"), cluster(pd))
        write_json(os.path.join(cfg_dir, f"colocated_{p + n}.json"),
                   cluster([instance(None) for _ in range(p + n)]))

    with open(os.path.join(REPO, SOURCE_WORKLOAD)) as f:
        source = [json.loads(line) for line in f if line.strip()]
    # Separate generators, so lengths and arrivals do not depend on how many
    # requests the highest rate needs.
    len_rng, arr_rng = random.Random(SEED), random.Random(SEED + 1)
    horizon = max(RATES) * DURATION_S
    lengths, unit_arrivals, t = [], [], 0.0
    while True:
        t += arr_rng.expovariate(1.0)
        if t >= horizon:
            break
        unit_arrivals.append(t)
        lengths.append(len_rng.choice(source))
    counts = []
    for rate in RATES:
        n = 0
        with open(os.path.join(wl_dir, f"rate{rate:g}.jsonl"), "w") as f:
            for row, t in zip(lengths, unit_arrivals):
                if t / rate >= DURATION_S:
                    break
                f.write(json.dumps({
                    "input_toks": row["input_toks"],
                    "output_toks": row["output_toks"],
                    "arrival_time_ns": int(round(t / rate * 1e9)),
                }) + "\n")
                n += 1
        counts.append(n)

    layouts = ", ".join(f"{p}P+{n}D" for p, n in PD_LAYOUTS)
    print(f"{MODEL_KEY}: {2 * len(PD_LAYOUTS)} configs ({layouts}, each with colocated "
          f"at the same GPU count), {len(RATES)} workloads "
          f"({DURATION_S} s of arrivals: {min(counts)}-{max(counts)} requests) "
          f"under {os.path.relpath(HERE, REPO)}/")


if __name__ == "__main__":
    main()
