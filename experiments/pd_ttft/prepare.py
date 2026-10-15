"""Generate cluster configs and long workloads for the TTFT-vs-load study.

Run from the repo root (stdlib only, works on the host or in the sim container):

    python3 experiments/pd_ttft/prepare.py

For each model it writes, under experiments/pd_ttft/:
    configs/<model>/colocated.json    2 GPUs, both prefill+decode, vLLM default batching
    configs/<model>/pd_custom.json    1 prefill + 1 decode GPU, per-role batching
    workloads/<model>/rate<R>.jsonl   NUM_REQUESTS requests, Poisson arrivals at R req/s

The workloads are built from the model's 300-request ShareGPT trace: request
lengths (input, output) are resampled with replacement to NUM_REQUESTS, and
arrivals are a fresh Poisson process. Every rate uses the same requests in the
same order on the same unit-rate arrival sequence, scaled by 1/R, so only the
spacing changes between rates. Token ids are dropped: run.sh disables prefix
caching, and without it the simulator reads only the lengths.

Which run uses which config/workload is decided in run.sh; this file only
writes them. The rate lists here and in run.sh must match.
"""

import copy
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

NUM_REQUESTS = 1000
SEED = 42

MODELS = {
    "llama8b": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "workload": "workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl",
        "rates": [2, 4, 6, 8, 10, 12, 15, 20, 30, 40],
    },
    # About 4x the compute per token of Llama-3.1-8B and 2x the KV per token,
    # so it saturates at a much lower rate.
    "qwen32b": {
        "model_name": "Qwen/Qwen3-32B",
        "workload": "workloads/sharegpt-qwen3-32b-300-sps10.jsonl",
        "rates": [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10],
    },
}

# One link for every P/D config. The first sweep showed the layer-by-layer KV
# transfer is fully hidden behind prefill compute above ~3 GB/s, so the value
# does not change the results; 50 GB/s = a 400 Gb/s RDMA network.
LINK_BW = 50
LINK_LATENCY = 5000

# Per-role batching ("pd_custom"). The prefill token budget stays at 2048
# because both latency profiles were measured up to 2048 tokens per step: a
# larger budget would be linearly extrapolated, which badly underestimates
# attention at long prefill chunks. 256 decode sequences is the profiled max.
CUSTOM_BATCHING = {
    "prefill": {"max_num_seqs": 32, "max_num_batched_tokens": 2048},
    "decode": {"max_num_seqs": 256, "max_num_batched_tokens": 256},
}


def instance(model_name, pd_type, batching=None):
    # No npu_mem: GPU memory size and bandwidth come from the measured
    # profiler/perf/RTXPRO6000/hardware.yaml. The link is set explicitly in
    # cluster() (LINK_BW), since it describes the P/D network, not the card.
    inst = {
        "model_name": model_name,
        "hardware": "RTXPRO6000",
        "tp_size": 1,
        "pd_type": pd_type,
    }
    if batching:
        inst.update(copy.deepcopy(batching))
    return inst


def cluster(instances, link_bw=LINK_BW, link_latency=LINK_LATENCY):
    return {
        "num_nodes": 1,
        "link_bw": link_bw,
        "link_latency": link_latency,
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


def rate_label(rate):
    return f"{rate:g}"


def main():
    for key, spec in MODELS.items():
        cfg_dir = os.path.join(HERE, "configs", key)
        wl_dir = os.path.join(HERE, "workloads", key)
        os.makedirs(cfg_dir, exist_ok=True)
        os.makedirs(wl_dir, exist_ok=True)
        name = spec["model_name"]

        write_json(os.path.join(cfg_dir, "colocated.json"),
                   cluster([instance(name, None), instance(name, None)]))
        write_json(os.path.join(cfg_dir, "pd_custom.json"),
                   cluster([instance(name, "prefill", CUSTOM_BATCHING["prefill"]),
                            instance(name, "decode", CUSTOM_BATCHING["decode"])]))

        with open(os.path.join(REPO, spec["workload"])) as f:
            source = [json.loads(line) for line in f if line.strip()]
        rng = random.Random(SEED)
        lengths = [rng.choice(source) for _ in range(NUM_REQUESTS)]
        unit_arrivals, t = [], 0.0          # seconds, at 1 req/s
        for _ in range(NUM_REQUESTS):
            t += rng.expovariate(1.0)
            unit_arrivals.append(t)
        for rate in spec["rates"]:
            path = os.path.join(wl_dir, f"rate{rate_label(rate)}.jsonl")
            with open(path, "w") as f:
                for row, t in zip(lengths, unit_arrivals):
                    f.write(json.dumps({
                        "input_toks": row["input_toks"],
                        "output_toks": row["output_toks"],
                        "arrival_time_ns": int(round(t / rate * 1e9)),
                    }) + "\n")

        print(f"{key}: 2 configs, {len(spec['rates'])} workloads "
              f"({NUM_REQUESTS} requests each) under {os.path.relpath(HERE, REPO)}/")


if __name__ == "__main__":
    main()
