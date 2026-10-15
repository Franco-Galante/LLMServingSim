"""Generate the cluster configs and rate-scaled workloads for the P/D KV-transfer sweep.

Run from the repo root (stdlib only, works on the host or in the sim container):

    python3 experiments/pd_kv_sweep/prepare.py

Writes:
    experiments/pd_kv_sweep/configs/pd_bw<BW>.json     1 prefill + 1 decode, one per link class
    experiments/pd_kv_sweep/configs/colocated.json     2 colocated instances (baseline)
    experiments/pd_kv_sweep/workloads/sharegpt_rate<R>.jsonl
"""

import copy
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

SOURCE_WORKLOAD = os.path.join(REPO, "workloads", "sharegpt-llama-3.1-8b-300-sps10.jsonl")
SOURCE_RATE = 10.0  # req/s the source trace was generated at

# (label, link_bw GB/s, link_latency ns)
LINKS = [
    ("12.5", 12.5, 5000),   # 100 Gb/s network
    ("50", 50, 5000),       # 400 Gb/s network
    ("300", 300, 1000),     # NVLink, mid-range
    ("900", 900, 1000),     # NVLink, top-end
]
RATES = [10, 20, 40]

INSTANCE = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "hardware": "RTXPRO6000",
    # No npu_mem: GPU memory size and bandwidth come from the measured
    # profiler/perf/RTXPRO6000/hardware.yaml. The link stays explicit per
    # config, because it is what this sweep varies.
    "tp_size": 1,
}


def cluster(pd_types, link_bw, link_latency):
    instances = []
    for pd_type in pd_types:
        inst = copy.deepcopy(INSTANCE)
        inst["pd_type"] = pd_type
        instances.append(inst)
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


def main():
    cfg_dir = os.path.join(HERE, "configs")
    wl_dir = os.path.join(HERE, "workloads")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(wl_dir, exist_ok=True)

    for label, bw, lat in LINKS:
        write_json(os.path.join(cfg_dir, f"pd_bw{label}.json"),
                   cluster(["prefill", "decode"], bw, lat))
    # TP=1 colocated instances exchange nothing over the link, so one
    # baseline covers every bandwidth.
    write_json(os.path.join(cfg_dir, "colocated.json"),
               cluster([None, None], 16, 20000))

    with open(SOURCE_WORKLOAD) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for rate in RATES:
        scale = SOURCE_RATE / rate
        with open(os.path.join(wl_dir, f"sharegpt_rate{rate}.jsonl"), "w") as f:
            for row in rows:
                row = dict(row)
                row["arrival_time_ns"] = int(round(row["arrival_time_ns"] * scale))
                f.write(json.dumps(row) + "\n")

    print(f"wrote {len(LINKS) + 1} configs to {os.path.relpath(cfg_dir, REPO)}/")
    print(f"wrote {len(RATES)} workloads ({len(rows)} requests each) to {os.path.relpath(wl_dir, REPO)}/")


if __name__ == "__main__":
    main()
