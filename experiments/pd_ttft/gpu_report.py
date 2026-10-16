"""Per-GPU report for the TTFT-vs-load study, from the outputs the runs already write.

    python3 experiments/pd_ttft/gpu_report.py [outputs/pd_ttft_long]

Every run has 2 GPUs: colocated (GPU 0 and GPU 1, both prefill + decode) or
P/D (instance 0 = prefill GPU, instance 1 = decode GPU). For each GPU:

    wait        time a request waits before this GPU starts on it
                  colocated / P/D prefill: CSV queuing_delay (arrival -> first scheduled)
                  P/D decode: CSV ITL[0], the gap between the 1st token (made on the
                  prefill GPU) and the 2nd: the deferred-handoff wait plus one step
    service     colocated / P/D prefill: TTFT; colocated / P/D decode: later token gaps
    running, waiting, memory   per-second samples from the log ("Running Instance[i]")

Writes to <dir>/:
    gpu_report.csv                       one row per (model, system, rate, GPU)
    gpu_report.txt                       the same, readable
    plots/<model>_gpus_vs_rate.png       per-GPU wait / running / waiting / memory vs rate
    plots/gpu_timelines/<model>_rate<R>.png   running, waiting, memory over time per GPU

Compute and memory-bandwidth use, every GPU, see assets/compute_and_memory.md:
    busy %, mean step time, MFU and MBU (busy and wall), from the run's step log
    (<run>.steps.csv, written by --step-log), plus
    plots/<model>_mfu_mbu_vs_rate.png             MFU / MBU vs rate
    plots/mfu_mbu_timelines/<model>_rate<R>.png   per-second MFU / MBU per GPU
A run without a step log falls back to a reconstruction from the per-second log
and the per-request CSV, which works for P/D runs only (prefill GPU MFU, decode
GPU MBU); a warning names those runs.

Run-level counters (preemptions, deferred handoffs) are in summary.csv; in P/D
the prefill GPU's memory never fills, so its evictions all happen on the decode GPU.
"""

import csv
import glob
import json
import os
import re
import sys

import matplotlib
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from plot import GRID, INK, INK_2, MODEL_TITLES, SURFACE, pct, style  # noqa: E402

MEM_UTIL_PCT = 90.0   # usable share of GPU memory (npu_mem.mem_util default)

# MFU / MBU denominators. Peak memory bandwidth is the one the simulator used:
# the cluster config's npu_mem.mem_bw if set, else profiler/perf/<hw>/hardware.yaml
# (1597.6 GB/s for RTXPRO6000, the vendor figure).
# The simulator has no peak FLOP/s: this is the datasheet's BF16 tensor-core
# figure, 1000 TFLOPS with 2:4 sparsity, halved for dense math.
PEAK_TFLOPS = 500.0
BYTES = 2             # bf16 weights and KV cache
MIB = 1024 * 1024     # the log's "MB" are MiB
FALLBACK_WARNING = (
    "WARNING: {n} run(s) have no step log (<run>.steps.csv): their MFU/MBU are "
    "reconstructed from the per-second log, which works only for the "
    "disaggregated (P/D) runs (prefill GPU MFU, decode GPU MBU) and is less "
    "accurate. Colocated runs without a step log get none. Rerun them with "
    "run.sh to get step logs. See experiments/pd_ttft/assets/compute_and_memory.md.")

# (system, instance) -> label, color, marker. Colocated GPUs are symmetric
# (load-balanced), so vs-rate panels show their average as one series.
GPUS = {
    ("colocated", 0): ("Colocated GPU 0", "#2a78d6", "o"),
    ("colocated", 1): ("Colocated GPU 1", "#4a3aa7", "D"),
    ("pd_custom", 0): ("P/D prefill GPU", "#eb6834", "^"),
    ("pd_custom", 1): ("P/D decode GPU", "#1baf7a", "s"),
}
RATE_SERIES = [
    ("colocated", None, "Colocated (avg of its 2 GPUs)", "#2a78d6", "o"),
    ("pd_custom", 0, "P/D prefill GPU", "#eb6834", "^"),
    ("pd_custom", 1, "P/D decode GPU", "#1baf7a", "s"),
]

LOG_TICK = re.compile(r"^\[([\d.]+)s\] Avg prompt throughput: ([\d.]+) tokens/s, "
                      r"Avg generation throughput: ([\d.]+) tokens/s")
LOG_INST = re.compile(r"Running Instance\[(\d+)\]: (\d+) reqs, Waiting: (\d+) reqs"
                      r".*Memory Usage ([\d.]+) MB \(([\d.]+) % Used\)")


UTIL_KEYS = ("busy_pct", "step_ms_mean", "mfu_busy_pct", "mfu_wall_pct",
             "mbu_busy_pct", "mbu_wall_pct")


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def read_log(path):
    """{instance: {"t", "running", "waiting", "mem" (%), "mem_mb", "gen_tps"}: lists}

    gen_tps is the whole cluster's generated tokens/s over the second ending at t
    (the log has no per-instance throughput); each instance carries a copy.
    """
    series, t, gen = {}, None, None
    with open(path) as f:
        for line in f:
            m = LOG_TICK.match(line)
            if m:
                t, gen = float(m.group(1)), float(m.group(3))
                continue
            m = LOG_INST.search(line)
            if m and t is not None:
                s = series.setdefault(int(m.group(1)), {k: [] for k in (
                    "t", "running", "waiting", "mem", "mem_mb", "gen_tps")})
                s["t"].append(t)
                s["running"].append(int(m.group(2)))
                s["waiting"].append(int(m.group(3)))
                s["mem_mb"].append(float(m.group(4)))
                s["mem"].append(float(m.group(5)))
                s["gen_tps"].append(gen)
    return series


def model_shape(model_key):
    """Model dimensions and the peak memory bandwidth from the run's configs."""
    with open(os.path.join(HERE, "configs", model_key, "pd_custom.json")) as f:
        inst = json.load(f)["nodes"][0]["instances"][0]
    with open(os.path.join(REPO, "configs", "model", inst["model_name"] + ".json")) as f:
        c = json.load(f)
    h, L = c["hidden_size"], c["num_hidden_layers"]
    head_dim = c.get("head_dim", h // c["num_attention_heads"])
    q_dim = c["num_attention_heads"] * head_dim
    kv_dim = c["num_key_value_heads"] * head_dim
    # Matmul weights of the transformer blocks: q/k/v, o, gate+up, down.
    block_params = L * (h * (q_dim + 2 * kv_dim) + q_dim * h + 3 * h * c["intermediate_size"])
    mem_bw = inst.get("npu_mem", {}).get("mem_bw")
    if mem_bw is None:
        hw_yaml = os.path.join(REPO, "profiler", "perf", inst["hardware"], "hardware.yaml")
        with open(hw_yaml) as f:
            mem_bw = yaml.safe_load(f)["defaults"]["npu_mem"]["mem_bw"]["value"]
    # Every weight the GPU holds (and every step reads): blocks, norms, embedding,
    # lm_head. Matches the simulator's own weight figure (15316.5 MiB for
    # Llama-3.1-8B, 62488.8 MiB for Qwen3-32B).
    norms = L * 2 * h + h + (L * 2 * head_dim if c["model_type"].startswith("qwen3") else 0)
    embed = h * c["vocab_size"]
    lm_head = 0 if c.get("tie_word_embeddings") else embed
    return {"layers": L, "q_dim": q_dim, "block_params": block_params,
            "lm_head_params": h * c["vocab_size"], "peak_bw": mem_bw * 1e9,
            "tp": inst["tp_size"],
            "weight_bytes": (block_params + norms + embed + lm_head) * BYTES,
            "kv_bytes_per_token": 2 * L * kv_dim * BYTES}


def prefill_flops(n, shape):
    """Model FLOPs of one request's whole prefill (n prompt tokens, causal)."""
    linear = 2 * shape["block_params"] * n                 # every prompt token
    attention = 2 * shape["layers"] * shape["q_dim"] * n * n  # QK^T + AV, causal half
    lm_head = 2 * shape["lm_head_params"]                  # last token only
    return linear + attention + lm_head


def step_utilization(path, shape):
    """Per-GPU MFU / MBU from a step log (one row per completed batch).

    Per step: FLOPs = 2*P_blocks*tokens + 4*L*q_dim*attn_pairs + 2*h*V*sequences
              bytes = weights + kv_read_tokens * KV bytes/token
    Divided per GPU by the TP degree. Run values: busy = sum / (sum of step
    times x peak), wall = sum / (first start to last end x peak). The per-second
    series spreads each step's work over its own interval.
    """
    peak_f = PEAK_TFLOPS * 1e12 * shape["tp"]
    peak_b = shape["peak_bw"] * shape["tp"]
    per = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            per.setdefault(int(r["instance"]), []).append(r)
    out = {}
    for inst, steps in per.items():
        bins_f, bins_b = {}, {}
        tot_f = tot_b = busy = 0.0
        t0 = min(int(r["start_ns"]) for r in steps) / 1e9
        t1 = max(int(r["end_ns"]) for r in steps) / 1e9
        for r in steps:
            s, e = int(r["start_ns"]) / 1e9, int(r["end_ns"]) / 1e9
            tokens = int(r["prefill_tokens"]) + int(r["decode_tokens"])
            flops = 0.0 if r["dummy"] == "1" else (
                2 * shape["block_params"] * tokens
                + 4 * shape["layers"] * shape["q_dim"] * int(r["attn_pairs"])
                + 2 * shape["lm_head_params"] * int(r["sequences"]))
            nbytes = shape["weight_bytes"] + int(r["kv_read_tokens"]) * shape["kv_bytes_per_token"]
            tot_f += flops
            tot_b += nbytes
            busy += e - s
            k = int(s) + 1                         # second k covers (k-1, k]
            while k - 1 < e or k == int(s) + 1:
                share = (min(e, k) - max(s, k - 1)) / (e - s) if e > s else 1.0
                if share > 0:
                    bins_f[k] = bins_f.get(k, 0.0) + flops * share
                    bins_b[k] = bins_b.get(k, 0.0) + nbytes * share
                k += 1
        t = list(range(1, max(bins_f) + 1))
        out[inst] = {
            "t": t,
            "mfu": [100 * bins_f.get(k, 0.0) / peak_f for k in t],
            "mbu": [100 * bins_b.get(k, 0.0) / peak_b for k in t],
            "busy_pct": 100 * busy / (t1 - t0),
            "step_ms_mean": 1e3 * busy / len(steps),
            "mfu_busy_pct": 100 * tot_f / (busy * peak_f),
            "mfu_wall_pct": 100 * tot_f / ((t1 - t0) * peak_f),
            "mbu_busy_pct": 100 * tot_b / (busy * peak_b),
            "mbu_wall_pct": 100 * tot_b / ((t1 - t0) * peak_b),
        }
    return out


def pd_utilization(rows, log, shape):
    """Per-second MFU of the prefill GPU and MBU of the decode GPU (P/D runs only).

    MFU: each request's prefill FLOPs (CSV input length) are spread evenly over
    its time on the prefill GPU, first scheduled (arrival + queuing_delay) to
    first token (arrival + TTFT), and summed in 1-s bins.
    MBU: decode steps/s = generated tokens/s / running requests on the decode GPU
    (each step makes one token per running request); each step reads the weights
    plus the KV cache in use (decode GPU memory used - weights).
    Single seconds are rough (binning, sampled running count) and can exceed
    100%; the run means are the numbers to read.
    """
    bins = {}                                   # second k covers (k-1, k]
    span = [float("inf"), 0.0]
    for r in rows:
        arrival = int(r["arrival"]) / 1e9
        s = arrival + int(r["queuing_delay"]) / 1e9
        e = arrival + int(r["TTFT"]) / 1e9
        span = [min(span[0], s), max(span[1], e)]
        f = prefill_flops(int(r["input"]), shape)
        if e <= s:
            k = int(s) + 1
            bins[k] = bins.get(k, 0.0) + f
            continue
        k = int(s) + 1
        while k - 1 < e:
            overlap = min(e, k) - max(s, k - 1)
            if overlap > 0:
                bins[k] = bins.get(k, 0.0) + f * overlap / (e - s)
            k += 1
    peak = PEAK_TFLOPS * 1e12
    mfu_t = list(range(1, max(bins) + 1)) if bins else []
    mfu = [100 * bins.get(k, 0.0) / peak for k in mfu_t]
    mfu_mean = 100 * sum(bins.values()) / ((span[1] - span[0]) * peak) if bins else float("nan")

    pre, dec = log.get(0), log.get(1)
    mbu_t, mbu = [], []
    if dec and pre:
        # The idle prefill GPU's memory is the weights alone (no KV in use).
        weights = min(pre["mem_mb"]) * MIB
        prev = 0
        for t, n_now, mb, gen in zip(dec["t"], dec["running"], dec["mem_mb"], dec["gen_tps"]):
            # Tokens are counted over the whole second, running is a sample at
            # its end: average it with the previous sample to match the window.
            n = (prev + n_now) / 2
            prev = n_now
            steps_per_s = gen / n if n else 0.0
            kv = max(mb * MIB - weights, 0.0)
            mbu_t.append(t)
            mbu.append(100 * steps_per_s * (weights + kv) / shape["peak_bw"])
    busy = [i for i, n in enumerate(dec["running"]) if n] if dec else []
    mbu_busy = mbu[busy[0]:busy[-1] + 1] if busy else []
    # Same shape as step_utilization: the prefill GPU has only MFU (its run
    # value is over its busy span, like "wall"), the decode GPU only MBU (mean
    # over the seconds it had requests running, closest to "busy").
    return {0: {"t": mfu_t, "mfu": mfu, "mbu": None, "mfu_wall_pct": mfu_mean},
            1: {"t": mbu_t, "mfu": None, "mbu": mbu, "mbu_busy_pct": mean(mbu_busy)}}


def gpu_rows(model, system, rate, base, shape):
    rows = list(csv.DictReader(open(base + ".csv")))
    log = read_log(base + ".log")
    sec = lambda ns: int(ns) / 1e9
    out = []
    for inst in (0, 1):
        if system == "colocated":
            reqs = [r for r in rows if int(r["instance id"]) == inst]
            wait = [sec(r["queuing_delay"]) for r in reqs]
            service = [sec(r["TTFT"]) for r in reqs]          # prefill side
            gaps = [g / 1e9 for r in reqs for g in json.loads(r["ITL"])]
            role = "prefill+decode"
        elif inst == 0:
            reqs = rows
            wait = [sec(r["queuing_delay"]) for r in rows]
            service = [sec(r["TTFT"]) for r in rows]
            gaps = []
            role = "prefill"
        else:
            reqs = rows
            itl = [json.loads(r["ITL"]) for r in rows]
            wait = [x[0] / 1e9 for x in itl if x]               # handoff -> 2nd token
            service = []
            gaps = [g / 1e9 for x in itl for g in x[1:]]
            role = "decode"
        s = log.get(inst, {"running": [], "waiting": [], "mem": []})
        out.append({
            "model": model, "system": system, "rate": f"{rate:g}", "gpu": inst, "role": role,
            "requests": len(reqs),
            "wait_mean_s": mean(wait), "wait_p50_s": pct(wait, 50) if wait else "",
            "wait_p99_s": pct(wait, 99) if wait else "",
            "ttft_mean_s": mean(service) if service else "",
            "token_gap_mean_ms": mean(gaps) * 1e3 if gaps else "",
            "token_gap_p50_ms": pct(gaps, 50) * 1e3 if gaps else "",
            "running_mean": mean(s["running"]), "running_max": max(s["running"], default=""),
            "waiting_mean": mean(s["waiting"]), "waiting_max": max(s["waiting"], default=""),
            "mem_mean_pct": mean(s["mem"]), "mem_max_pct": max(s["mem"], default=""),
        })
    if os.path.exists(base + ".steps.csv"):
        util, source = step_utilization(base + ".steps.csv", shape), "step_log"
    elif system == "pd_custom":
        util, source = pd_utilization(rows, log, shape), "reconstructed"
    else:
        util, source = None, ""
    for inst, row in enumerate(out):
        u = (util or {}).get(inst, {})
        for key in UTIL_KEYS:
            row[key] = u.get(key, "")
        row["util_source"] = source if u else ""
    return out, log, util


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_reports(rows, out_dir):
    fields = list(rows[0].keys())
    with open(os.path.join(out_dir, "gpu_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(out_dir, "gpu_report.txt"), "w") as f:
        hdr = (f"{'model':<8}{'system':<11}{'rate':>5} {'GPU':<17}{'wait mean':>10}{'p99':>9}"
               f"{'TTFT':>8}{'tok gap':>9}{'running':>9}{'waiting':>9}{'mem %':>7}"
               f"{'busy %':>8}{'step ms':>9}{'MFU b/w %':>13}{'MBU b/w %':>13}")
        f.write("wait/TTFT in s, token gap in ms; running/waiting/mem are means of "
                "per-second samples.\nbusy %, step ms, MFU/MBU from the step log: b = over "
                "the GPU's step time, w = over the run span (see assets/compute_and_memory.md);\n"
                "'*' = reconstructed without a step log (P/D only, less accurate)\n"
                + hdr + "\n" + "-" * len(hdr) + "\n")
        for r in rows:
            gpu = f"{r['gpu']} {r['role']}"
            f.write(f"{r['model']:<8}{r['system']:<11}{r['rate']:>5} {gpu:<17}"
                    f"{fmt(r['wait_mean_s']):>10}{fmt(r['wait_p99_s']):>9}"
                    f"{fmt(r['ttft_mean_s']):>8}{fmt(r['token_gap_mean_ms'], 1):>9}"
                    f"{fmt(r['running_mean'], 1):>9}{fmt(r['waiting_mean'], 1):>9}"
                    f"{fmt(r['mem_mean_pct'], 1):>7}"
                    f"{fmt(r['busy_pct'], 1):>8}{fmt(r['step_ms_mean'], 1):>9}"
                    f"{pair(r, 'mfu'):>13}{pair(r, 'mbu'):>13}\n")


def pair(r, key):
    """'busy/wall' for the text report; '*' marks a reconstructed value."""
    b, w = r[f"{key}_busy_pct"], r[f"{key}_wall_pct"]
    if b == "" and w == "":
        return ""
    mark = "*" if r["util_source"] == "reconstructed" else ""
    return f"{fmt(b, 1) if b != '' else '-'}/{fmt(w, 1) if w != '' else '-'}{mark}"


def legend_top(fig, handles):
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1 - 0.4 / fig.get_figheight()))


def plot_vs_rate(model, rows, out_path):
    panels = [
        ("Wait before the GPU takes a request (s, log)", "wait_mean_s", True),
        ("Requests running (mean)", "running_mean", False),
        ("Requests waiting (mean)", "waiting_mean", False),
        ("Memory used (% of GPU, mean)", "mem_mean_pct", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.2), facecolor=SURFACE)
    for ax, (title, key, log) in zip(axes, panels):
        style(ax)
        if log:
            ax.set_yscale("log")
        for system, inst, label, color, marker in RATE_SERIES:
            pts = {}
            for r in rows:
                if r["model"] != model or r["system"] != system:
                    continue
                if inst is not None and r["gpu"] != inst:
                    continue
                pts.setdefault(float(r["rate"]), []).append(r[key])
            xs = sorted(pts)
            ys = [mean(pts[x]) for x in xs]
            if log:
                xs, ys = zip(*[(x, y) for x, y in zip(xs, ys) if y > 0]) if any(
                    y > 0 for y in ys) else ([], [])
            if xs:
                ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2)
        if key == "mem_mean_pct":
            ax.axhline(MEM_UTIL_PCT, color=INK_2, linewidth=1, linestyle="--", zorder=0)
            ax.text(ax.get_xlim()[0], MEM_UTIL_PCT, " usable limit", va="bottom",
                    fontsize=8, color=INK_2)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle(f"{MODEL_TITLES.get(model, model)} on RTXPRO6000: per-GPU view vs request rate",
                 fontsize=12, color=INK, y=1 - 0.08 / fig.get_figheight())
    legend_top(fig, [plt.Line2D([], [], color=c, marker=mk, linestyle="", markersize=7, label=l)
                     for _, _, l, c, mk in RATE_SERIES])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_timeline(model, rate, logs, out_path):
    """Columns: colocated, P/D. Rows: running, waiting, memory. One line per GPU."""
    metrics = [("running", "Requests running"), ("waiting", "Requests waiting"),
               ("mem", "Memory used (% of GPU)")]
    systems = [s for s in ("colocated", "pd_custom") if s in logs]
    fig, axes = plt.subplots(len(metrics), len(systems), figsize=(6.5 * len(systems), 7.6),
                             sharex="col", squeeze=False, facecolor=SURFACE)
    for col, system in enumerate(systems):
        for row, (key, title) in enumerate(metrics):
            ax = axes[row, col]
            style(ax)
            for inst in (0, 1):
                s = logs[system].get(inst)
                if not s:
                    continue
                label, color, _ = GPUS[(system, inst)]
                ax.plot(s["t"], s[key], color=color, linewidth=1.6, label=label)
            if key == "mem":
                ax.axhline(MEM_UTIL_PCT, color=INK_2, linewidth=1, linestyle="--", zorder=0)
            ax.set_ylim(bottom=0)
            name = "Colocated" if system == "colocated" else "P/D"
            ax.set_title(f"{name}: {title}", fontsize=10, color=INK, loc="left")
            if row == len(metrics) - 1:
                ax.set_xlabel("simulated time (s)", fontsize=9, color=INK_2)
    fig.suptitle(f"{MODEL_TITLES.get(model, model)} at {rate:g} req/s: each GPU over time",
                 fontsize=12, color=INK, y=1 - 0.08 / fig.get_figheight())
    legend_top(fig, [plt.Line2D([], [], color=c, linewidth=2, label=l)
                     for (sysname, _), (l, c, _) in GPUS.items() if sysname in systems])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_util_vs_rate(model, rows, out_path):
    """MFU and MBU vs rate per GPU role: solid = busy (over step time),
    dashed = wall (over the run span)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, key, title in zip(axes, ("mfu", "mbu"),
                              ("MFU (% of peak bf16 FLOP/s)", "MBU (% of peak memory bandwidth)")):
        style(ax)
        for system, inst, label, color, marker in RATE_SERIES:
            for kind, ls in (("busy", "-"), ("wall", "--")):
                pts = {}
                for r in rows:
                    if r["model"] != model or r["system"] != system:
                        continue
                    if inst is not None and r["gpu"] != inst:
                        continue
                    v = r[f"{key}_{kind}_pct"]
                    if v != "":
                        pts.setdefault(float(r["rate"]), []).append(v)
                if pts:
                    xs = sorted(pts)
                    ax.plot(xs, [mean(pts[x]) for x in xs], color=color, marker=marker,
                            markersize=5 if ls == "-" else 0, linewidth=2 if ls == "-" else 1.3,
                            linestyle=ls, alpha=1 if ls == "-" else 0.8)
        ax.set_ylim(0, 100)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle(f"{MODEL_TITLES.get(model, model)} on RTXPRO6000: compute and "
                 f"memory-bandwidth use per GPU vs request rate",
                 fontsize=12, color=INK, y=1 - 0.08 / fig.get_figheight())
    legend_top(fig, [plt.Line2D([], [], color=c, marker=mk, linestyle="", markersize=7, label=l)
                     for _, _, l, c, mk in RATE_SERIES]
               + [plt.Line2D([], [], color=INK_2, linewidth=2, label="busy (step time)"),
                  plt.Line2D([], [], color=INK_2, linewidth=1.3, linestyle="--",
                             label="wall (run span)")])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_util_timeline(model, rate, by_system, out_path):
    """Columns: colocated, P/D. Rows: MFU, MBU per second. One line per GPU."""
    systems = [s for s in ("colocated", "pd_custom") if by_system.get(s)]
    fig, axes = plt.subplots(2, len(systems), figsize=(6.5 * len(systems), 6.2),
                             sharex="col", squeeze=False, facecolor=SURFACE)
    for col, system in enumerate(systems):
        for row, (key, title) in enumerate((("mfu", "MFU (% of peak FLOP/s)"),
                                           ("mbu", "MBU (% of peak bandwidth)"))):
            ax = axes[row, col]
            style(ax)
            for inst in (0, 1):
                u = by_system[system].get(inst)
                if not u or u.get(key) is None:
                    continue
                label, color, _ = GPUS[(system, inst)]
                ax.plot(u["t"], u[key], color=color, linewidth=1.2, label=label)
            ax.axhline(100, color=INK_2, linewidth=1, linestyle=":", zorder=0)
            ax.set_ylim(bottom=0)
            name = "Colocated" if system == "colocated" else "P/D"
            ax.set_title(f"{name}: {title}", fontsize=10, color=INK, loc="left")
            if row == 1:
                ax.set_xlabel("simulated time (s)", fontsize=9, color=INK_2)
    fig.suptitle(f"{MODEL_TITLES.get(model, model)} at {rate:g} req/s: per-second MFU and "
                 f"MBU per GPU", fontsize=12, color=INK, y=1 - 0.08 / fig.get_figheight())
    legend_top(fig, [plt.Line2D([], [], color=c, linewidth=2, label=l)
                     for (sysname, _), (l, c, _) in GPUS.items() if sysname in systems])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/pd_ttft_long"
    rows, logs, utils, shapes = [], {}, {}, {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*", "*_rate*.csv"))):
        model = os.path.basename(os.path.dirname(path))
        m = re.match(r"(.+)_rate([0-9.]+)\.csv$", os.path.basename(path))
        if not m or m.group(1) not in ("colocated", "pd_custom"):
            continue
        system, rate = m.group(1), float(m.group(2))
        if model not in shapes:
            shapes[model] = model_shape(model)
        r, log, util = gpu_rows(model, system, rate, path[:-len(".csv")], shapes[model])
        rows += r
        logs.setdefault(model, {}).setdefault(rate, {})[system] = log
        if util:
            utils.setdefault(model, {}).setdefault(rate, {})[system] = util
    if not rows:
        print(f"no finished runs under {out_dir}")
        return
    fallback = {(r["model"], r["system"], r["rate"]) for r in rows
                if r["util_source"] != "step_log"}
    if fallback:
        print(FALLBACK_WARNING.format(n=len(fallback)), file=sys.stderr)
    rows.sort(key=lambda r: (r["model"], float(r["rate"]), r["system"], r["gpu"]))
    write_reports(rows, out_dir)

    plot_dir = os.path.join(out_dir, "plots")
    tl_dir = os.path.join(plot_dir, "gpu_timelines")
    os.makedirs(tl_dir, exist_ok=True)
    for model in sorted(logs):
        plot_vs_rate(model, rows, os.path.join(plot_dir, f"{model}_gpus_vs_rate.png"))
        for rate, by_system in sorted(logs[model].items()):
            plot_timeline(model, rate, by_system,
                          os.path.join(tl_dir, f"{model}_rate{rate:g}.png"))
    util_dir = os.path.join(plot_dir, "mfu_mbu_timelines")
    os.makedirs(util_dir, exist_ok=True)
    for model in sorted(utils):
        plot_util_vs_rate(model, rows, os.path.join(plot_dir, f"{model}_mfu_mbu_vs_rate.png"))
        for rate, by_system in sorted(utils[model].items()):
            plot_util_timeline(model, rate, by_system,
                               os.path.join(util_dir, f"{model}_rate{rate:g}.png"))
    print(f"wrote {os.path.join(out_dir, 'gpu_report.txt')} (+ .csv), "
          f"{plot_dir}/<model>_gpus_vs_rate.png, {tl_dir}/, "
          f"{plot_dir}/<model>_mfu_mbu_vs_rate.png and {util_dir}/")


if __name__ == "__main__":
    main()
