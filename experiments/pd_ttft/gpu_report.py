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

P/D (disaggregated) runs only, see assets/compute_and_memory.md:
    MFU of the prefill GPU and MBU of the decode GPU, per second and per run
    (mfu_* / mbu_* columns), plus
    plots/<model>_mfu_mbu_vs_rate.png         mean and peak 1-s MFU / MBU vs rate
    plots/mfu_mbu_timelines/<model>_rate<R>.png   MFU / MBU over time
Colocated runs get no MFU/MBU: their GPUs mix prefill and decode, and the
outputs cannot tell which tokens ran on which GPU.

Not available without a per-step log: GPU busy %, step times, batch sizes.
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
PD_ONLY_WARNING = (
    "WARNING: MFU and MBU are computed only for the disaggregated (P/D) runs. "
    "In a colocated run each GPU runs prefill and decode together, and the "
    "simulator outputs do not say which tokens ran on which GPU, so colocated "
    "runs get no MFU/MBU. See experiments/pd_ttft/assets/compute_and_memory.md.")

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
    return {"layers": L, "q_dim": q_dim, "block_params": block_params,
            "lm_head_params": h * c["vocab_size"], "peak_bw": mem_bw * 1e9}


def prefill_flops(n, shape):
    """Model FLOPs of one request's whole prefill (n prompt tokens, causal)."""
    linear = 2 * shape["block_params"] * n                 # every prompt token
    attention = 2 * shape["layers"] * shape["q_dim"] * n * n  # QK^T + AV, causal half
    lm_head = 2 * shape["lm_head_params"]                  # last token only
    return linear + attention + lm_head


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
    return {"mfu_t": mfu_t, "mfu": mfu, "mbu_t": mbu_t, "mbu": mbu,
            "mfu_mean_pct": mfu_mean, "mbu_mean_pct": mean(mbu_busy)}


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
            "mfu_mean_pct": "", "mbu_mean_pct": "",
        })
    util = None
    if system == "pd_custom":
        util = pd_utilization(rows, log, shape)
        out[0]["mfu_mean_pct"] = util["mfu_mean_pct"]   # prefill GPU
        out[1]["mbu_mean_pct"] = util["mbu_mean_pct"]   # decode GPU
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
               f"{'MFU %':>7}{'MBU %':>7}")
        f.write("wait/TTFT in s, token gap in ms; running/waiting/mem are means of "
                "per-second samples; MFU (prefill GPU) / MBU (decode GPU) are run means,\n"
                "P/D only (see assets/compute_and_memory.md)\n"
                + hdr + "\n" + "-" * len(hdr) + "\n")
        for r in rows:
            gpu = f"{r['gpu']} {r['role']}"
            f.write(f"{r['model']:<8}{r['system']:<11}{r['rate']:>5} {gpu:<17}"
                    f"{fmt(r['wait_mean_s']):>10}{fmt(r['wait_p99_s']):>9}"
                    f"{fmt(r['ttft_mean_s']):>8}{fmt(r['token_gap_mean_ms'], 1):>9}"
                    f"{fmt(r['running_mean'], 1):>9}{fmt(r['waiting_mean'], 1):>9}"
                    f"{fmt(r['mem_mean_pct'], 1):>7}"
                    f"{fmt(r['mfu_mean_pct'], 1):>7}{fmt(r['mbu_mean_pct'], 1):>7}\n")


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
    """P/D only: prefill GPU MFU and decode GPU MBU vs rate (run means)."""
    panels = [(0, "mfu", "Prefill GPU: MFU (% of peak FLOP/s)", "#eb6834"),
              (1, "mbu", "Decode GPU: MBU (% of peak bandwidth)", "#1baf7a")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, (gpu, key, title, color) in zip(axes, panels):
        style(ax)
        pts = sorted((float(r["rate"]), r[f"{key}_mean_pct"])
                     for r in rows if r["model"] == model and r["system"] == "pd_custom"
                     and r["gpu"] == gpu)
        if pts:
            xs, means = zip(*pts)
            ax.plot(xs, means, color=color, marker="o", markersize=6, linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle(f"{MODEL_TITLES.get(model, model)}, P/D on RTXPRO6000: compute and "
                 f"memory-bandwidth use vs request rate",
                 fontsize=12, color=INK, y=1 - 0.08 / fig.get_figheight())
    fig.text(0.5, 1 - 0.5 / fig.get_figheight(), "mean over each GPU's busy span",
             ha="center", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_util_timeline(model, rate, util, out_path):
    """P/D only: prefill GPU MFU and decode GPU MBU per second."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor=SURFACE)
    for ax, (t, y, title, color) in zip(axes, [
            (util["mfu_t"], util["mfu"], "Prefill GPU: MFU (% of peak FLOP/s)", "#eb6834"),
            (util["mbu_t"], util["mbu"], "Decode GPU: MBU (% of peak bandwidth)", "#1baf7a")]):
        style(ax)
        ax.plot(t, y, color=color, linewidth=1.4)
        ax.axhline(100, color=INK_2, linewidth=1, linestyle=":", zorder=0)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[-1].set_xlabel("simulated time (s)", fontsize=9, color=INK_2)
    fig.suptitle(f"{MODEL_TITLES.get(model, model)}, P/D at {rate:g} req/s: "
                 f"per-second MFU and MBU", fontsize=12, color=INK,
                 y=1 - 0.08 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/pd_ttft_long"
    print(PD_ONLY_WARNING, file=sys.stderr)
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
            utils.setdefault(model, {})[rate] = util
    if not rows:
        print(f"no finished runs under {out_dir}")
        return
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
        for rate, util in sorted(utils[model].items()):
            plot_util_timeline(model, rate, util,
                               os.path.join(util_dir, f"{model}_rate{rate:g}.png"))
    print(f"wrote {os.path.join(out_dir, 'gpu_report.txt')} (+ .csv), "
          f"{plot_dir}/<model>_gpus_vs_rate.png, {tl_dir}/, "
          f"{plot_dir}/<model>_mfu_mbu_vs_rate.png and {util_dir}/ (P/D only)")


if __name__ == "__main__":
    main()
