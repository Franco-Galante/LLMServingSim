"""Plot the TTFT-vs-load study.

    python3 experiments/pd_ttft/plot.py [outputs/pd_ttft_long]

Per model, writes to <dir>/plots/:
    <model>_ttft_timeline.png  TTFT of every request vs its arrival time,
                               one panel per request rate
    <model>_vs_rate.png        TTFT mean / p50 / p99 and requests served per second vs rate
    <model>_kv_pressure.png    preemptions and deferred P/D handoffs vs rate
and <dir>/summary.csv with one row per (model, system, rate).
"""

import csv
import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_TITLES = {
    "llama8b": "Llama-3.1-8B",
    "qwen32b": "Qwen3-32B",
}
# Fixed order and identity: categorical slots 1-2 of the dataviz reference
# palette, plus a distinct marker each so identity never rests on color alone.
SYSTEMS = [
    ("colocated", "Colocated (2 GPUs, default batching)", "#2a78d6", "o"),
    ("pd_custom", "P/D 1 prefill + 1 decode GPU (decode cap 256)", "#eb6834", "^"),
]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3dd"
SURFACE = "#fcfcfb"


def pct(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def throughput(run):
    """Completed requests per second, first arrival to last completion.

    Includes the drain of the last long outputs, so it reads below the offered
    rate even when the system keeps up.
    """
    return len(run["end_s"]) / (max(run["end_s"]) - min(run["arrival_s"]))


# Run-level KV-pressure counters, from the summary the simulator prints
# (a line is only printed when its count is non-zero).
LOG_COUNTERS = {
    "preemptions": "Preemptions:",
    "recomputed_tokens": "Recomputed prompt tokens (preemption):",
    "deferred_handoffs": "Deferred P/D handoffs (decode KV full):",
}


def read_counters(log_path):
    counts = {key: 0 for key in LOG_COUNTERS}
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                for key, label in LOG_COUNTERS.items():
                    if line.startswith(label):
                        counts[key] = int(line.split()[-1])
    return counts


def load_run(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {
        "arrival_s": [int(r["arrival"]) / 1e9 for r in rows],
        "ttft_s": [int(r["TTFT"]) / 1e9 for r in rows],
        "tpot_ms": [int(r["TPOT"]) / 1e6 for r in rows],
        "end_s": [int(r["end_time"]) / 1e9 for r in rows],
        **read_counters(path[:-len(".csv")] + ".log"),
    }


def collect(out_dir):
    """{model: {rate: {system: run-dict or 'failed'}}}"""
    data = {}
    for path in glob.glob(os.path.join(out_dir, "*", "*_rate*.*")):
        model = os.path.basename(os.path.dirname(path))
        m = re.match(r"(.+)_rate([0-9.]+)\.(csv|failed)$", os.path.basename(path))
        if not m:
            continue
        system, rate, kind = m.group(1), float(m.group(2)), m.group(3)
        slot = data.setdefault(model, {}).setdefault(rate, {})
        if kind == "csv":
            slot[system] = load_run(path)
        else:
            slot.setdefault(system, "failed")
    return data


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def legend(fig, failed_any, y):
    """Legend just under the title at figure-height fraction y; failure note as a caption."""
    handles = [plt.Line2D([], [], color=c, marker=mk, linestyle="", markersize=7, label=lbl)
               for _, lbl, c, mk in SYSTEMS]
    fig.legend(handles=handles, loc="upper center", ncol=len(SYSTEMS), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, y))
    if failed_any:
        fig.text(0.01, 0.01, "A missing point or series means that run failed; see its .log "
                 "in the results folder.", ha="left", va="bottom", fontsize=8, color=INK_2)


def plot_timeline(model, runs, out_path):
    rates = sorted(runs)
    ncols = 5 if len(rates) > 6 else 3
    nrows = math.ceil(len(rates) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows + 1.0),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    failed_any = False
    for ax, rate in zip(axes.flat, rates):
        style(ax)
        ax.set_yscale("log")
        failed = []
        for key, label, color, marker in SYSTEMS:
            run = runs[rate].get(key)
            if run is None:
                continue
            if run == "failed":
                failed.append(label.split(" (")[0])
                failed_any = True
                continue
            ax.scatter(run["arrival_s"], run["ttft_s"], s=9, color=color, marker=marker,
                       alpha=0.75, linewidths=0)
        ax.set_title(f"{rate:g} req/s", fontsize=9, color=INK, loc="left")
        if failed:
            ax.text(0.98, 0.04, "failed: " + ", ".join(failed), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=INK_2)
    for ax in list(axes.flat)[len(rates):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("TTFT (s, log)", fontsize=9, color=INK_2)
    for ax in axes[-1, :]:
        ax.set_xlabel("arrival time (s)", fontsize=9, color=INK_2)
    fig.suptitle(f"{MODEL_TITLES.get(model, model)} on RTXPRO6000: TTFT of each request vs its arrival",
                 fontsize=12, color=INK, x=0.5, y=1 - 0.08 / fig.get_figheight())
    top = 1 - 0.65 / fig.get_figheight()      # leave ~0.65 in for title + legend
    legend(fig, failed_any, y=1 - 0.4 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0.03 if failed_any else 0, 1, top))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_vs_rate(model, runs, out_path):
    metrics = [
        ("TTFT mean (s, log)", lambda r: sum(r["ttft_s"]) / len(r["ttft_s"]), True),
        ("TTFT p50 (s, log)", lambda r: pct(r["ttft_s"], 50), True),
        ("TTFT p99 (s, log)", lambda r: pct(r["ttft_s"], 99), True),
        ("Requests served per second", throughput, False),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4.2), facecolor=SURFACE)
    failed_any = False
    for ax, (ylabel, fn, log) in zip(axes, metrics):
        style(ax)
        if log:
            ax.set_yscale("log")
        for key, label, color, marker in SYSTEMS:
            xs, ys = [], []
            for rate in sorted(runs):
                run = runs[rate].get(key)
                if run == "failed":
                    failed_any = True
                if isinstance(run, dict):
                    xs.append(rate)
                    ys.append(fn(run))
            if xs:
                ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(ylabel, fontsize=10, color=INK, loc="left")
        if fn is throughput:
            # y stops a bit above the best served rate; the reference line runs
            # out of the panel where a setup stops keeping up.
            served = [throughput(r) for v in runs.values() for r in v.values()
                      if isinstance(r, dict)]
            top = max(served) * 1.3 if served else max(runs)
            ax.plot([0, top], [0, top], color=INK_2, linewidth=1, linestyle="--", zorder=0)
            ax.text(top * 1.04, top * 0.97, "served = offered", ha="left", va="top",
                    fontsize=8, color=INK_2)
            ax.set_xlim(left=0)
            ax.set_ylim(0, top)
    fig.suptitle(f"{MODEL_TITLES.get(model, model)} on RTXPRO6000: latency vs request rate",
                 fontsize=12, color=INK, y=1 - 0.08 / fig.get_figheight())
    top = 1 - 0.65 / fig.get_figheight()
    legend(fig, failed_any, y=1 - 0.4 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0.04 if failed_any else 0, 1, top))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_kv_pressure(model, runs, out_path):
    """Evictions (preemptions) and deferred P/D handoffs vs request rate."""
    panels = [
        ("Preemptions (evicted running requests)", "preemptions"),
        ("P/D handoffs deferred (decode KV full)", "deferred_handoffs"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.2),
                             facecolor=SURFACE)
    failed_any = False
    for ax, (title, key) in zip(axes, panels):
        style(ax)
        for system, label, color, marker in SYSTEMS:
            if key == "deferred_handoffs" and system == "colocated":
                continue  # no handoffs without P/D
            xs, ys = [], []
            for rate in sorted(runs):
                run = runs[rate].get(system)
                if run == "failed":
                    failed_any = True
                if isinstance(run, dict):
                    xs.append(rate)
                    ys.append(run[key])
            if xs:
                ax.plot(xs, ys, color=color, marker=marker, markersize=6, linewidth=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(f"{title}, per run", fontsize=10, color=INK, loc="left")
    fig.suptitle(f"{MODEL_TITLES.get(model, model)} on RTXPRO6000: KV-cache pressure",
                 fontsize=12, color=INK, y=1 - 0.08 / fig.get_figheight())
    top = 1 - 0.65 / fig.get_figheight()
    legend(fig, failed_any, y=1 - 0.4 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0.04 if failed_any else 0, 1, top))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_summary(data, path):
    fields = ["model", "system", "rate", "status", "requests", "ttft_mean_s", "ttft_p50_s",
              "ttft_p99_s", "tpot_mean_ms", "tpot_p99_ms", "served_req_per_s",
              *LOG_COUNTERS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model in sorted(data):
            for rate in sorted(data[model]):
                for key, *_ in SYSTEMS:
                    run = data[model][rate].get(key)
                    if run is None:
                        continue
                    row = {"model": model, "system": key, "rate": f"{rate:g}"}
                    if run == "failed":
                        row["status"] = "failed"
                    else:
                        row.update(status="ok", requests=len(run["ttft_s"]),
                                   ttft_mean_s=round(sum(run["ttft_s"]) / len(run["ttft_s"]), 4),
                                   ttft_p50_s=round(pct(run["ttft_s"], 50), 4),
                                   ttft_p99_s=round(pct(run["ttft_s"], 99), 4),
                                   tpot_mean_ms=round(sum(run["tpot_ms"]) / len(run["tpot_ms"]), 3),
                                   served_req_per_s=round(throughput(run), 3),
                                   tpot_p99_ms=round(pct(run["tpot_ms"], 99), 3),
                                   **{key: run[key] for key in LOG_COUNTERS})
                    w.writerow(row)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/pd_ttft_long"
    data = collect(out_dir)
    if not data:
        print(f"no results under {out_dir}")
        return
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    for model, runs in sorted(data.items()):
        plot_timeline(model, runs, os.path.join(plot_dir, f"{model}_ttft_timeline.png"))
        plot_vs_rate(model, runs, os.path.join(plot_dir, f"{model}_vs_rate.png"))
        plot_kv_pressure(model, runs, os.path.join(plot_dir, f"{model}_kv_pressure.png"))
    write_summary(data, os.path.join(out_dir, "summary.csv"))
    print(f"wrote plots to {plot_dir}/ and {os.path.join(out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
