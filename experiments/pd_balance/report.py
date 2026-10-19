"""Plots and per-GPU report for the P/D balance study (Qwen3-32B, TP1).

    python3 experiments/pd_balance/report.py [outputs/pd_balance]

For every (P, N) in prepare.PD_LAYOUTS it compares, at equal GPU count G = P + N
(same requests and arrival times at every rate):

    pd_<P>p<N>d    P prefill + N decode GPUs
    colocated_<G>  G GPUs, each prefill + decode

Roles and TP come from each run's cluster config. Busy %, MFU and MBU come from
the step log (<run>.steps.csv); see experiments/docs/compute_and_memory.md.

Steady state only: every workload spans DURATION_S of arrivals (prepare.py),
and every run value covers the window [WARMUP_S, DURATION_S] -- requests that
arrive in it (TTFT, TPOT, delay, queue), completions in it (served), and
per-second samples and step time in it (busy %, MFU/MBU, running, waiting,
memory). Preemptions, recomputed tokens and deferred handoffs are whole-run
totals per 1000 requests.

Goodput = the highest tested rate whose run meets the latency target
(TTFT p99 <= SLO_TTFT_P99_S and TPOT p99 <= SLO_TPOT_P99_MS). Within one
comparison both layouts have G GPUs, so throughput is shown in total; per-GPU
values only appear in the summary, where G differs.

Writes to <dir>/:
    summary.csv, gpu_report.csv / .txt     one row per (layout, rate) / (layout, rate, GPU)
    summary.md                             every metric per (layout, rate), one table per comparison
    plots/G<G>_<P>p<N>d_vs_colocated<G>/   one folder per comparison:
        latency_throughput.png             TTFT p50/mean/p99, TPOT mean/p99, total served
        delay.png                          end-to-end delay (arrival -> last token) p50/mean/p99,
                                           and mean delay per output token
        gpu_load.png                       per role: busy %, MFU, MBU, memory, running, waiting
        kv_pressure.png                    preemptions, recomputed tokens, deferred handoffs
        ttft_timeline.png                  TTFT of each request vs its arrival, per rate
        timelines/<layout>_rate<R>.png     each GPU over time
    plots/summary/
        capacity.png                       vs G: max served, goodput, goodput per GPU
        pd_balance.png                     P/D layouts: which side saturates first
        delay.png                          end-to-end delay p50/p99 vs rate, every layout
"""

import csv
import json
import math
import os
import re
import sys

import matplotlib
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from prepare import DURATION_S, PD_LAYOUTS, WARMUP_S, MODEL_KEY as MODEL  # noqa: E402

MODEL_TITLE = "Qwen3-32B"

# Latency target for goodput.
SLO_TTFT_P99_S = 2.0
SLO_TPOT_P99_MS = 100.0

# Peak bf16 FLOP/s (datasheet 1000 TFLOPS with 2:4 sparsity, halved for dense);
# not in the simulator, so every MFU scales with it.
PEAK_TFLOPS = 500.0
BYTES = 2
MEM_UTIL_PCT = 90.0

PD_COLOR, CO_COLOR = "#7b3fbf", "#2a78d6"
ROLE_COLORS = {"prefill": "#eb6834", "decode": "#1baf7a", "colocated": "#2a78d6"}
PREFILL_SHADES = ["#eb6834", "#b8461a", "#f5a27a"]
DECODE_SHADES = ["#0b6e4a", "#15946a", "#1baf7a", "#52c495", "#86d6b3"]
COLOCATED_SHADES = ["#123f7a", "#1c5aa8", "#2a78d6", "#5694e0", "#7fb0ea", "#a6c9f1"]
# Light to dark by GPU count, for plots that show every layout at once.
PD_RAMP = ["#c9aef0", "#9c6bd8", "#7b3fbf", "#4f2187", "#2a0f52"]
CO_RAMP = ["#a6c9f1", "#5694e0", "#2a78d6", "#123f7a", "#071f40"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"

LOG_TICK = re.compile(r"^\[([\d.]+)s\] Avg prompt throughput")
LOG_INST = re.compile(r"Running Instance\[(\d+)\]: (\d+) reqs, Waiting: (\d+) reqs"
                      r".*Memory Usage ([\d.]+) MB \(([\d.]+) % Used\)")
LOG_COUNTERS = {
    "preemptions": "Preemptions:",
    "recomputed_tokens": "Recomputed prompt tokens (preemption):",
    "deferred_handoffs": "Deferred P/D handoffs (decode KV full):",
}


# ----------------------------------------------------------------- helpers

def mean(v):
    return sum(v) / len(v) if v else float("nan")


def pct(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def plain_log_ticks(ax):
    """Label a log y axis with plain numbers (40, 50, 100) instead of 4x10^1."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="y", which="minor", labelsize=7, colors=INK_2)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)


def legend_top(fig, handles, ncol=None):
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles), frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1 - 0.4 / fig.get_figheight()))


def finish(fig, out_path, dpi=150, top_in=0.7):
    fig.tight_layout(rect=(0, 0, 1, 1 - top_in / fig.get_figheight()))
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def mark_window(ax):
    """Grey out what the run values leave out: warm-up and time after the last arrival."""
    ax.axvspan(0, WARMUP_S, color=GRID, alpha=0.5, linewidth=0, zorder=0)
    ax.axvline(DURATION_S, color=INK_2, linewidth=1, linestyle=":", zorder=0)


def suptitle(fig, text):
    fig.suptitle(text, fontsize=12, color=INK, y=1 - 0.08 / fig.get_figheight())


# ------------------------------------------------------------ model shape

def model_shape(config):
    inst = config["nodes"][0]["instances"][0]
    with open(os.path.join(REPO, "configs", "model", inst["model_name"] + ".json")) as f:
        c = json.load(f)
    h, L = c["hidden_size"], c["num_hidden_layers"]
    head_dim = c.get("head_dim", h // c["num_attention_heads"])
    q_dim = c["num_attention_heads"] * head_dim
    kv_dim = c["num_key_value_heads"] * head_dim
    block_params = L * (h * (q_dim + 2 * kv_dim) + q_dim * h + 3 * h * c["intermediate_size"])
    norms = L * 2 * h + h + (L * 2 * head_dim if c["model_type"].startswith("qwen3") else 0)
    embed = h * c["vocab_size"]
    lm_head = 0 if c.get("tie_word_embeddings") else embed
    mem_bw = inst.get("npu_mem", {}).get("mem_bw")
    if mem_bw is None:
        with open(os.path.join(REPO, "profiler", "perf", inst["hardware"], "hardware.yaml")) as f:
            mem_bw = yaml.safe_load(f)["defaults"]["npu_mem"]["mem_bw"]["value"]
    return {"layers": L, "q_dim": q_dim, "block_params": block_params,
            "lm_head_params": embed, "peak_bw": mem_bw * 1e9,
            "weight_bytes": (block_params + norms + embed + lm_head) * BYTES,
            "kv_bytes_per_token": 2 * L * kv_dim * BYTES}


# --------------------------------------------------------------- loading

def read_counters(path):
    counts = {k: 0 for k in LOG_COUNTERS}
    with open(path) as f:
        for line in f:
            for key, label in LOG_COUNTERS.items():
                if line.startswith(label):
                    counts[key] = int(line.split()[-1])
    return counts


def read_log(path):
    """{instance: {"t", "running", "waiting", "mem"}} from the per-second log."""
    series, t = {}, None
    with open(path) as f:
        for line in f:
            m = LOG_TICK.match(line)
            if m:
                t = float(m.group(1))
                continue
            m = LOG_INST.search(line)
            if m and t is not None:
                s = series.setdefault(int(m.group(1)), {k: [] for k in ("t", "running", "waiting", "mem")})
                s["t"].append(t)
                s["running"].append(int(m.group(2)))
                s["waiting"].append(int(m.group(3)))
                s["mem"].append(float(m.group(5)))
    return series


def step_utilization(path, shape, tps, w0, w1):
    """Per instance: busy %, step time, MFU/MBU (busy and wall) over the window
    [w0, w1] s (steps cut at its edges count pro rata), per-second series."""
    per = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            per.setdefault(int(r["instance"]), []).append(r)
    out = {}
    for inst, steps in per.items():
        tp = tps[inst]
        peak_f, peak_b = PEAK_TFLOPS * 1e12 * tp, shape["peak_bw"] * tp
        bins_f, bins_b = {}, {}
        tot_f = tot_b = busy = 0.0
        n_steps = recompute = 0
        for r in steps:
            s, e = int(r["start_ns"]) / 1e9, int(r["end_ns"]) / 1e9
            tokens = int(r["prefill_tokens"]) + int(r["decode_tokens"])
            flops = 0.0 if r["dummy"] == "1" else (
                2 * shape["block_params"] * tokens
                + 4 * shape["layers"] * shape["q_dim"] * int(r["attn_pairs"])
                + 2 * shape["lm_head_params"] * int(r["sequences"]))
            nbytes = shape["weight_bytes"] + int(r["kv_read_tokens"]) * shape["kv_bytes_per_token"]
            inside = (min(e, w1) - max(s, w0)) / (e - s) if e > s else float(w0 <= s < w1)
            if inside > 0:
                tot_f += flops * inside
                tot_b += nbytes * inside
                busy += (e - s) * inside
                n_steps += inside
                recompute += int(r["recompute_tokens"]) * inside
            k = int(s) + 1                                 # second k covers (k-1, k]
            while k - 1 < e or k == int(s) + 1:
                share = (min(e, k) - max(s, k - 1)) / (e - s) if e > s else 1.0
                if share > 0:
                    bins_f[k] = bins_f.get(k, 0.0) + flops * share
                    bins_b[k] = bins_b.get(k, 0.0) + nbytes * share
                k += 1
        ts = list(range(1, max(bins_f) + 1))
        out[inst] = {
            "t": ts,
            "mfu": [100 * bins_f.get(k, 0.0) / peak_f for k in ts],
            "mbu": [100 * bins_b.get(k, 0.0) / peak_b for k in ts],
            "busy_pct": 100 * busy / (w1 - w0),
            "step_ms_mean": 1e3 * busy / n_steps if n_steps else 0.0,
            "mfu_busy_pct": 100 * tot_f / (busy * peak_f) if busy else 0.0,
            "mfu_wall_pct": 100 * tot_f / ((w1 - w0) * peak_f),
            "mbu_busy_pct": 100 * tot_b / (busy * peak_b) if busy else 0.0,
            "mbu_wall_pct": 100 * tot_b / ((w1 - w0) * peak_b),
            "recompute_tokens": round(recompute),
        }
    return out


def window_log(log, w0, w1):
    """The per-second samples with w0 < t <= w1."""
    out = {}
    for inst, s in log.items():
        keep = [i for i, t in enumerate(s["t"]) if w0 < t <= w1]
        out[inst] = {k: [v[i] for i in keep] for k, v in s.items()}
    return out


def load_run(base, config, shape, gpus):
    with open(base + ".csv") as f:
        all_rows = list(csv.DictReader(f))
    # Steady state: requests arriving in the window (warm-up and the drain after
    # the last arrival excluded).
    w0, w1 = WARMUP_S, DURATION_S
    rows = [r for r in all_rows if w0 <= int(r["arrival"]) / 1e9 < w1]
    instances = config["nodes"][0]["instances"]
    tps = [inst.get("tp_size", 1) for inst in instances]
    run = {
        "arrival_s": [int(r["arrival"]) / 1e9 for r in rows],
        "ttft_s": [int(r["TTFT"]) / 1e9 for r in rows],
        "tpot_ms": [int(r["TPOT"]) / 1e6 for r in rows],
        "queue_s": [int(r["queuing_delay"]) / 1e9 for r in rows],
        "end_s": [int(r["end_time"]) / 1e9 for r in rows],
        "delay_s": [int(r["latency"]) / 1e9 for r in rows],
        "delay_per_tok_ms": [int(r["latency"]) / 1e6 / int(r["output"]) for r in rows],
        "roles": [inst.get("pd_type") or "colocated" for inst in instances],
        "gpus": gpus,
        "log": read_log(base + ".log"),
        "util": (step_utilization(base + ".steps.csv", shape, tps, w0, w1)
                 if os.path.exists(base + ".steps.csv") else {}),
        "total_requests": len(all_rows),
    }
    run["log_win"] = window_log(run["log"], w0, w1)
    # Run totals, per 1000 requests: comparable across rates with different counts.
    for k, v in read_counters(base + ".log").items():
        run[k] = 1000 * v / len(all_rows)
    # Throughput: completions inside the window, whatever their arrival.
    run["served"] = sum(1 for r in all_rows if w0 <= int(r["end_time"]) / 1e9 < w1) / (w1 - w0)
    run["ttft_p50"] = pct(run["ttft_s"], 50)
    run["ttft_mean"] = mean(run["ttft_s"])
    run["ttft_p99"] = pct(run["ttft_s"], 99)
    run["tpot_mean"] = mean(run["tpot_ms"])
    run["delay_p50"] = pct(run["delay_s"], 50)
    run["delay_mean"] = mean(run["delay_s"])
    run["delay_p99"] = pct(run["delay_s"], 99)
    run["delay_per_tok_mean"] = mean(run["delay_per_tok_ms"])
    run["queue_mean"] = mean(run["queue_s"])
    run["tpot_p99"] = pct(run["tpot_ms"], 99)
    run["meets_slo"] = run["ttft_p99"] <= SLO_TTFT_P99_S and run["tpot_p99"] <= SLO_TPOT_P99_MS
    return run


def load_layout(out_dir, key, family, gpus, label, shape_cache):
    cfg_path = os.path.join(HERE, "configs", MODEL, f"{key}.json")
    folder = os.path.join(out_dir, MODEL)
    if not os.path.exists(cfg_path) or not os.path.isdir(folder):
        return None
    with open(cfg_path) as f:
        config = json.load(f)
    shape = shape_cache.setdefault("shape", model_shape(config))
    runs = {}
    for name in os.listdir(folder):
        m = re.match(rf"{key}_rate([0-9.]+)\.csv$", name)
        if m:
            runs[float(m.group(1))] = load_run(os.path.join(folder, name[:-4]), config, shape, gpus)
    if not runs:
        return None
    return {"key": key, "family": family, "gpus": gpus, "label": label, "runs": runs}


def collect(out_dir):
    """[{p, n, gpus, pd, colocated}] per P/D layout, each side None if not run yet."""
    cache, pairs = {}, []
    for p, n in PD_LAYOUTS:
        g = p + n
        pairs.append({
            "p": p, "n": n, "gpus": g,
            "pd": load_layout(out_dir, f"pd_{p}p{n}d", "pd", g, f"P/D {p} prefill + {n} decode", cache),
            "colocated": load_layout(out_dir, f"colocated_{g}", "colocated", g, f"colocated x{g}", cache),
        })
    return pairs


def goodput(layout):
    ok = [rate for rate, run in layout["runs"].items() if run["meets_slo"]]
    return max(ok) if ok else 0.0


def layouts_of(pairs):
    return [lay for p in pairs for lay in (p["pd"], p["colocated"]) if lay]


# ----------------------------------------------------------------- reports

def gpu_rows(pairs):
    rows = []
    for lay in layouts_of(pairs):
        for rate, run in sorted(lay["runs"].items()):
            for inst, role in enumerate(run["roles"]):
                u = run["util"].get(inst, {})
                s = run["log_win"].get(inst, {"running": [], "waiting": [], "mem": []})
                idx = sum(1 for r in run["roles"][:inst] if r == role)
                rows.append({
                    "layout": lay["key"], "gpus": lay["gpus"], "rate": f"{rate:g}",
                    "gpu": inst, "role": f"{role} {idx}",
                    "busy_pct": u.get("busy_pct", ""), "step_ms_mean": u.get("step_ms_mean", ""),
                    "mfu_busy_pct": u.get("mfu_busy_pct", ""), "mfu_wall_pct": u.get("mfu_wall_pct", ""),
                    "mbu_busy_pct": u.get("mbu_busy_pct", ""), "mbu_wall_pct": u.get("mbu_wall_pct", ""),
                    "recompute_tokens": u.get("recompute_tokens", ""),
                    "running_mean": mean(s["running"]), "waiting_mean": mean(s["waiting"]),
                    "mem_mean_pct": mean(s["mem"]), "mem_max_pct": max(s["mem"], default=""),
                })
    return rows


def fmt(v, nd=1):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_summary_md(pairs, path):
    """All per-run metrics as Markdown, one table per comparison (equal GPU count)."""
    cols = [("Layout", None), ("Rate", None), ("Served", "served", 2), ("SLO", None),
            ("Queue", "queue_mean", 2), ("TTFT mean", "ttft_mean", 2), ("TTFT p50", "ttft_p50", 2),
            ("TTFT p99", "ttft_p99", 2), ("TPOT mean", "tpot_mean", 1), ("TPOT p99", "tpot_p99", 1),
            ("Delay mean", "delay_mean", 1), ("Delay p50", "delay_p50", 1), ("Delay p99", "delay_p99", 1),
            ("ms/token", "delay_per_tok_mean", 1), ("Preempt /1k", "preemptions", 1),
            ("Recomputed tok /1k", "recomputed_tokens", 0), ("Deferred /1k", "deferred_handoffs", 1)]
    with open(path, "w") as f:
        f.write(f"# {MODEL_TITLE} P/D balance: all runs\n\n"
                "Units: served = requests completed per second (total); queue, TTFT and delay in s "
                "(means unless marked); TPOT and ms/token in ms. Delay = arrival to last token; "
                "ms/token = delay / output tokens. Queue = arrival to first scheduled (the prefill "
                "queue in P/D). All values are steady state: requests arriving (and, for served, "
                f"completions) between {WARMUP_S} s and {DURATION_S} s. SLO = TTFT p99 <= "
                f"{SLO_TTFT_P99_S:g} s and TPOT p99 <= {SLO_TPOT_P99_MS:g} ms. Preempt, recomputed "
                "and deferred (P/D handoffs held while the decode KV is full) are whole-run "
                "totals per 1000 requests.\n")
        for pair in pairs:
            lays = [l for l in (pair["pd"], pair["colocated"]) if l]
            if not lays:
                continue
            f.write(f"\n## {pair['gpus']} GPUs\n\n"
                    + "  \n".join(f"Goodput {l['label']}: {goodput(l):g} req/s" for l in lays) + "\n\n")
            f.write("| " + " | ".join(c[0] for c in cols) + " |\n")
            f.write("|" + "|".join("---" if c[1] is None else "---:" for c in cols) + "|\n")
            for lay in lays:
                for rate, run in sorted(lay["runs"].items()):
                    cells = [lay["label"], f"{rate:g}"]
                    for name, key, *nd in cols[2:]:
                        if name == "SLO":
                            cells.append("yes" if run["meets_slo"] else "no")
                        else:
                            v = run[key]
                            cells.append(f"{v:,}" if isinstance(v, int) else f"{v:.{nd[0]}f}")
                    f.write("| " + " | ".join(cells) + " |\n")


def write_reports(pairs, out_dir):
    lays = layouts_of(pairs)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        fields = ["layout", "gpus", "rate", "requests", "ttft_mean_s", "ttft_p50_s", "ttft_p99_s",
                  "tpot_mean_ms", "tpot_p99_ms", "delay_mean_s", "delay_p50_s", "delay_p99_s",
                  "delay_per_token_mean_ms", "queue_mean_s", "served_req_per_s", "meets_slo",
                  *(f"{k}_per_1k_req" for k in LOG_COUNTERS)]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for lay in lays:
            for rate, run in sorted(lay["runs"].items()):
                w.writerow({
                    "layout": lay["key"], "gpus": lay["gpus"], "rate": f"{rate:g}",
                    "requests": len(run["ttft_s"]), "ttft_mean_s": round(run["ttft_mean"], 4),
                    "ttft_p50_s": round(run["ttft_p50"], 4), "ttft_p99_s": round(run["ttft_p99"], 4),
                    "tpot_mean_ms": round(run["tpot_mean"], 3), "tpot_p99_ms": round(run["tpot_p99"], 3),
                    "delay_mean_s": round(run["delay_mean"], 3), "delay_p50_s": round(run["delay_p50"], 3),
                    "delay_p99_s": round(run["delay_p99"], 3),
                    "delay_per_token_mean_ms": round(run["delay_per_tok_mean"], 3),
                    "queue_mean_s": round(run["queue_mean"], 3),
                    "served_req_per_s": round(run["served"], 3), "meets_slo": int(run["meets_slo"]),
                    **{f"{k}_per_1k_req": round(run[k], 2) for k in LOG_COUNTERS}})
    write_summary_md(pairs, os.path.join(out_dir, "summary.md"))
    rows = gpu_rows(pairs)
    if not rows:
        return
    with open(os.path.join(out_dir, "gpu_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(out_dir, "gpu_report.txt"), "w") as f:
        hdr = (f"{'layout':<13}{'rate':>5}  {'GPU':<13}{'busy %':>7}{'step ms':>9}"
               f"{'MFU busy':>9}{'MBU busy':>9}{'running':>9}{'waiting':>9}{'mem %':>7}{'recompute':>11}")
        f.write("busy %, step ms, MFU/MBU (busy = over the GPU's step time) from the step log; "
                "running / waiting / mem % are means of per-second samples\n"
                + hdr + "\n" + "-" * len(hdr) + "\n")
        for r in rows:
            f.write(f"{r['layout']:<13}{r['rate']:>5}  {r['role']:<13}{fmt(r['busy_pct']):>7}"
                    f"{fmt(r['step_ms_mean']):>9}{fmt(r['mfu_busy_pct']):>9}{fmt(r['mbu_busy_pct']):>9}"
                    f"{fmt(r['running_mean']):>9}{fmt(r['waiting_mean']):>9}{fmt(r['mem_mean_pct']):>7}"
                    f"{r['recompute_tokens']:>11}\n")
        f.write(f"\nGoodput (highest tested rate with TTFT p99 <= {SLO_TTFT_P99_S:g} s and "
                f"TPOT p99 <= {SLO_TPOT_P99_MS:g} ms):\n")
        for lay in lays:
            g = goodput(lay)
            f.write(f"  {lay['label']:<26} {lay['gpus']} GPUs: {g:g} req/s total "
                    f"({g / lay['gpus']:.2f} per GPU)\n")


# ------------------------------------------------- per-comparison plots

def pair_handles(pair):
    hs = []
    if pair["pd"]:
        hs.append(plt.Line2D([], [], color=PD_COLOR, marker="s", linewidth=2, label=pair["pd"]["label"]))
    if pair["colocated"]:
        hs.append(plt.Line2D([], [], color=CO_COLOR, marker="o", linewidth=2, linestyle="--",
                             label=pair["colocated"]["label"]))
    return hs


def draw(ax, lay, ys_of):
    pd = lay["family"] == "pd"
    xs = sorted(lay["runs"])
    ax.plot(xs, [ys_of(lay["runs"][x]) for x in xs], color=PD_COLOR if pd else CO_COLOR,
            marker="s" if pd else "o", markersize=5, linewidth=2, linestyle="-" if pd else "--")


def plot_latency_throughput(pair, out_path):
    panels = [
        ("TTFT p50 (s, log)", "ttft_p50", "log", None),
        ("TTFT mean (s, log)", "ttft_mean", "log", None),
        ("TTFT p99 (s, log)", "ttft_p99", "log", SLO_TTFT_P99_S),
        ("TPOT mean (ms)", "tpot_mean", "lin", None),
        ("TPOT p99 (ms)", "tpot_p99", "lin", SLO_TPOT_P99_MS),
        ("Requests served per second (total)", "served", "served", None),
    ]
    lays = [l for l in (pair["pd"], pair["colocated"]) if l]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=SURFACE)
    for ax, (title, key, scale, slo) in zip(axes.flat, panels):
        style(ax)
        for lay in lays:
            draw(ax, lay, lambda r, k=key: r[k])
        if scale == "log":
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        if slo is not None:
            ax.axhline(slo, color=INK_2, linewidth=1, linestyle=":", zorder=0)
            ax.text(ax.get_xlim()[0], slo, " target", va="bottom", fontsize=8, color=INK_2)
        if scale == "served":
            # y stops a bit above the best served rate; the reference line runs
            # out of the panel where a layout stops keeping up.
            ymax = max(r["served"] for l in lays for r in l["runs"].values()) * 1.3
            top = max(max(l["runs"]) for l in lays)
            ax.plot([0, top], [0, top], color=INK_2, linewidth=1, linestyle=":", zorder=0)
            ax.text(ymax * 0.97, ymax * 0.97, "served = offered", ha="right", va="top",
                    fontsize=8, color=INK_2)
            ax.set_xlim(left=0)
            ax.set_ylim(0, ymax)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    good = "   ".join(f"{l['label']}: goodput {goodput(l):g} req/s" for l in lays)
    suptitle(fig, f"{MODEL_TITLE}, {pair['gpus']} GPUs: latency and throughput vs request rate\n{good}")
    legend_top(fig, pair_handles(pair))
    finish(fig, out_path, top_in=0.95)


def plot_delay(pair, out_path):
    """End-to-end delay = arrival -> last token (queueing + prefill + handoff + decode)."""
    panels = [
        ("Delay p50 (s, log)", "delay_p50"),
        ("Delay mean (s, log)", "delay_mean"),
        ("Delay p99 (s, log)", "delay_p99"),
        ("Mean delay per output token (ms, log)\n= delay ÷ output tokens, per request", "delay_per_tok_mean"),
    ]
    lays = [l for l in (pair["pd"], pair["colocated"]) if l]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.9), facecolor=SURFACE)
    for ax, (title, key) in zip(axes, panels):
        style(ax)
        for lay in lays:
            draw(ax, lay, lambda r, k=key: r[k])
        ax.set_yscale("log")
        plain_log_ticks(ax)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    suptitle(fig, f"{MODEL_TITLE}, {pair['gpus']} GPUs: end-to-end delay (arrival → last token) "
                  f"vs request rate")
    legend_top(fig, pair_handles(pair))
    finish(fig, out_path, top_in=0.6)


def role_stats(run, role, key, source):
    """Mean, min and max of a per-GPU value over the GPUs with this role."""
    vals = []
    for i, r in enumerate(run["roles"]):
        if r != role:
            continue
        if source == "util" and i in run["util"]:
            vals.append(run["util"][i][key])
        elif source == "log" and i in run["log_win"]:
            vals.append(mean(run["log_win"][i][key]))
    return (mean(vals), min(vals), max(vals)) if vals else None


def plot_gpu_load(pair, out_path):
    panels = [
        ("Busy % (share of time running a step)", "busy_pct", "util", (0, 105)),
        ("MFU while busy (% of peak FLOP/s)", "mfu_busy_pct", "util", (0, 100)),
        ("MBU while busy (% of peak bandwidth)", "mbu_busy_pct", "util", (0, 100)),
        ("Memory used (% of GPU)", "mem", "log", (0, 100)),
        ("Requests running per GPU", "running", "log", None),
        ("Requests waiting per GPU (log)", "waiting", "log", "log"),
    ]
    series = []           # (layout, role, color, linestyle)
    if pair["pd"]:
        series += [(pair["pd"], "prefill", ROLE_COLORS["prefill"], "-"),
                   (pair["pd"], "decode", ROLE_COLORS["decode"], "-")]
    if pair["colocated"]:
        series.append((pair["colocated"], "colocated", ROLE_COLORS["colocated"], "--"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=SURFACE)
    for ax, (title, key, source, ylim) in zip(axes.flat, panels):
        style(ax)
        for lay, role, color, ls in series:
            pts = [(x, role_stats(lay["runs"][x], role, key, source)) for x in sorted(lay["runs"])]
            pts = [(x, v) for x, v in pts if v is not None and (ylim != "log" or v[0] > 0)]
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ax.plot(xs, [v[0] for _, v in pts], color=color, linestyle=ls, linewidth=2,
                    marker="s" if ls == "-" else "o", markersize=4)
            if role != "prefill" or pair["p"] > 1:
                ax.fill_between(xs, [v[1] for _, v in pts], [v[2] for _, v in pts],
                                color=color, alpha=0.15, linewidth=0)
        if key == "mem":
            ax.axhline(MEM_UTIL_PCT, color=INK_2, linewidth=1, linestyle=":", zorder=0)
            ax.text(ax.get_xlim()[0], MEM_UTIL_PCT, " usable limit", va="bottom", fontsize=8, color=INK_2)
        if ylim == "log":
            ax.set_yscale("log")
        elif ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    suptitle(fig, f"{MODEL_TITLE}, {pair['gpus']} GPUs: how loaded each kind of GPU is "
                  f"(line = mean over GPUs of that kind, band = min-max)")
    p, n = pair["p"], pair["n"]
    legend_top(fig, [
        plt.Line2D([], [], color=ROLE_COLORS["prefill"], marker="s", linewidth=2,
                   label=f"P/D: prefill GPU{'s' if p > 1 else ''} ({p})"),
        plt.Line2D([], [], color=ROLE_COLORS["decode"], marker="s", linewidth=2,
                   label=f"P/D: decode GPU{'s' if n > 1 else ''} ({n})"),
        plt.Line2D([], [], color=ROLE_COLORS["colocated"], marker="o", linewidth=2, linestyle="--",
                   label=f"colocated GPUs ({pair['gpus']})"),
    ])
    finish(fig, out_path)


def plot_kv_pressure(pair, out_path):
    panels = [("Preemptions per 1000 requests", "preemptions"),
              ("Recomputed tokens per 1000 requests", "recomputed_tokens"),
              ("P/D handoffs deferred (decode KV full)\nper 1000 requests", "deferred_handoffs")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
    for ax, (title, key) in zip(axes, panels):
        style(ax)
        for lay in (pair["pd"], pair["colocated"]):
            if lay and not (key == "deferred_handoffs" and lay["family"] != "pd"):
                draw(ax, lay, lambda r, k=key: r[k])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    suptitle(fig, f"{MODEL_TITLE}, {pair['gpus']} GPUs: KV-cache pressure")
    legend_top(fig, pair_handles(pair))
    finish(fig, out_path)


def plot_ttft_timeline(pair, out_path):
    lays = [l for l in (pair["colocated"], pair["pd"]) if l]
    rates = sorted(set().union(*(l["runs"] for l in lays)))
    ncols = 5 if len(rates) > 6 else 3
    nrows = math.ceil(len(rates) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows + 1.0),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    for ax, rate in zip(axes.flat, rates):
        style(ax)
        ax.set_yscale("log")
        for lay in lays:
            run = lay["runs"].get(rate)
            if run:
                pd = lay["family"] == "pd"
                ax.scatter(run["arrival_s"], run["ttft_s"], s=8, color=PD_COLOR if pd else CO_COLOR,
                           marker="s" if pd else "o", alpha=0.6, linewidths=0)
        ax.set_title(f"{rate:g} req/s", fontsize=9, color=INK, loc="left")
        mark_window(ax)
    for ax in list(axes.flat)[len(rates):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("TTFT (s, log)", fontsize=9, color=INK_2)
    for ax in axes[-1, :]:
        ax.set_xlabel("arrival time (s)", fontsize=9, color=INK_2)
    suptitle(fig, f"{MODEL_TITLE}, {pair['gpus']} GPUs: TTFT of each request vs its arrival")
    legend_top(fig, pair_handles(pair))
    finish(fig, out_path)


def plot_timeline(rate, run, title, out_path):
    """One run: running, memory, MFU and MBU of each GPU over time."""
    rows = [("running", "Requests running", "log"), ("mem", "Memory used (% of GPU)", "log"),
            ("mfu", "MFU (% of peak FLOP/s)", "util"), ("mbu", "MBU (% of peak bandwidth)", "util")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 11), sharex=True, facecolor=SURFACE)
    handles = []
    for ax, (key, sub, source) in zip(axes, rows):
        style(ax)
        d = pf = 0
        for inst, role in enumerate(run["roles"]):
            if role == "prefill":
                color, label = PREFILL_SHADES[pf % len(PREFILL_SHADES)], f"prefill GPU {pf}"
                pf += 1
            elif role == "decode":
                color, label = DECODE_SHADES[d % len(DECODE_SHADES)], f"decode GPU {d}"
                d += 1
            else:
                color, label = COLOCATED_SHADES[d % len(COLOCATED_SHADES)], f"GPU {d}"
                d += 1
            series = (run["log"] if source == "log" else run["util"]).get(inst)
            if not series:
                continue
            ax.plot(series["t"], series[key], color=color, linewidth=1.1)
            if key == "running":
                handles.append(plt.Line2D([], [], color=color, linewidth=2, label=label))
        if key == "mem":
            ax.axhline(MEM_UTIL_PCT, color=INK_2, linewidth=1, linestyle=":", zorder=0)
        ax.set_ylim(bottom=0)
        ax.set_title(sub, fontsize=10, color=INK, loc="left")
        mark_window(ax)
    axes[-1].set_xlabel(f"simulated time (s); grey = warm-up, dotted = last arrival "
                        f"(run values use {WARMUP_S}-{DURATION_S} s)", fontsize=9, color=INK_2)
    suptitle(fig, f"{MODEL_TITLE}, {title} at {rate:g} req/s: each GPU over time")
    ncol = min(len(handles), 6)
    legend_top(fig, handles, ncol=ncol)
    finish(fig, out_path, dpi=130, top_in=0.45 + 0.25 * math.ceil(len(handles) / ncol))


# ----------------------------------------------------------- summary plots

def plot_capacity(pairs, out_path):
    """Across GPU counts: totals, and per GPU (the only place per-GPU is shown)."""
    panels = [
        ("Max requests served per second (total, steady state)", lambda l: max(r["served"] for r in l["runs"].values())),
        (f"Goodput, total (req/s)\nhighest rate with TTFT p99 ≤ {SLO_TTFT_P99_S:g} s, "
         f"TPOT p99 ≤ {SLO_TPOT_P99_MS:g} ms", goodput),
        ("Goodput per GPU (req/s)\n= total ÷ GPUs: efficiency, not capacity",
         lambda l: goodput(l) / l["gpus"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), facecolor=SURFACE)
    for ax, (title, fn) in zip(axes, panels):
        style(ax)
        for family, color, marker, ls in (("pd", PD_COLOR, "s", "-"), ("colocated", CO_COLOR, "o", "--")):
            pts = sorted((p["gpus"], fn(p[family])) for p in pairs if p[family])
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], color=color, marker=marker,
                        markersize=7, linewidth=2, linestyle=ls)
                for x, y in pts:
                    ax.annotate(f"{y:.2g}", (x, y), textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=8, color=color)
        ax.set_xticks([p["gpus"] for p in pairs])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("GPUs", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    suptitle(fig, f"{MODEL_TITLE} on RTXPRO6000 (TP1): capacity by GPU count "
                  f"(goodput is limited to the tested rates)")
    pd_names = ", ".join(f"{p['p']}P+{p['n']}D" for p in pairs if p["pd"])
    legend_top(fig, [plt.Line2D([], [], color=PD_COLOR, marker="s", linewidth=2,
                                label=f"P/D: {pd_names}"),
                     plt.Line2D([], [], color=CO_COLOR, marker="o", linewidth=2, linestyle="--",
                                label="colocated")])
    finish(fig, out_path, top_in=0.8)


def plot_pd_balance(pairs, out_path):
    """P/D layouts: which side saturates first, one line per decode count."""
    panels = [
        ("Prefill GPUs: busy %", "prefill", "busy_pct", "util", (0, 105)),
        ("Prefill GPUs: requests waiting per GPU (log)", "prefill", "waiting", "log", "log"),
        ("Decode GPUs: memory used (% of GPU)", "decode", "mem", "log", (60, 95)),
        ("Decode GPUs: requests running per GPU", "decode", "running", "log", None),
        ("Decode GPUs: MBU while busy (%)", "decode", "mbu_busy_pct", "util", (0, 100)),
        ("Preemptions per 1000 requests", None, "preemptions", "run", None),
    ]
    pds = [p["pd"] for p in pairs if p["pd"]]
    shade = {l["key"]: PD_RAMP[min(i, len(PD_RAMP) - 1)] for i, l in enumerate(pds)}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor=SURFACE)
    for ax, (title, role, key, source, ylim) in zip(axes.flat, panels):
        style(ax)
        for lay in pds:
            pts = []
            for x in sorted(lay["runs"]):
                run = lay["runs"][x]
                if source == "run":
                    v = run[key]
                else:
                    st = role_stats(run, role, key, source)
                    v = st[0] if st else None
                if v is not None and (ylim != "log" or v > 0):
                    pts.append((x, v))
            if pts:
                ax.plot([x for x, _ in pts], [v for _, v in pts], color=shade[lay["key"]],
                        marker="s", markersize=4, linewidth=2)
        if key == "mem":
            ax.axhline(MEM_UTIL_PCT, color=INK_2, linewidth=1, linestyle=":", zorder=0)
        if ylim == "log":
            ax.set_yscale("log")
        elif ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    suptitle(fig, f"{MODEL_TITLE}, P/D: which side saturates first, by layout (mean per GPU of each role)")
    legend_top(fig, [plt.Line2D([], [], color=shade[l["key"]], marker="s", linewidth=2,
                                label=l["label"]) for l in pds])
    finish(fig, out_path)


def plot_delay_summary(pairs, out_path):
    """Every layout on one axis: delay p50 and p99 vs rate, shaded by GPU count."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    handles = []
    for i, p in enumerate(pairs):
        for family, shades, marker, ls in (("pd", PD_RAMP, "s", "-"),
                                           ("colocated", CO_RAMP, "o", "--")):
            lay = p[family]
            if not lay:
                continue
            color = shades[min(i, len(shades) - 1)]  # darker = more GPUs
            xs = sorted(lay["runs"])
            for ax, key in zip(axes, ("delay_p50", "delay_p99")):
                ax.plot(xs, [lay["runs"][x][key] for x in xs], color=color, marker=marker,
                        markersize=4, linewidth=1.8, linestyle=ls)
            handles.append(plt.Line2D([], [], color=color, marker=marker, linewidth=1.8, linestyle=ls,
                                      label=f"{lay['label']} ({lay['gpus']} GPUs)"))
    for ax, title in zip(axes, ("Delay p50 (s, log)", "Delay p99 (s, log)")):
        style(ax)
        ax.set_yscale("log")
        plain_log_ticks(ax)
        ax.set_xlabel("request rate (req/s)", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    suptitle(fig, f"{MODEL_TITLE}: end-to-end delay (arrival → last token) vs request rate, all layouts")
    legend_top(fig, handles, ncol=4)
    finish(fig, out_path, top_in=0.8)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/pd_balance"
    pairs = collect(out_dir)
    if not layouts_of(pairs):
        print(f"no finished runs under {out_dir}")
        return
    write_reports(pairs, out_dir)
    plot_dir = os.path.join(out_dir, "plots")
    for pair in pairs:
        if not (pair["pd"] or pair["colocated"]):
            continue
        g = pair["gpus"]
        sub = os.path.join(plot_dir, f"G{g}_{pair['p']}p{pair['n']}d_vs_colocated{g}")
        os.makedirs(os.path.join(sub, "timelines"), exist_ok=True)
        plot_latency_throughput(pair, os.path.join(sub, "latency_throughput.png"))
        plot_delay(pair, os.path.join(sub, "delay.png"))
        plot_gpu_load(pair, os.path.join(sub, "gpu_load.png"))
        plot_kv_pressure(pair, os.path.join(sub, "kv_pressure.png"))
        plot_ttft_timeline(pair, os.path.join(sub, "ttft_timeline.png"))
        for lay in (pair["pd"], pair["colocated"]):
            for rate, run in sorted((lay or {"runs": {}})["runs"].items()):
                plot_timeline(rate, run, lay["label"],
                              os.path.join(sub, "timelines", f"{lay['key']}_rate{rate:g}.png"))
    os.makedirs(os.path.join(plot_dir, "summary"), exist_ok=True)
    plot_capacity(pairs, os.path.join(plot_dir, "summary", "capacity.png"))
    plot_pd_balance(pairs, os.path.join(plot_dir, "summary", "pd_balance.png"))
    plot_delay_summary(pairs, os.path.join(plot_dir, "summary", "delay.png"))
    missing = [f"G{p['gpus']}" for p in pairs if not (p["pd"] and p["colocated"])]
    if missing:
        print(f"note: comparisons not complete yet: {', '.join(missing)}", file=sys.stderr)
    print(f"wrote {out_dir}/summary.csv, gpu_report.csv/.txt and plots in {plot_dir}/")


if __name__ == "__main__":
    main()
