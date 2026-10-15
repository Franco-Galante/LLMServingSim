"""Summarize the sweep's per-request CSVs into one table.

    python3 experiments/pd_kv_sweep/summarize.py [outputs/pd_kv_sweep]

Prints a table and writes <dir>/summary.csv. Latencies in ms, throughput
from first arrival to last completion.
"""

import csv
import glob
import os
import re
import sys


def pct(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def mean(values):
    return sum(values) / len(values)


def summarize(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ms = lambda col: [int(r[col]) / 1e6 for r in rows]
    ttft, tpot, lat, queue = ms("TTFT"), ms("TPOT"), ms("latency"), ms("queuing_delay")
    span_s = (max(int(r["end_time"]) for r in rows) - min(int(r["arrival"]) for r in rows)) / 1e9
    out_toks = sum(int(r["output"]) for r in rows)
    return {
        "requests": len(rows),
        "ttft_mean": mean(ttft), "ttft_p50": pct(ttft, 50), "ttft_p99": pct(ttft, 99),
        "tpot_mean": mean(tpot), "tpot_p99": pct(tpot, 99),
        "latency_mean": mean(lat), "queue_mean": mean(queue),
        "req_per_s": len(rows) / span_s, "gen_tok_per_s": out_toks / span_s,
    }


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/pd_kv_sweep"
    results = []
    for path in glob.glob(os.path.join(out_dir, "*_rate*.csv")):
        m = re.match(r"(.+)_rate(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        system, rate = m.group(1), int(m.group(2))
        results.append({"system": system, "rate": rate, **summarize(path)})

    if not results:
        print(f"no results in {out_dir}")
        return

    def order(r):
        bw = r["system"].removeprefix("pd_bw")
        return (r["rate"], -1.0 if r["system"] == "colocated" else float(bw))
    results.sort(key=order)

    cols = list(results[0].keys())
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    hdr = f"{'system':<11}{'rate':>5}{'reqs':>6}{'TTFT mean':>11}{'p50':>9}{'p99':>10}" \
          f"{'TPOT mean':>11}{'p99':>8}{'queue':>9}{'req/s':>8}{'gen tok/s':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['system']:<11}{r['rate']:>5}{r['requests']:>6}"
              f"{r['ttft_mean']:>11.1f}{r['ttft_p50']:>9.1f}{r['ttft_p99']:>10.1f}"
              f"{r['tpot_mean']:>11.2f}{r['tpot_p99']:>8.2f}{r['queue_mean']:>9.1f}"
              f"{r['req_per_s']:>8.2f}{r['gen_tok_per_s']:>11.1f}")
    print(f"\nlatencies in ms; wrote {os.path.join(out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
