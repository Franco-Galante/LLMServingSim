#!/bin/bash
# P/D disaggregation KV-transfer sweep: link bandwidth x request rate,
# plus a colocated 2-GPU baseline at each rate.
#
# Run from the repo root inside the simulator container:
#   ./experiments/pd_kv_sweep/run_sweep.sh
#
# Resumable: a run whose CSV already exists is skipped. Delete
# outputs/pd_kv_sweep/ to start over.

set -euo pipefail

EXP=experiments/pd_kv_sweep
OUT=outputs/pd_kv_sweep
LINKS=(12.5 50 300 900)
# 10 and 40 first: if time runs short, the two extremes are already done.
RATES=(10 40 20)
# No new run starts after this many minutes; rerun the script to finish the rest.
BUDGET_MIN=${BUDGET_MIN:-95}

mkdir -p "$OUT"
python3 "$EXP/prepare.py"

run() {
    local name=$1 config=$2 rate=$3
    local csv="$OUT/${name}_rate${rate}.csv"
    if [[ -f "$csv" ]]; then
        echo "skip  $name rate=$rate (exists)"
        return
    fi
    if (( SECONDS >= BUDGET_MIN * 60 )); then
        echo "stop  $name rate=$rate (past the ${BUDGET_MIN}-min budget)"
        return
    fi
    echo "run   $name rate=$rate  [$(( SECONDS / 60 )) min elapsed]"
    # Defaults are vLLM's: max-num-seqs 128, max-num-batched-tokens 2048,
    # chunked prefill and prefix caching on, LOAD routing.
    python -m serving --cluster-config "$config" \
        --block-size 16 \
        --dataset "$EXP/workloads/sharegpt_rate${rate}.jsonl" \
        --output "$csv" --num-reqs 300 \
        > "$OUT/${name}_rate${rate}.log" 2>&1 \
        || { echo "FAIL  $name rate=$rate, see $OUT/${name}_rate${rate}.log"; rm -f "$csv"; }
}

for rate in "${RATES[@]}"; do
    run colocated "$EXP/configs/colocated.json" "$rate"
    for bw in "${LINKS[@]}"; do
        run "pd_bw${bw}" "$EXP/configs/pd_bw${bw}.json" "$rate"
    done
done

python3 "$EXP/summarize.py" "$OUT"
