#!/bin/bash
# TTFT vs load: {colocated, P/D with per-role batching}
# x request rate, for Llama-3.1-8B and Qwen3-32B on RTXPRO6000.
# Prefix caching off: an evicted (preempted) request recomputes its whole
# sequence. Each run simulates every request in its workload file
# (NUM_REQUESTS in prepare.py).
#
# Run from the repo root inside the simulator container:
#   ./experiments/pd_ttft/run.sh                   # both models, 8 runs at a time
#   JOBS=12 ./experiments/pd_ttft/run.sh           # more runs in parallel
#   MODELS="llama8b" ./experiments/pd_ttft/run.sh  # one model
#
# Each simulation is single-threaded, so independent runs go in parallel
# (each writes its ASTRA-Sim inputs to its own astra-sim/inputs/runs/<run_id>).
# Slowest runs (lowest rates) start first, one ASTRA-Sim startup at a time
# (see the staggered start below), and a watchdog stops any run whose
# ASTRA-Sim dies, marking it failed with the reason in its .failed file.
#
# Resumable: a run whose CSV already exists is skipped. A failed run keeps its
# .log (and a .failed marker) and is retried on the next invocation.
# Delete outputs/pd_ttft_long/ to start over. Plots land in outputs/pd_ttft_long/plots/.

set -uo pipefail

EXP=experiments/pd_ttft
OUT=outputs/pd_ttft_long
MODELS=${MODELS:-"llama8b qwen32b"}
JOBS=${JOBS:-8}
SYSTEMS=(colocated pd_custom)
# Must match MODELS[...]["rates"] in prepare.py.
declare -A RATES=(
    [llama8b]="2 4 6 8 10 12 15 20 30 40"
    [qwen32b]="0.5 1 1.5 2 3 4 5 6 8 10"
)

mkdir -p "$OUT"
# Two copies writing the same CSVs would corrupt each other's results.
exec 9> "$OUT/.run.lock"
if ! flock -n 9; then
    echo "another run.sh is already running on $OUT; not starting a second one" >&2
    exit 1
fi

python3 "$EXP/prepare.py" || exit 1

# The simulator prints this right after it has started ASTRA-Sim.
STARTED_MARK="Starting simulation"

# True once the run's ASTRA-Sim has exited while the simulator is still waiting
# on it. The simulator never notices that by itself: at startup it spins on
# empty reads until it runs out of memory (about an hour). A normal finish is
# excluded: ASTRA-Sim exits just before the simulator prints its results.
astra_died() {
    local py=$1 log=$2 child
    grep -q "$STARTED_MARK" "$log" || return 1          # ASTRA-Sim not started yet
    grep -q "Exiting simulation" "$log" && return 1     # normal end of the run
    child=$(pgrep -P "$py" | head -n 1)
    [[ -z "$child" ]] && return 0                       # gone
    [[ "$(ps -o stat= -p "$child")" == Z* ]]            # exited, not yet reaped
}

run_one() {
    local model=$1 system=$2 rate=$3
    local base="$OUT/$model/${system}_rate${rate}"
    local start=$SECONDS reason=""
    rm -f "$base.failed" "$base.steps.csv" "$base.steps.csv.tmp"
    # --step-log: one row per GPU step, for MFU/MBU in gpu_report.py. Each run
    # has its own file, and the simulator writes it to .tmp and renames it only
    # on a normal end, so parallel runs never share or half-write one.
    python -m serving --cluster-config "$EXP/configs/$model/$system.json" \
        --block-size 16 --no-enable-prefix-caching \
        --dataset "$EXP/workloads/$model/rate${rate}.jsonl" \
        --output "$base.csv" --step-log "$base.steps.csv" > "$base.log" 2>&1 &
    local py=$!
    # Watchdog: stop the run once its ASTRA-Sim has been dead on 3 checks in a
    # row (>10 s). A normal finish also passes through "ASTRA-Sim gone, simulator
    # still writing results", and the log may not show that yet (it is buffered),
    # but that lasts a second or two; a stuck run stays stuck.
    local dead_checks=0
    while kill -0 "$py" 2>/dev/null; do
        if astra_died "$py" "$base.log"; then
            dead_checks=$((dead_checks + 1))
            if (( dead_checks >= 3 )); then
                reason="ASTRA-Sim exited while the simulator was waiting on it"
                kill -9 "$py" 2>/dev/null
                # A killed simulator cannot clean up its ASTRA-Sim inputs.
                local run_id
                run_id=$(grep -o -m 1 "run_[0-9]*_[0-9]*" "$base.log")
                [[ -n "$run_id" ]] && rm -rf "astra-sim/inputs/runs/$run_id"
                break
            fi
        else
            dead_checks=0
        fi
        sleep 5
    done
    wait "$py"
    local rc=$?
    if [[ -z "$reason" && $rc -eq 0 ]]; then
        printf '%-8s %-11s rate=%-4s ok      (%d min)\n' "$model" "$system" "$rate" \
            $(( (SECONDS - start) / 60 ))
    else
        [[ -z "$reason" ]] && reason="$(grep -E 'Error|error' "$base.log" | tail -n 1 | cut -c1-100)"
        rm -f "$base.csv" "$base.steps.csv" "$base.steps.csv.tmp"
        echo "$reason" > "$base.failed"
        printf '%-8s %-11s rate=%-4s FAILED: %s\n' "$model" "$system" "$rate" "$reason"
    fi
}

# Return once the run logging to $1 (run_one's job $2) has started ASTRA-Sim,
# plus a second to let ASTRA-Sim finish its startup, or once that job has ended.
wait_for_startup() {
    local log=$1 job=$2
    while kill -0 "$job" 2>/dev/null && ! grep -q "$STARTED_MARK" "$log" 2>/dev/null; do
        sleep 0.2
    done
    sleep 1
}

# Queue every missing run, lowest rate (longest simulated span) first.
queue=()
skipped=0
for model in $MODELS; do
    mkdir -p "$OUT/$model"
    for rate in ${RATES[$model]}; do
        for system in "${SYSTEMS[@]}"; do
            if [[ -f "$OUT/$model/${system}_rate${rate}.csv" ]]; then
                skipped=$((skipped + 1))
            else
                queue+=("$rate $model $system")
            fi
        done
    done
done
mapfile -t queue < <(printf '%s\n' "${queue[@]}" | sort -g -k1,1)

echo "$((${#queue[@]})) runs to do ($skipped already done), $JOBS at a time"
for item in "${queue[@]}"; do
    [[ -z "$item" ]] && continue
    read -r rate model system <<< "$item"
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        wait -n
    done
    printf '%-8s %-11s rate=%-4s started [%3d min]\n' "$model" "$system" "$rate" $((SECONDS / 60))
    log="$OUT/$model/${system}_rate${rate}.log"
    : > "$log"
    run_one "$model" "$system" "$rate" &
    # Staggered start: every ASTRA-Sim writes, reads back and deletes the same
    # temp file (astra-sim/tmp__mem/remote_mem.json) at startup, so two starting
    # together can kill each other (42 of 240 simultaneous starts in a test, 0 of
    # 240 when isolated). Launch the next run only once this one's ASTRA-Sim is up.
    wait_for_startup "$log" $!
done
wait

echo
ok=$(ls "$OUT"/*/*_rate*[0-9].csv 2>/dev/null | wc -l)
failed=$(ls "$OUT"/*/*.failed 2>/dev/null | wc -l)
echo "done in $((SECONDS / 60)) min: $ok runs ok, $failed failed"
python3 "$EXP/plot.py" "$OUT"
python3 "$EXP/gpu_report.py" "$OUT"
