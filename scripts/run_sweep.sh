#!/usr/bin/env bash
# Sequential hyperparameter sweep runner.
# Runs each sweep config one after another and logs timings.
set -e

CONFIGS=(
  "configs/sweep/sweep_a2_t15.yaml"
  "configs/sweep/sweep_a2_t30.yaml"
  "configs/sweep/sweep_a25_t15.yaml"
  "configs/sweep/sweep_a25_t30.yaml"
  "configs/sweep/sweep_a3_t15.yaml"
  "configs/sweep/sweep_a3_t30.yaml"
)

LOG_DIR="runs/sweep_logs"
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/sweep_$(date +%Y%m%d_%H%M%S).log"

echo "[SWEEP] Starting sequential sweep of ${#CONFIGS[@]} configs" | tee "$SWEEP_LOG"
echo "[SWEEP] Log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    idx=$((i + 1))
    total=${#CONFIGS[@]}

    echo "" | tee -a "$SWEEP_LOG"
    echo "[SWEEP ${idx}/${total}] Starting: $name" | tee -a "$SWEEP_LOG"
    echo "[SWEEP] Time: $(date +%T)" | tee -a "$SWEEP_LOG"

    start_ts=$(date +%s)
    .venv/Scripts/python.exe run_train.py --config "$cfg" --train-ga --no-viz \
        >> "$SWEEP_LOG" 2>&1 || echo "[SWEEP] $name FAILED" | tee -a "$SWEEP_LOG"
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    echo "[SWEEP ${idx}/${total}] Done: $name in ${elapsed}s" | tee -a "$SWEEP_LOG"
done

echo "" | tee -a "$SWEEP_LOG"
echo "[SWEEP] All done. Log: $SWEEP_LOG" | tee -a "$SWEEP_LOG"
