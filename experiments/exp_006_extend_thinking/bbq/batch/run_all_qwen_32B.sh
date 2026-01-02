#!/bin/bash
# Run all BBQ experiments for Qwen/Qwen3-32B
#
# This script runs experiments in two phases with a GPU work queue:
# - Phase 1: All non-dynamic experiments (22 total) - can run in parallel
# - Phase 2: All dynamic experiments (12 total) - require baseline token stats
#
# Features:
# - GPU work queue: Jobs start immediately when a GPU becomes free
# - Lock-based GPU assignment: No GPU conflicts
# - SSH resilient: Run inside tmux to survive disconnects
#
# Usage:
#   chmod +x run_all_qwen_32B.sh
#   tmux new-session -d -s qwen32b './run_all_qwen_32B.sh 2>&1 | tee qwen_32B_master.log'
#   tmux attach -t qwen32b

set -e
cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-32B"
PREFIX="qwen_32B"
LOCKDIR="/tmp/qwen32b_gpu_locks"
NUM_GPUS=4

mkdir -p outputs

# =============================================================================
# GPU WORK QUEUE FUNCTIONS
# =============================================================================

acquire_gpu() {
    # Block until a GPU is available, then return its ID
    while true; do
        for gpu in $(seq 0 $((NUM_GPUS - 1))); do
            if mkdir "$LOCKDIR/gpu_${gpu}" 2>/dev/null; then
                echo $gpu
                return
            fi
        done
        sleep 5
    done
}

release_gpu() {
    # Release GPU lock
    rmdir "$LOCKDIR/gpu_${1}" 2>/dev/null || true
}

run_experiment() {
    local config=$1
    local name=$(basename "$config" .yaml)
    local gpu=$(acquire_gpu)
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $name on GPU $gpu"
    
    CUDA_VISIBLE_DEVICES=$gpu python run_question_batch.py "configs/${config}" \
        --model "$MODEL" \
        --model-prefix "$PREFIX" \
        > "outputs/${PREFIX}_${name}.log" 2>&1
    
    local status=$?
    release_gpu $gpu
    
    if [ $status -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Finished $name (success)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Finished $name (exit: $status)"
    fi
}

# =============================================================================
# INITIALIZATION
# =============================================================================

# Cleanup any stale locks from previous runs
rm -rf "$LOCKDIR"
mkdir -p "$LOCKDIR"

echo "=========================================="
echo "Qwen-32B BBQ Experiments"
echo "=========================================="
echo "Model: $MODEL"
echo "Prefix: $PREFIX"
echo "GPUs: $NUM_GPUS"
echo "Lock dir: $LOCKDIR"
echo "Started: $(date)"
echo "=========================================="
echo ""

# =============================================================================
# PHASE 1: Non-dynamic experiments (22 total)
# =============================================================================
echo "============================================================"
echo "PHASE 1: Starting 22 non-dynamic experiments"
echo "============================================================"
echo ""

# 11 baseline experiments (one per category)
PHASE1_CONFIGS=(
    baseline_age
    baseline_disability
    baseline_gender
    baseline_nationality
    baseline_appearance
    baseline_race
    baseline_race_ses
    baseline_race_gender
    baseline_religion
    baseline_ses
    baseline_sexual_orientation
    force_immediate_answer
    blank_static_5
    blank_static_10
    blank_static_50
    blank_static_100
    blank_static_500
    incorrect_answer_1
    incorrect_answer_2
    incorrect_answer_6
    incorrect_answer_12
    incorrect_answer_62
)

echo "Launching ${#PHASE1_CONFIGS[@]} experiments..."
echo ""

for config in "${PHASE1_CONFIGS[@]}"; do
    run_experiment "${config}.yaml" &
done

echo "All Phase 1 experiments launched. Waiting for completion..."
wait

echo ""
echo "============================================================"
echo "PHASE 1 COMPLETE - $(date)"
echo "============================================================"
echo ""

# =============================================================================
# PHASE 2: Dynamic experiments (12 total)
# These require baseline token stats from Phase 1
# =============================================================================
echo "============================================================"
echo "PHASE 2: Starting 12 dynamic experiments"
echo "(These use baseline token statistics from Phase 1)"
echo "============================================================"
echo ""

PHASE2_CONFIGS=(
    blank_dynamic_median_1x
    blank_dynamic_median_2x
    blank_dynamic_median_5x
    blank_dynamic_max_1x
    blank_dynamic_max_2x
    blank_dynamic_max_5x
    incorrect_dynamic_median_1x
    incorrect_dynamic_median_2x
    incorrect_dynamic_median_5x
    incorrect_dynamic_max_1x
    incorrect_dynamic_max_2x
    incorrect_dynamic_max_5x
)

echo "Launching ${#PHASE2_CONFIGS[@]} experiments..."
echo ""

for config in "${PHASE2_CONFIGS[@]}"; do
    run_experiment "${config}.yaml" &
done

echo "All Phase 2 experiments launched. Waiting for completion..."
wait

echo ""
echo "============================================================"
echo "PHASE 2 COMPLETE - $(date)"
echo "============================================================"
echo ""

# =============================================================================
# CLEANUP
# =============================================================================
rm -rf "$LOCKDIR"

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Results are in: outputs/${PREFIX}_*/"
echo "Logs are in: outputs/${PREFIX}_*.log"
echo ""

# =============================================================================
# AUTO-SHUTDOWN (if enabled)
# =============================================================================
if [[ "${AUTO_SHUTDOWN:-false}" == "true" ]]; then
    echo "AUTO_SHUTDOWN is enabled. Stopping pod in 60 seconds..."
    echo "(Cancel with: tmux kill-session -t qwen32b)"
    sleep 60
    echo "Stopping pod now..."
    runpodctl stop pod
fi

