#!/bin/bash
# Run all BBQ experiments for Qwen/Qwen3-1.7B
#
# This script runs experiments in two phases:
# - Phase 1: All non-dynamic experiments (22 total) - can run in parallel
# - Phase 2: All dynamic experiments (12 total) - require baseline token stats
#
# Usage:
#   chmod +x run_all_qwen_1.7B.sh
#   ./run_all_qwen_1.7B.sh

set -e

cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-1.7B"
PREFIX="qwen_1.7B"
OUTPUTS_DIR="outputs"

echo "============================================================"
echo "BBQ Experiments - Qwen/Qwen3-1.7B"
echo "============================================================"
echo ""
echo "Model: $MODEL"
echo "Output prefix: $PREFIX"
echo ""

# Create log directory
mkdir -p "$OUTPUTS_DIR"

# =============================================================================
# PHASE 1: Non-dynamic experiments (22 total)
# =============================================================================
echo "============================================================"
echo "PHASE 1: Running non-dynamic experiments (22 total)"
echo "============================================================"

# Track PIDs for waiting
PHASE1_PIDS=()

# 11 baseline experiments (one per category)
CATEGORIES=(
    "age"
    "disability"
    "gender"
    "nationality"
    "appearance"
    "race"
    "race_ses"
    "race_gender"
    "religion"
    "ses"
    "sexual_orientation"
)

for cat in "${CATEGORIES[@]}"; do
    echo "Starting: baseline_${cat}"
    python run_question_batch.py "configs/baseline_${cat}.yaml" \
        --model "$MODEL" \
        --model-prefix "$PREFIX" \
        > "$OUTPUTS_DIR/${PREFIX}_baseline_${cat}.log" 2>&1 &
    PHASE1_PIDS+=($!)
done

# 1 force_immediate_answer
echo "Starting: force_immediate_answer"
python run_question_batch.py configs/force_immediate_answer.yaml \
    --model "$MODEL" \
    --model-prefix "$PREFIX" \
    > "$OUTPUTS_DIR/${PREFIX}_force_immediate_answer.log" 2>&1 &
PHASE1_PIDS+=($!)

# 5 blank_static experiments
for n in 5 10 50 100 500; do
    echo "Starting: blank_static_${n}"
    python run_question_batch.py "configs/blank_static_${n}.yaml" \
        --model "$MODEL" \
        --model-prefix "$PREFIX" \
        > "$OUTPUTS_DIR/${PREFIX}_blank_static_${n}.log" 2>&1 &
    PHASE1_PIDS+=($!)
done

# 5 incorrect_static experiments
for n in 1 2 6 12 62; do
    echo "Starting: incorrect_answer_${n}"
    python run_question_batch.py "configs/incorrect_answer_${n}.yaml" \
        --model "$MODEL" \
        --model-prefix "$PREFIX" \
        > "$OUTPUTS_DIR/${PREFIX}_incorrect_answer_${n}.log" 2>&1 &
    PHASE1_PIDS+=($!)
done

echo ""
echo "Phase 1: Started ${#PHASE1_PIDS[@]} experiments"
echo "PIDs: ${PHASE1_PIDS[*]}"
echo ""
echo "Waiting for Phase 1 to complete..."

# Wait for all Phase 1 experiments
FAILED=0
for pid in "${PHASE1_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "WARNING: Process $pid failed"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED Phase 1 experiments failed. Check logs."
else
    echo "Phase 1 complete: All experiments succeeded"
fi

echo ""

# =============================================================================
# PHASE 2: Dynamic experiments (12 total)
# These require baseline token stats from Phase 1
# =============================================================================
echo "============================================================"
echo "PHASE 2: Running dynamic experiments (12 total)"
echo "============================================================"

PHASE2_PIDS=()

# 6 blank_dynamic experiments
for mode in "median" "max"; do
    for mult in 1 2 5; do
        name="blank_dynamic_${mode}_${mult}x"
        echo "Starting: $name"
        python run_question_batch.py "configs/${name}.yaml" \
            --model "$MODEL" \
            --model-prefix "$PREFIX" \
            > "$OUTPUTS_DIR/${PREFIX}_${name}.log" 2>&1 &
        PHASE2_PIDS+=($!)
    done
done

# 6 incorrect_dynamic experiments
for mode in "median" "max"; do
    for mult in 1 2 5; do
        name="incorrect_dynamic_${mode}_${mult}x"
        echo "Starting: $name"
        python run_question_batch.py "configs/${name}.yaml" \
            --model "$MODEL" \
            --model-prefix "$PREFIX" \
            > "$OUTPUTS_DIR/${PREFIX}_${name}.log" 2>&1 &
        PHASE2_PIDS+=($!)
    done
done

echo ""
echo "Phase 2: Started ${#PHASE2_PIDS[@]} experiments"
echo "PIDs: ${PHASE2_PIDS[*]}"
echo ""
echo "Waiting for Phase 2 to complete..."

# Wait for all Phase 2 experiments
FAILED2=0
for pid in "${PHASE2_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "WARNING: Process $pid failed"
        FAILED2=$((FAILED2 + 1))
    fi
done

if [ $FAILED2 -gt 0 ]; then
    echo "WARNING: $FAILED2 Phase 2 experiments failed. Check logs."
else
    echo "Phase 2 complete: All experiments succeeded"
fi

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Phase 1 failures: $FAILED"
echo "Phase 2 failures: $FAILED2"
echo ""
echo "Results are in: $OUTPUTS_DIR/${PREFIX}_*/"
echo "Logs are in: $OUTPUTS_DIR/${PREFIX}_*.log"

