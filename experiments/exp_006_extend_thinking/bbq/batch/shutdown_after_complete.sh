#!/bin/bash
# ROBUST shutdown watcher with multiple safety mechanisms

HARD_TIMEOUT_HOURS=3  # Absolute maximum runtime
LOG="/workspace/experiments/exp_006_extend_thinking/bbq/batch/shutdown_watcher.log"

echo "=========================================="
echo "SHUTDOWN WATCHER - STARTED $(date)"
echo "=========================================="
echo "Hard timeout: ${HARD_TIMEOUT_HOURS} hours"
echo "Will shutdown when qwen32b session ends OR after ${HARD_TIMEOUT_HOURS}h"
echo ""

# Calculate hard deadline
DEADLINE=$(($(date +%s) + HARD_TIMEOUT_HOURS * 3600))
echo "Hard deadline: $(date -d @$DEADLINE)"
echo ""

while true; do
    NOW=$(date +%s)
    
    # Check hard timeout
    if [[ $NOW -ge $DEADLINE ]]; then
        echo ""
        echo "[$(date)] HARD TIMEOUT REACHED - Forcing shutdown!"
        echo "Stopping pod in 30 seconds..."
        sleep 30
        runpodctl stop pod
        exit 0
    fi
    
    # Check if experiments finished
    if ! tmux has-session -t qwen32b 2>/dev/null; then
        echo ""
        echo "[$(date)] Experiments complete! (qwen32b session ended)"
        echo "Stopping pod in 60 seconds... (Ctrl+C to cancel)"
        sleep 60
        echo "Stopping pod now..."
        runpodctl stop pod
        exit 0
    fi
    
    # Status update every 5 minutes
    REMAINING=$(( (DEADLINE - NOW) / 60 ))
    echo "[$(date)] Experiments running... (${REMAINING} min until hard timeout)"
    
    sleep 300  # Check every 5 minutes
done
