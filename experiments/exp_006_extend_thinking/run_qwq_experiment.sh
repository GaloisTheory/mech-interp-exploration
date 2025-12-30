#!/bin/bash
# Run QwQ-32B experiment and stop pod when done

cd /workspace/experiments/exp_006_extend_thinking

python run_experiment.py \
    --name QWQ_32B_1 \
    --samples 5 \
    --max-pairs 50 \
    --conditions normal extended_1x extended_2x extended_5x

# Stop the pod when experiment completes
runpodctl stop pod $RUNPOD_POD_ID
