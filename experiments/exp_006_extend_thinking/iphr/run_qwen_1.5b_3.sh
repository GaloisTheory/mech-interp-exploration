#!/bin/bash
# Run Qwen_1.5B_3 experiment with 5 parallel shards in tmux
# 
# This experiment uses diverse question types (4 recommended datasets with ~13 pairs each = 52 total)
# Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# Conditions: normal, extended_1x, extended_2x, extended_5x
# Samples per question: 5
#
# Usage:
#   chmod +x run_qwen_1.5b_3.sh
#   ./run_qwen_1.5b_3.sh
#
# After starting, detach with Ctrl+B, D to keep running after SSH disconnect
# Reattach with: tmux attach -t qwen_1.5b_3
# When done, merge with: python merge_results.py Qwen_1.5B_3

cd /workspace/experiments/exp_006_extend_thinking/iphr

# Create outputs directory if it doesn't exist
mkdir -p outputs

# Kill any existing session with this name
tmux kill-session -t qwen_1.5b_3 2>/dev/null

echo "Starting Qwen_1.5B_3 experiment with 5 shards..."

# Start 5 shards in separate tmux windows
tmux new-session -d -s qwen_1.5b_3 -n shard1 \
  "python run_experiment.py --name Qwen_1.5B_3 --shard 1/5 --samples 5 2>&1 | tee outputs/Qwen_1.5B_3_shard1.log; exec bash"

for i in 2 3 4 5; do
  tmux new-window -t qwen_1.5b_3 -n "shard$i" \
    "python run_experiment.py --name Qwen_1.5B_3 --shard $i/5 --samples 5 2>&1 | tee outputs/Qwen_1.5B_3_shard$i.log; exec bash"
done

echo ""
echo "Started 5 shards in tmux session 'qwen_1.5b_3'"
echo ""
echo "Commands:"
echo "  Attach:  tmux attach -t qwen_1.5b_3"
echo "  Detach:  Ctrl+B, D (keeps running after SSH disconnect)"
echo "  Windows: Ctrl+B, n (next) / Ctrl+B, p (previous)"
echo "  Merge:   python merge_results.py Qwen_1.5B_3"
echo ""
echo "Attaching to session now..."
tmux attach -t qwen_1.5b_3

