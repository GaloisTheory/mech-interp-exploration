#!/usr/bin/env python3
"""Lightweight BBQ Results Explorer

Usage:
    - Run cells interactively with #%% markers in VS Code/Cursor
    - Or import and use functions directly
"""

#%% Imports
import os
import sys

_bbq_dir = '/workspace/experiments/exp_006_extend_thinking/bbq'
if _bbq_dir not in sys.path:
    sys.path.insert(0, _bbq_dir)

# Ensure we can import from the bbq directory
os.chdir(_bbq_dir)

from analyze_results_utils import (
    load_experiment,
    print_summary,
    inspect_item,
    list_experiments,
    sample_table,
    filter_wrong,
)

#%% Config
EXPERIMENT_NAME = "full_11cat_merged"

#%% List available experiments
list_experiments()

#%% Load experiment
data, config = load_experiment(EXPERIMENT_NAME)

#%% Summary
print_summary(data)

#%% Sample table - show answers for first few items
sample_table(data, n=10)

#%% Inspect a specific item with full reasoning
inspect_item(data, idx=0)

#%% Filter to only wrong answers (biased responses)
wrong = filter_wrong(data)
print(f"Found {len(wrong)} wrong answers")

#%% Inspect first wrong answer
if wrong:
    inspect_item(data, idx=wrong[0]['_idx'])

# %%

