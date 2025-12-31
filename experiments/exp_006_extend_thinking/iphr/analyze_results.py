#!/usr/bin/env python3
"""Lightweight IPHR Results Explorer

Usage:
    - Run cells interactively with #%% markers in VS Code/Cursor
    - Or import and use functions directly
"""

#%% Imports
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_results_utils import (
    load_experiment,
    print_summary,
    print_backtracking,
    analyze_backtracking,
    inspect_pair,
    compare_conditions,
    list_experiments,
    sample_table,
)

#%% Config
# EXPERIMENT_NAME = "QWQ_32B_1"
EXPERIMENT_NAME = "test_qwen3_3shot"
# NOTE: Set USE_MERGED=True for sharded experiments (like Qwen_1.5B_1)
USE_MERGED = True   # True for Qwen_1.5B_1 (sharded), False for QWQ_32B_1
USE_FIXED = False    # Always True to use corrected answer extraction


#%% List available experiments
list_experiments()

#%% Load experiment
data, config, exp_folder = load_experiment(EXPERIMENT_NAME, USE_MERGED, USE_FIXED)

#%% Summary
print_summary(data)

# %%
print(sample_table(data, pair_idx=3,))
# %%

# %%
#%% Inspect a specific pair with reasoning
inspect_pair(data, condition="normal", pair_idx=3, sample=3)
# %%
inspect_pair(data, condition="extended_1x", pair_idx=3, sample=3)

#%% Compare conditions for a pair
compare_conditions(data, pair_idx=0)

# %%
