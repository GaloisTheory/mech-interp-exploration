#!/usr/bin/env python3
"""IPHR Faithfulness Experiment - Interactive Notebook

IPHR = Implicit Post-Hoc Rationalization

This experiment tests whether forcing extended thinking in reasoning models 
increases unfaithful chain-of-thought by measuring how often models give
contradictory answers (YES/YES or NO/NO) to complementary question pairs.

Run cells interactively with #%% markers in VS Code/Cursor.

NOTE: This notebook uses the same functions as run_experiment.py via experiment_utils.py
to ensure identical behavior between interactive and CLI modes.
"""

#%% Imports
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import (
    load_model,
    load_pairs,
    run_experiment,
    save_results,
    print_results_summary,
)
from data.question_pairs import format_prompt
from generation import generate_response, generate_batch
from config import GenerationConfig

print("Imports complete ✓")

#%% Configuration
# ============================================================================
# EXPERIMENT SETTINGS - Modify these as needed
# ============================================================================

EXPERIMENT_NAME = "debug_run"

# Conditions to compare
CONDITIONS = ["normal", "extended_1x", "extended_2x", "extended_5x"]
# CONDITIONS = ["normal", "extended_1x", "extended_2x", "extended_5x", "extended_10x"]  # Full (slow)

# Number of samples per question (more = more stable estimates)
SAMPLES_PER_QUESTION = 3

# Test mode: use fewer pairs for quick debugging  
TEST_MODE = False

# Sharding for parallel execution
# Set to None to run all pairs, or (shard_id, total_shards) e.g. (1, 3)
SHARD = None  # Options: None, (1, 3), (2, 3), (3, 3)

# Verbose: print detailed per-sample output
VERBOSE = False

# Save full model outputs (large files)
SAVE_RAW = False

print("Configuration:")
print(f"  Experiment:  {EXPERIMENT_NAME}")
print(f"  Test mode:   {TEST_MODE}")
print(f"  Conditions:  {CONDITIONS}")
print(f"  Samples/Q:   {SAMPLES_PER_QUESTION}")
print(f"  Shard:       {SHARD if SHARD else 'None (all pairs)'}")
print(f"  Verbose:     {VERBOSE}")

#%% Load model
model, tokenizer = load_model()

#%% Load question pairs
pairs, shard_info = load_pairs(shard=SHARD, test_mode=TEST_MODE)

print(f"Total generations needed: {len(pairs) * len(CONDITIONS) * SAMPLES_PER_QUESTION * 2}")

# Preview first pair
q1, q2, q1_exp, q2_exp, category = pairs[0]
print(f"\nExample pair [{category}]:")
print(f"  Q1: {q1[:70]}...")
print(f"  Q2: {q2[:70]}...")
print(f"  Expected: Q1={q1_exp}, Q2={q2_exp}")

#%% Run experiment
results = run_experiment(
    model=model,
    tokenizer=tokenizer,
    pairs=pairs,
    conditions=CONDITIONS,
    samples_per_question=SAMPLES_PER_QUESTION,
    verbose=VERBOSE,
    save_raw=SAVE_RAW,
)

#%% Display results
print_results_summary(results, CONDITIONS)

#%% Save results
output_file = save_results(
    results_by_condition=results,
    experiment_name=EXPERIMENT_NAME,
    conditions=CONDITIONS,
    samples_per_question=SAMPLES_PER_QUESTION,
    shard=shard_info,
    test_mode=TEST_MODE,
    save_raw=SAVE_RAW,
)

# ============================================================================
# DEBUG CELLS BELOW - Use for interactive exploration
# ============================================================================

#%% Debug: Test a single generation with extended thinking
test_q = pairs[0][0]  # First question
test_prompt = format_prompt(test_q)

print(f"Testing: {test_q[:60]}...")
print("-" * 60)

result = generate_response(model, tokenizer, test_prompt, "extended_5x", verbose=True)

print(f"\n{'='*60}")
print(f"Token count: {result.token_count}")
print(f"</think> positions: {result.think_end_positions}")
print(f"Answer extracted: {result.answer}")
print(f"\n--- Raw output (last 500 chars) ---")
print(result.full_output[-500:])

#%% Debug: Inspect specific results
# Uncomment and modify as needed

# condition = "extended_1x"
# pair_idx = 0
# 
# if condition in results and len(results[condition]) > pair_idx:
#     r = results[condition][pair_idx]
#     print(f"Category: {r.category}")
#     print(f"Q1: {r.q1_text[:80]}...")
#     print(f"Q1 answers: {r.q1_answers}")
#     print(f"Q2: {r.q2_text[:80]}...")
#     print(f"Q2 answers: {r.q2_answers}")
#     print(f"Unfaithful: {r.is_unfaithful} ({r.unfaithfulness_type})")

#%% Debug: Quick single-question test
# Uncomment to test a single generation

# test_q = "Is 17 × 23 greater than 400? Think step by step, then answer Yes or No."
# test_prompt = format_prompt(test_q)
# print(f"Testing: {test_q}")
# print("-" * 40)
# 
# result = generate_response(model, tokenizer, test_prompt, "normal", verbose=True)
# print(f"\nAnswer: {result.answer}")
# print(f"Tokens: {result.token_count}")
