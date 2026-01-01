#!/usr/bin/env python3
"""Interactive BBQ Batch Results Explorer

Usage:
    - Run cells interactively with #%% markers in VS Code/Cursor
    - Or import and use functions directly

Cells:
    1. List available experiments
    2. Load experiment(s)
    3. View summary statistics
    4. Inspect individual questions and samples
    5. View Chain-of-Thought reasoning
    6. Filter and compare results
"""

#%% Imports
import os
import sys
import importlib.util

# Determine batch directory - works in both script and Jupyter contexts
try:
    _batch_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ not defined in interactive/Jupyter context
    _batch_dir = '/workspace/experiments/exp_006_extend_thinking/bbq/batc/'
_bbq_dir = os.path.dirname(_batch_dir)

for path in [_batch_dir, _bbq_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Load batch_utils from absolute path (more reliable in Jupyter)
_batch_utils_path = os.path.join(_batch_dir, 'batch_utils.py')
_spec = importlib.util.spec_from_file_location("batch_utils", _batch_utils_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load batch_utils from {_batch_utils_path}")
batch_utils = importlib.util.module_from_spec(_spec)
sys.modules["batch_utils"] = batch_utils
_spec.loader.exec_module(batch_utils)

# Import the functions we need
list_batch_experiments = batch_utils.list_batch_experiments
load_batch_experiment = batch_utils.load_batch_experiment
print_batch_summary = batch_utils.print_batch_summary
inspect_question = batch_utils.inspect_question
show_cot = batch_utils.show_cot
filter_wrong_samples = batch_utils.filter_wrong_samples
accuracy_by_category = batch_utils.accuracy_by_category
compare_experiments = batch_utils.compare_experiments

#%% List available experiments
# Run this to see what experiments are available
experiments = list_batch_experiments()

#%% Load a single experiment
# Change the name/index to load different experiments
# Use index (0, 1, 2...) or partial name match ("force_immediate", "baseline")
exp = load_batch_experiment(0)  # Load most recent

#%% Print summary
print_batch_summary(exp)

#%% Accuracy by category
acc_by_cat = accuracy_by_category(exp)

#%% Inspect a specific question
# Shows the question, choices, and all sample answers
inspect_question(exp, question_idx=0)

#%% Show Chain-of-Thought for a specific sample
# Full reasoning trace for question 0, sample 0
show_cot(exp, question_idx=0, sample_idx=0)

#%% Another sample from the same question
show_cot(exp, question_idx=0, sample_idx=1)

#%% Filter to wrong answers (potential bias cases)
wrong = filter_wrong_samples(exp)
print(f"Found {len(wrong)} wrong samples")

# Show first few
for w in wrong[:5]:
    print(f"  Q{w['question_idx']} S{w['sample_idx']}: {w['category']} | Model: {w['model_answer']} | Correct: {w['correct_answer']}")

#%% Inspect a wrong answer
if wrong:
    w = wrong[0]
    inspect_question(exp, w['question_idx'])
    show_cot(exp, w['question_idx'], w['sample_idx'])

# =============================================================================
# COMPARING MULTIPLE EXPERIMENTS
# =============================================================================

#%% Load multiple experiments for comparison
# Uncomment and modify as needed:
#
# exp1 = load_batch_experiment("force_immediate")
# exp2 = load_batch_experiment("baseline")
# exp3 = load_batch_experiment("extended_1x")

#%% Compare experiments
# compare_experiments(
#     [exp1, exp2, exp3],
#     names=["Force Immediate", "Baseline", "Extended 1x"]
# )

# =============================================================================
# CUSTOM ANALYSIS
# =============================================================================

#%% Access raw data for custom analysis
# The experiment dict has this structure:
#
# exp = {
#     "config": { ... },            # Original YAML config
#     "started_at": "...",
#     "completed_at": "...",
#     "results": [                  # List of questions
#         {
#             "question_idx": 0,
#             "category": "race",
#             "context": "...",
#             "question": "...",
#             "choices": ["A", "B", "C"],
#             "correct_answer": "C",
#             "accuracy": 0.9,
#             "answer_distribution": {"A": 1, "C": 9},
#             "samples": [          # List of samples
#                 {
#                     "sample_idx": 0,
#                     "answer": "C",
#                     "correct": True,
#                     "tokens": 42,
#                     "time_s": 1.2,
#                     "full_output": "<think>...</think>\n\nC"
#                 },
#                 ...
#             ]
#         },
#         ...
#     ],
#     "summary": {
#         "total_questions": 110,
#         "total_samples": 1100,
#         "overall_accuracy": 0.82,
#         "total_time_s": 1234.5,
#         "by_category": {"race": 0.85, ...}
#     }
# }

#%% Example: Find questions where model was always wrong
if exp:
    always_wrong = [
        r for r in exp.get("results", [])
        if r.get("accuracy", 1) == 0
    ]
    print(f"Questions where model was always wrong: {len(always_wrong)}")
    for q in always_wrong[:5]:
        print(f"  [{q['question_idx']}] {q['category']}: {q['question'][:50]}...")

#%% Example: Find questions with high variance (split answers)
if exp:
    high_variance = [
        r for r in exp.get("results", [])
        if len(r.get("answer_distribution", {})) > 1
    ]
    print(f"Questions with split answers: {len(high_variance)}")
    for q in high_variance[:5]:
        print(f"  [{q['question_idx']}] {q['category']}: {q['answer_distribution']}")

#%% Example: Extract all CoT traces for a category
if exp:
    category = "race"
    race_cots = []
    for r in exp.get("results", []):
        if r.get("category") == category:
            for s in r.get("samples", []):
                race_cots.append({
                    "question": r["question"],
                    "answer": s["answer"],
                    "correct": s["correct"],
                    "cot": s["full_output"]
                })
    print(f"Extracted {len(race_cots)} CoT traces for {category}")

