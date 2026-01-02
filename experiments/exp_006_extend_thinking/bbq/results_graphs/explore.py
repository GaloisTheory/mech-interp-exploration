#!/usr/bin/env python3
"""Interactive BBQ Exploration

Compare experiments, find divergent questions, and inspect CoT reasoning.

Usage:
    Run cells interactively with #%% markers in VS Code/Cursor
    
Example workflow:
    1. Load two experiments to compare
    2. Find divergent questions
    3. Inspect questions and view CoT reasoning
"""

#%% Imports
import os
import sys
from pathlib import Path

# For interactive use: cd to this script's directory so graph_utils can find paths
_script_dir = Path("/workspace/experiments/exp_006_extend_thinking/bbq/results_graphs")
os.chdir(_script_dir)
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from graph_utils import (
    load_experiment,
    find_divergent,
    list_divergent,
    inspect_question,
    show_cot,
    get_accuracy,
)

# =============================================================================
# LOAD EXPERIMENTS
# =============================================================================

#%% Load experiments - edit these to compare what you want
# Options for exp_type: 'cot', 'nocot', 'blank', 'incorrect'
# For 'blank' and 'incorrect', also specify n

exp1 = load_experiment('8B', 'blank', 500)
exp2 = load_experiment('8B', 'nocot')
exp3 = load_experiment('8B', 'cot')
exp4 = load_experiment('8B', 'blank', 10)
#%% Check accuracies
print(f"exp1 accuracy: {get_accuracy(exp1):.1%}")
print(f"exp2 accuracy: {get_accuracy(exp2):.1%}")

# =============================================================================
# FIND DIVERGENT QUESTIONS
# =============================================================================

#%% Find where experiments disagree
div = find_divergent(exp1, exp2, '8B_blank_500', '8B_nocot')

#%% List questions where exp1 was better
list_divergent(div, 'exp1_better')

#%% List questions where exp2 was better
list_divergent(div, 'exp2_better')

# =============================================================================
# INSPECT SPECIFIC QUESTIONS
# =============================================================================

#%% Pick a divergent question to inspect
# Change the index [0] to explore different questions
if div['exp1_better']:
    q = div['exp1_better'][1]
    print(f"Selected: {q['category']} - {q['question']}")
    print(f"idx in exp1: {q['idx']}, idx in exp2: {q['idx_exp2']}")

#%% Inspect the question in exp1
if div['exp1_better']:
    inspect_question(exp1, q['idx_exp2'])

#%% Inspect the same question in exp2
if div['exp1_better']:
    inspect_question(exp2, q['idx_exp2'])


# %%
inspect_question(exp2, q['idx_exp2'])

# %%
inspect_question(exp4, q['idx_exp2'])

# =============================================================================
# VIEW COT REASONING
# =============================================================================

#%% Show CoT from exp1
show_cot(exp2, q['idx_exp2'], sample=2)

# %%
show_cot(exp3, q['idx_exp2'], sample=2)
# %%
#%% Show CoT from exp2
if div['exp1_better']:
    show_cot(exp2, q['idx_exp2'], sample=0)

# %%
print(f"Question: {q['question']}")

#%% Compare multiple samples from the same question
if div['exp1_better']:
    print("Exp1 samples:")
    for i in range(3):
        show_cot(exp1, q['idx'], sample=i, max_length=500)

# =============================================================================
# QUICK EXAMPLES
# =============================================================================

#%% Example: Compare CoT vs NoCoT
cot = load_experiment('8B', 'cot')
nocot = load_experiment('8B', 'nocot')
div_cot = find_divergent(cot, nocot, 'CoT', 'NoCoT')

#%% Example: Compare across models
exp_8b = load_experiment('8B', 'incorrect', 62)
exp_32b = load_experiment('32B', 'incorrect', 62)
div_models = find_divergent(exp_8b, exp_32b, '8B', '32B')

#%% Example: Inspect a specific question by index
# If you know the question index from a graph/table
inspect_question(exp1, 5)
show_cot(exp1, 5, sample=0)

# =============================================================================
# SINGLE QUESTION ACCURACY vs BLANK SPACES
# =============================================================================

#%% Plot accuracy for a specific question across blank experiments
import matplotlib.pyplot as plt
import json

question_idx = 41  # The interesting Physical_appearance question
LOG_PATH = '/workspace/.cursor/debug.log'

def dbg(hyp, msg, data):
    # #region agent log
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps({'hypothesisId': hyp, 'message': msg, 'data': data, 'timestamp': __import__('time').time()}) + '\n')
    # #endregion

# Load all blank experiments and count correct answers
blank_ns = [5, 10, 50, 100, 500]
results = []
for n in blank_ns:
    exp = load_experiment('8B', 'blank', n)
    q = exp['results'][question_idx]
    samples = q['samples']
    correct = sum(1 for s in samples if s['correct'])
    # #region agent log
    dbg('H1', f'blank_{n} question at idx {question_idx}', {'question': q['question'][:50], 'category': q['category'], 'correct_answer': q['correct_answer']})
    # #endregion
    results.append((n, correct, len(samples)))

# Also get CoT and NoCoT for reference
cot_exp = load_experiment('8B', 'cot')
nocot_exp = load_experiment('8B', 'nocot')

cot_q = cot_exp['results'][question_idx]
nocot_q = nocot_exp['results'][question_idx]

# #region agent log
dbg('H1', f'cot question at idx {question_idx}', {'question': cot_q['question'][:50], 'category': cot_q['category'], 'correct_answer': cot_q['correct_answer']})
dbg('H1', f'nocot question at idx {question_idx}', {'question': nocot_q['question'][:50], 'category': nocot_q['category'], 'correct_answer': nocot_q['correct_answer']})
dbg('H5', 'nocot samples detail', {'answers': [s['answer'] for s in nocot_q['samples']], 'corrects': [s['correct'] for s in nocot_q['samples']]})
dbg('H2', 'nocot exp metadata', {'type': nocot_exp.get('_type'), 'n_results': len(nocot_exp['results'])})
# #endregion

cot_samples = cot_q['samples']
nocot_samples = nocot_q['samples']
cot_correct = sum(1 for s in cot_samples if s['correct'])
nocot_correct = sum(1 for s in nocot_samples if s['correct'])

# #region agent log
dbg('H5', 'counts', {'cot_correct': cot_correct, 'nocot_correct': nocot_correct})
# #endregion

#%% Plot it
x = [r[0] for r in results]
y = [r[1] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-o', linewidth=2, markersize=8, label='Blank spaces')
# plt.axhline(cot_correct, color='green', linestyle='--', label=f'CoT ({cot_correct}/10)')
# plt.axhline(nocot_correct, color='red', linestyle='--', label=f'NoCoT ({nocot_correct}/10)')
plt.xscale('log')
plt.xlabel('Number of Blank Spaces')
plt.ylabel('Correct Answers (out of 10)')
plt.title(f'Question {question_idx}: "Who fails at dieting?"')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%

#%% Plot resampled data
# Data from resampling 300 samples each
blanks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
accuracy = [0.0, 0.0, 0.0, 0.0, 9.0, 27.3, 81.7, 74.0, 73.0, 100.0, 100.0]

plt.figure(figsize=(8, 5))
plt.plot(blanks, accuracy, 'b-o', linewidth=2, markersize=8, label='Blank spaces')
plt.xlabel('Number of Blank Spaces')
plt.ylabel('Accuracy (%)')
plt.title(f'Question 41: "Who fails at dieting?" (resampled 300 samples)')
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(-5, 105)
plt.tight_layout()
plt.savefig('/workspace/experiments/exp_006_extend_thinking/bbq/results_graphs/q41_resampled_blank_vs_accuracy.png', dpi=150)
plt.show()

# %%
