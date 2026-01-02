#!/usr/bin/env python3
"""BBQ Model Comparison - Summary Table

Compare results across Qwen-1.7B, Qwen-8B, and Qwen-32B models.

Usage:
    Run cells interactively with #%% markers in VS Code/Cursor
"""

#%% Imports and Setup
import os
import sys
import json
import importlib.util
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

# Setup paths
_batch_dir = '/workspace/experiments/exp_006_extend_thinking/bbq/batch'
_output_dir = os.path.join(_batch_dir, 'outputs')
if _batch_dir not in sys.path:
    sys.path.insert(0, _batch_dir)

# Load batch_utils
_spec = importlib.util.spec_from_file_location("batch_utils", os.path.join(_batch_dir, 'batch_utils.py'))
batch_utils = importlib.util.module_from_spec(_spec)
sys.modules["batch_utils"] = batch_utils
_spec.loader.exec_module(batch_utils)

# Load analyze_batch_utils
_spec2 = importlib.util.spec_from_file_location("analyze_batch_utils", os.path.join(_batch_dir, 'analyze_batch_utils.py'))
analyze_utils = importlib.util.module_from_spec(_spec2)
sys.modules["analyze_batch_utils"] = analyze_utils
_spec2.loader.exec_module(analyze_utils)

load_batch_experiment = batch_utils.load_batch_experiment
load_baseline_combined = analyze_utils.load_baseline_combined
accuracy_by_question = analyze_utils.accuracy_by_question


def load_8B_experiment(name):
    """Load 8B experiment - explicitly exclude qwen_* prefixed folders."""
    folders = [f for f in os.listdir(_output_dir) if os.path.isdir(os.path.join(_output_dir, f))]
    # Filter: must contain name, must NOT start with qwen_
    matches = [f for f in folders if name.lower() in f.lower() and not f.startswith('qwen_')]
    if not matches:
        return None
    # Sort by modification time, get most recent
    matches.sort(key=lambda f: os.path.getmtime(os.path.join(_output_dir, f)), reverse=True)
    folder_path = os.path.join(_output_dir, matches[0])
    results_files = glob(os.path.join(folder_path, "results_*.json"))
    if not results_files:
        return None
    with open(max(results_files, key=os.path.getmtime), 'r') as f:
        return json.load(f)


print("✓ Imports loaded")

#%% Helper Functions
def get_accuracies_with_counts(exp):
    """Get question and sample accuracy with counts for an experiment."""
    if not exp or not exp.get('results'):
        return None, None
    
    results = exp.get('results', [])
    q_correct = q_total = s_correct = s_total = 0
    
    for r in results:
        q_total += 1
        # Question-level: majority vote correct?
        answer_dist = r.get('answer_distribution', {})
        if answer_dist:
            correct_answer = r.get('correct_answer', '')
            max_count = max(answer_dist.values())
            answers_with_max = [a for a, c in answer_dist.items() if c == max_count]
            if len(answers_with_max) == 1 and answers_with_max[0] == correct_answer:
                q_correct += 1
        # Sample-level
        for s in r.get('samples', []):
            s_total += 1
            if s.get('correct'):
                s_correct += 1
    
    q_acc = q_correct / q_total if q_total > 0 else 0
    s_acc = s_correct / s_total if s_total > 0 else 0
    return {'acc': q_acc, 'correct': q_correct, 'total': q_total}, {'acc': s_acc, 'correct': s_correct, 'total': s_total}

def load_model_data(prefix=None):
    """Load all experiments for a model and compute accuracies with counts."""
    # For 8B (no prefix), use special loader to avoid matching qwen_* folders
    if prefix is None:
        def load_exp(name):
            return load_8B_experiment(name)
    else:
        def load_exp(name):
            return load_batch_experiment(f'{prefix}_{name}')
    
    # Baselines
    cot = load_baseline_combined(load_batch_experiment, prefix=prefix)
    nocot = load_exp('force_immediate')
    cot_q, cot_s = get_accuracies_with_counts(cot)
    nocot_q, nocot_s = get_accuracies_with_counts(nocot)
    
    # Blank experiments
    all_blank = {}
    for n in [5, 10, 50, 100, 500]:
        exp = load_exp(f'blank_static_{n}_')
        q, s = get_accuracies_with_counts(exp)
        if q is not None:
            all_blank[f'static_{n}'] = {'q': q, 's': s}
    for mult in [1, 2, 5]:
        for typ in ['median', 'max']:
            exp = load_exp(f'blank_dynamic_{typ}_{mult}x_')
            q, s = get_accuracies_with_counts(exp)
            if q is not None:
                all_blank[f'{typ}_{mult}x'] = {'q': q, 's': s}
    
    # Incorrect experiments
    all_incorrect = {}
    for n in [1, 2, 6, 12, 62]:
        exp = load_exp(f'incorrect_answer_{n}_')
        q, s = get_accuracies_with_counts(exp)
        if q is not None:
            all_incorrect[f'static_{n}'] = {'q': q, 's': s}
    for mult in [1, 2, 5]:
        for typ in ['median', 'max']:
            exp = load_exp(f'incorrect_dynamic_{typ}_{mult}x_')
            q, s = get_accuracies_with_counts(exp)
            if q is not None:
                all_incorrect[f'{typ}_{mult}x'] = {'q': q, 's': s}
    
    # Find best/worst (by sample accuracy)
    blank_best = blank_worst = None
    if all_blank:
        blank_best_key = max(all_blank.keys(), key=lambda k: all_blank[k]['s']['acc'])
        blank_worst_key = min(all_blank.keys(), key=lambda k: all_blank[k]['s']['acc'])
        blank_best = all_blank[blank_best_key]
        blank_worst = all_blank[blank_worst_key]
    
    incorrect_best = incorrect_worst = None
    if all_incorrect:
        incorrect_best_key = max(all_incorrect.keys(), key=lambda k: all_incorrect[k]['s']['acc'])
        incorrect_worst_key = min(all_incorrect.keys(), key=lambda k: all_incorrect[k]['s']['acc'])
        incorrect_best = all_incorrect[incorrect_best_key]
        incorrect_worst = all_incorrect[incorrect_worst_key]
    
    return {
        'cot': {'q': cot_q, 's': cot_s},
        'nocot': {'q': nocot_q, 's': nocot_s},
        'blank_best': blank_best,
        'blank_worst': blank_worst,
        'incorrect_best': incorrect_best,
        'incorrect_worst': incorrect_worst,
    }

def fmt(stats):
    """Format accuracy with counts: '94.5% (100/110)'."""
    if stats is None:
        return 'N/A'
    return f"{stats['acc']:.1%} ({stats['correct']}/{stats['total']})"

#%% Load All Models
print("Loading experiments for all models...")
m_1_7B = load_model_data(prefix='qwen_1.7B')
m_8B = load_model_data(prefix=None)
m_32B = load_model_data(prefix='qwen_32B')
print("✓ All models loaded")

#%% Build Summary Table
rows = ['CoT (Baseline)', 'NoCoT', 'Blank Best', 'Blank Worst', 'Incorrect Best', 'Incorrect Worst']
keys = ['cot', 'nocot', 'blank_best', 'blank_worst', 'incorrect_best', 'incorrect_worst']

def get_vals(model_data, key):
    """Extract Q and S stats for a row key."""
    val = model_data[key]
    if val is None:
        return None, None
    return val['q'], val['s']

data = {'Experiment': rows}
for model_name, model_data in [('1.7B', m_1_7B), ('8B', m_8B), ('32B', m_32B)]:
    q_col, s_col = [], []
    for key in keys:
        q, s = get_vals(model_data, key)
        q_col.append(fmt(q))
        s_col.append(fmt(s))
    data[f'{model_name} Q'] = q_col
    data[f'{model_name} S'] = s_col

df = pd.DataFrame(data)
print(df.to_string(index=False))

#%% Create and Save Table Figure
fig, ax = plt.subplots(figsize=(18, 4))
ax.axis('off')
ax.set_title('BBQ Experiment Summary - Model Comparison', fontsize=14, fontweight='bold', pad=20)

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.16] + [0.14] * 6
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

# Style header
for j in range(len(df.columns)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Row colors
row_colors = {1: '#E2EFDA', 2: '#FCE4D6', 3: '#E2EFDA', 4: '#F8CBAD', 5: '#E2EFDA', 6: '#F8CBAD'}
for i in range(1, len(df) + 1):
    for j in range(len(df.columns)):
        if i in row_colors:
            table[(i, j)].set_facecolor(row_colors[i])

plt.tight_layout()

# Save
output_dir = '/workspace/experiments/exp_006_extend_thinking/bbq/results_graphs'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'compare_models_table.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {save_path}")

plt.show()


# %%
