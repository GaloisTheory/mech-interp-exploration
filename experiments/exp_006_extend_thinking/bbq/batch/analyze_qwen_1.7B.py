#!/usr/bin/env python3
"""BBQ Experiment Analysis for Qwen-1.7B

Visual analysis of CoT insertion experiments with:
- Summary table comparing CoT, NoCoT, best/worst experiments
- Line graphs showing accuracy vs tokens/insertions

Usage:
    Run cells interactively with #%% markers in VS Code/Cursor
"""

#%% Imports and Configuration
import os
import sys
import importlib.util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration
SAVE_FIGURES = False  # Toggle to save outputs
OUTPUT_DIR = '/workspace/experiments/exp_006_extend_thinking/bbq/batch/outputs/qwen_1.7B_graphs'
MODEL_PREFIX = 'qwen_1.7B'  # Prefix for experiment names

# Setup paths
_batch_dir = '/workspace/experiments/exp_006_extend_thinking/bbq/batch'
_bbq_dir = os.path.dirname(_batch_dir)

for path in [_batch_dir, _bbq_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Load batch_utils
_batch_utils_path = os.path.join(_batch_dir, 'batch_utils.py')
_spec = importlib.util.spec_from_file_location("batch_utils", _batch_utils_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load batch_utils from {_batch_utils_path}")
batch_utils = importlib.util.module_from_spec(_spec)
sys.modules["batch_utils"] = batch_utils
_spec.loader.exec_module(batch_utils)

# Load analyze_batch_utils
_analyze_utils_path = os.path.join(_batch_dir, 'analyze_batch_utils.py')
_spec2 = importlib.util.spec_from_file_location("analyze_batch_utils", _analyze_utils_path)
if _spec2 is None or _spec2.loader is None:
    raise ImportError(f"Could not load analyze_batch_utils from {_analyze_utils_path}")
analyze_utils = importlib.util.module_from_spec(_spec2)
sys.modules["analyze_batch_utils"] = analyze_utils
_spec2.loader.exec_module(analyze_utils)

# Import functions
load_batch_experiment = batch_utils.load_batch_experiment
load_baseline_combined = analyze_utils.load_baseline_combined
accuracy_by_question = analyze_utils.accuracy_by_question

print("✓ Imports loaded successfully")

#%% Load All Experiments
print("Loading all experiments...")

# Helper to load with prefix
def load_exp(name):
    """Load experiment with model prefix."""
    return load_batch_experiment(f'{MODEL_PREFIX}_{name}')

# Baselines
cot = load_baseline_combined(load_batch_experiment, prefix=MODEL_PREFIX)
nocot = load_exp('force_immediate')

# Blank space experiments (static)
blank_static = {}
blank_static_values = [5, 10, 50, 100, 500]
for n in blank_static_values:
    exp = load_exp(f'blank_static_{n}_')
    if exp:
        blank_static[n] = exp

# Blank space experiments (dynamic median)
blank_median = {}
for mult in [1, 2, 5]:
    exp = load_exp(f'blank_dynamic_median_{mult}x_')
    if exp:
        blank_median[mult] = exp

# Blank space experiments (dynamic max)
blank_max = {}
for mult in [1, 2, 5]:
    exp = load_exp(f'blank_dynamic_max_{mult}x_')
    if exp:
        blank_max[mult] = exp

# Incorrect answer experiments (static)
incorrect_static = {}
incorrect_static_values = [1, 2, 6, 12, 62]
for n in incorrect_static_values:
    exp = load_exp(f'incorrect_answer_{n}_')
    if exp:
        incorrect_static[n] = exp

# Incorrect answer experiments (dynamic median)
incorrect_median = {}
for mult in [1, 2, 5]:
    exp = load_exp(f'incorrect_dynamic_median_{mult}x_')
    if exp:
        incorrect_median[mult] = exp

# Incorrect answer experiments (dynamic max)
incorrect_max = {}
for mult in [1, 2, 5]:
    exp = load_exp(f'incorrect_dynamic_max_{mult}x_')
    if exp:
        incorrect_max[mult] = exp

print("\n✓ Done loading experiments")

#%% Compute Accuracy Stats for All Experiments
def get_accuracies(exp):
    """Get both question and sample accuracy for an experiment."""
    if not exp or not exp.get('results'):
        return None, None
    stats = accuracy_by_question(exp)
    return stats['overall_by_question'], stats['overall_by_sample']

# Baseline accuracies
cot_q, cot_s = get_accuracies(cot)
nocot_q, nocot_s = get_accuracies(nocot)

print(f"\nBaseline Accuracies:")
print(f"  CoT:   Question={cot_q:.1%}, Sample={cot_s:.1%}")
print(f"  NoCoT: Question={nocot_q:.1%}, Sample={nocot_s:.1%}")

# Collect all blank experiment accuracies
all_blank = {}
for n, exp in blank_static.items():
    q, s = get_accuracies(exp)
    if q is not None:
        all_blank[f'static_{n}'] = {'q': q, 's': s, 'exp': exp}

for mult, exp in blank_median.items():
    q, s = get_accuracies(exp)
    if q is not None:
        all_blank[f'median_{mult}x'] = {'q': q, 's': s, 'exp': exp}

for mult, exp in blank_max.items():
    q, s = get_accuracies(exp)
    if q is not None:
        all_blank[f'max_{mult}x'] = {'q': q, 's': s, 'exp': exp}

# Collect all incorrect experiment accuracies
all_incorrect = {}
for n, exp in incorrect_static.items():
    q, s = get_accuracies(exp)
    if q is not None:
        all_incorrect[f'static_{n}'] = {'q': q, 's': s, 'exp': exp}

for mult, exp in incorrect_median.items():
    q, s = get_accuracies(exp)
    if q is not None:
        all_incorrect[f'median_{mult}x'] = {'q': q, 's': s, 'exp': exp}

for mult, exp in incorrect_max.items():
    q, s = get_accuracies(exp)
    if q is not None:
        all_incorrect[f'max_{mult}x'] = {'q': q, 's': s, 'exp': exp}

# Find best/worst for blank (by sample accuracy)
if all_blank:
    blank_best_key = max(all_blank.keys(), key=lambda k: all_blank[k]['s'])
    blank_worst_key = min(all_blank.keys(), key=lambda k: all_blank[k]['s'])
    blank_best = all_blank[blank_best_key]
    blank_worst = all_blank[blank_worst_key]
    print(f"\nBlank Space:")
    print(f"  Best:  {blank_best_key} (Q={blank_best['q']:.1%}, S={blank_best['s']:.1%})")
    print(f"  Worst: {blank_worst_key} (Q={blank_worst['q']:.1%}, S={blank_worst['s']:.1%})")
else:
    blank_best = blank_worst = None
    blank_best_key = blank_worst_key = "N/A"

# Find best/worst for incorrect (by sample accuracy)
if all_incorrect:
    incorrect_best_key = max(all_incorrect.keys(), key=lambda k: all_incorrect[k]['s'])
    incorrect_worst_key = min(all_incorrect.keys(), key=lambda k: all_incorrect[k]['s'])
    incorrect_best = all_incorrect[incorrect_best_key]
    incorrect_worst = all_incorrect[incorrect_worst_key]
    print(f"\nIncorrect Answer:")
    print(f"  Best:  {incorrect_best_key} (Q={incorrect_best['q']:.1%}, S={incorrect_best['s']:.1%})")
    print(f"  Worst: {incorrect_worst_key} (Q={incorrect_worst['q']:.1%}, S={incorrect_worst['s']:.1%})")
else:
    incorrect_best = incorrect_worst = None
    incorrect_best_key = incorrect_worst_key = "N/A"

#%% Summary Table (Visual)
# Build summary data
summary_data = {
    'Experiment': ['CoT (Baseline)', 'NoCoT', 
                   f'Blank Best ({blank_best_key})', f'Blank Worst ({blank_worst_key})',
                   f'Incorrect Best ({incorrect_best_key})', f'Incorrect Worst ({incorrect_worst_key})'],
    'Question Acc': [
        f'{cot_q:.1%}' if cot_q else 'N/A',
        f'{nocot_q:.1%}' if nocot_q else 'N/A',
        f'{blank_best["q"]:.1%}' if blank_best else 'N/A',
        f'{blank_worst["q"]:.1%}' if blank_worst else 'N/A',
        f'{incorrect_best["q"]:.1%}' if incorrect_best else 'N/A',
        f'{incorrect_worst["q"]:.1%}' if incorrect_worst else 'N/A',
    ],
    'Sample Acc': [
        f'{cot_s:.1%}' if cot_s else 'N/A',
        f'{nocot_s:.1%}' if nocot_s else 'N/A',
        f'{blank_best["s"]:.1%}' if blank_best else 'N/A',
        f'{blank_worst["s"]:.1%}' if blank_worst else 'N/A',
        f'{incorrect_best["s"]:.1%}' if incorrect_best else 'N/A',
        f'{incorrect_worst["s"]:.1%}' if incorrect_worst else 'N/A',
    ],
}

df_summary = pd.DataFrame(summary_data)

# Create figure for table
fig_table, ax_table = plt.subplots(figsize=(10, 4))
ax_table.axis('off')
ax_table.set_title('BBQ Experiment Summary - Qwen-1.7B', fontsize=14, fontweight='bold', pad=20)

# Create table
table = ax_table.table(
    cellText=df_summary.values,
    colLabels=df_summary.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.45, 0.25, 0.25]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color header row
for j in range(len(df_summary.columns)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Color code rows based on experiment type
row_colors = {
    1: '#E2EFDA',  # CoT - light green
    2: '#FCE4D6',  # NoCoT - light orange
    3: '#E2EFDA',  # Blank best - light green
    4: '#F8CBAD',  # Blank worst - light red/orange
    5: '#E2EFDA',  # Incorrect best - light green
    6: '#F8CBAD',  # Incorrect worst - light red/orange
}

for i in range(1, len(df_summary) + 1):
    for j in range(len(df_summary.columns)):
        if i in row_colors:
            table[(i, j)].set_facecolor(row_colors[i])

plt.tight_layout()

if SAVE_FIGURES:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, 'qwen_1.7B_summary_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")

plt.show()

#%% Detailed Category Breakdown Table (MultiIndex)
def get_category_accuracies(exp):
    """Get per-category question and sample accuracy."""
    if not exp or not exp.get('results'):
        return {}, {}
    stats = accuracy_by_question(exp)
    return stats['by_category_question'], stats['by_category_sample']

def build_detailed_table(experiments_dict):
    """
    Build a detailed DataFrame with MultiIndex columns.
    
    Args:
        experiments_dict: Dict of {name: experiment_data}
        
    Returns:
        DataFrame with categories as rows, (Experiment, Metric) as columns
    """
    # Collect all categories
    all_categories = set()
    for exp in experiments_dict.values():
        if exp and exp.get('results'):
            for r in exp['results']:
                all_categories.add(r.get('category', 'unknown'))
    
    categories = sorted(all_categories)
    
    # Build data for MultiIndex DataFrame
    data = {}
    for exp_name, exp in experiments_dict.items():
        q_acc, s_acc = get_category_accuracies(exp)
        
        # Question accuracy column
        data[(exp_name, 'Q%')] = [q_acc.get(cat, None) for cat in categories]
        # Sample accuracy column
        data[(exp_name, 'S%')] = [s_acc.get(cat, None) for cat in categories]
    
    # Create MultiIndex DataFrame
    df = pd.DataFrame(data, index=categories)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    
    # Add overall row
    overall_data = {}
    for exp_name, exp in experiments_dict.items():
        q, s = get_accuracies(exp)
        overall_data[(exp_name, 'Q%')] = q
        overall_data[(exp_name, 'S%')] = s
    
    overall_row = pd.DataFrame([overall_data], index=['OVERALL'])
    overall_row.columns = pd.MultiIndex.from_tuples(overall_row.columns)
    
    df = pd.concat([df, overall_row])
    
    return df

# Build experiments dict for detailed table - SAME as summary table
detailed_experiments = {
    'CoT': cot,
    'NoCoT': nocot,
    f'Blank Best': blank_best['exp'] if blank_best else None,
    f'Blank Worst': blank_worst['exp'] if blank_worst else None,
    f'Inc Best': incorrect_best['exp'] if incorrect_best else None,
    f'Inc Worst': incorrect_worst['exp'] if incorrect_worst else None,
}

# Remove None entries
detailed_experiments = {k: v for k, v in detailed_experiments.items() if v is not None}

df_detailed = build_detailed_table(detailed_experiments)

#%% Detailed Table - Visual (matplotlib)
def plot_detailed_table(df, exp_labels, title="Detailed Category Breakdown", save=False, filename=None):
    """
    Render the detailed DataFrame as a styled matplotlib table.
    
    Args:
        df: DataFrame with MultiIndex columns (Experiment, Q%/S%)
        exp_labels: Dict mapping short names to display labels
        title: Table title
        save: Whether to save the figure
        filename: Filename if saving
    """
    # Format for display
    df_fmt = df.copy()
    for col in df_fmt.columns:
        df_fmt[col] = df_fmt[col].apply(lambda x: f'{x:.0%}' if pd.notna(x) else '-')
    
    # Get unique experiment names in order
    exp_names = []
    seen = set()
    for col in df_fmt.columns:
        if col[0] not in seen:
            exp_names.append(col[0])
            seen.add(col[0])
    
    n_experiments = len(exp_names)
    n_categories = len(df_fmt)
    
    # Figure dimensions
    fig_width = 15
    fig_height = max(9, n_categories * 0.5 + 3)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title(f'{title}\nQ% = Question Accuracy, S% = Sample Accuracy', 
                 fontsize=12, fontweight='bold', pad=10)
    
    # Build single header row with combined "Exp\nQ%" and "Exp\nS%" labels
    col_labels = ['Category']
    for exp in exp_names:
        label = exp_labels.get(exp, exp).replace('\n', ' ')
        # Shorten long labels
        if len(label) > 15:
            label = label[:14] + '…'
        col_labels.append(f'{label}\nQ%')
        col_labels.append(f'{label}\nS%')
    
    # Data rows
    cell_text = []
    for idx in df_fmt.index:
        row = [idx]
        for exp in exp_names:
            row.append(df_fmt.loc[idx, (exp, 'Q%')])
            row.append(df_fmt.loc[idx, (exp, 'S%')])
        cell_text.append(row)
    
    # Column widths
    cat_width = 0.13
    val_width = 0.058
    col_widths = [cat_width] + [val_width] * (n_experiments * 2)
    
    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    
    n_cols = len(col_labels)
    n_data_rows = len(cell_text)
    
    # Style header row (row 0 in table with colLabels)
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        
        # Color based on experiment type (derived from column label)
        label = col_labels[j]
        if j == 0:
            cell.set_facecolor('#4472C4')  # Category header
        elif 'Best' in label:
            cell.set_facecolor('#2E7D32')  # Dark green for best
        elif 'Worst' in label:
            cell.set_facecolor('#C65911')  # Dark orange for worst
        elif 'CoT' in label and 'NoCoT' not in label:
            cell.set_facecolor('#1F4E79')  # Dark blue for CoT
        elif 'NoCoT' in label:
            cell.set_facecolor('#7F6000')  # Dark yellow/brown for NoCoT
        else:
            cell.set_facecolor('#4472C4')
    
    # Style data rows
    # Row indices: 0 = header, 1 to n_data_rows = data rows
    overall_row_idx = n_data_rows  # Last data row (OVERALL)
    
    for i in range(1, n_data_rows + 1):
        is_overall = (i == overall_row_idx)
        base_color = '#D6DCE4' if is_overall else ('#F2F2F2' if i % 2 == 0 else 'white')
        
        # Category column
        table[(i, 0)].set_facecolor(base_color)
        if is_overall:
            table[(i, 0)].set_text_props(fontweight='bold')
        
        # Data columns - color by experiment type
        col_idx = 1
        for exp in exp_names:
            if is_overall:
                color = '#D6DCE4'
            elif 'Best' in exp:
                color = '#E2EFDA'  # Light green
            elif 'Worst' in exp:
                color = '#FCE4D6'  # Light orange
            elif exp == 'CoT':
                color = '#DEEBF7'  # Light blue
            elif exp == 'NoCoT':
                color = '#FFF2CC'  # Light yellow
            else:
                color = base_color
            
            table[(i, col_idx)].set_facecolor(color)
            table[(i, col_idx + 1)].set_facecolor(color)
            if is_overall:
                table[(i, col_idx)].set_text_props(fontweight='bold')
                table[(i, col_idx + 1)].set_text_props(fontweight='bold')
            col_idx += 2
    
    plt.tight_layout()
    
    if save and filename:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    return fig

# Create experiment labels with detail about which specific experiment is best/worst
exp_labels = {
    'CoT': 'CoT (Baseline)',
    'NoCoT': 'NoCoT',
    'Blank Best': f'Blank Best\n({blank_best_key})',
    'Blank Worst': f'Blank Worst\n({blank_worst_key})',
    'Inc Best': f'Inc Best\n({incorrect_best_key})',
    'Inc Worst': f'Inc Worst\n({incorrect_worst_key})',
}

fig_detailed = plot_detailed_table(
    df_detailed,
    exp_labels=exp_labels,
    title="BBQ Detailed Category Breakdown - Qwen-1.7B",
    save=SAVE_FIGURES,
    filename='qwen_1.7B_detailed_table.png'
)

#%% Full Detailed Table (All Experiments)
# Include more experiments for comprehensive view
full_experiments = {
    'CoT': cot,
    'NoCoT': nocot,
}

# All blank static
for n in blank_static_values:
    if n in blank_static:
        full_experiments[f'B{n}'] = blank_static[n]

# All incorrect static  
for n in incorrect_static_values:
    if n in incorrect_static:
        full_experiments[f'I{n}'] = incorrect_static[n]

df_full = build_detailed_table(full_experiments)

# Display as formatted string
df_full_display = df_full.copy()
for col in df_full_display.columns:
    df_full_display[col] = df_full_display[col].apply(lambda x: f'{x:.0%}' if pd.notna(x) else '-')

print("\n" + "="*120)
print("FULL EXPERIMENT COMPARISON - All Static Experiments")
print("="*120)
print("B = Blank spaces, I = Incorrect answer repetitions")
print("Q% = Question Accuracy, S% = Sample Accuracy")
print()
print(df_full_display.to_string())

# Also display the DataFrame directly (works nicely in Jupyter/IPython)
df_full_display

#%% Line Graphs - Helper Function
def plot_accuracy_grid(accuracy_type='sample', save=False):
    """
    Create 2x3 grid of accuracy plots.
    
    Args:
        accuracy_type: 'sample' or 'question'
        save: Whether to save the figure
    """
    acc_key = 's' if accuracy_type == 'sample' else 'q'
    cot_acc = cot_s if accuracy_type == 'sample' else cot_q
    nocot_acc = nocot_s if accuracy_type == 'sample' else nocot_q
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'BBQ {accuracy_type.title()} Accuracy vs. Token/Insertion Count - Qwen-1.7B', 
                 fontsize=14, fontweight='bold')
    
    # Define plot configurations
    plot_configs = [
        # Row 0: Blank experiments
        {
            'ax': axes[0, 0],
            'title': 'Blank Space - Static',
            'data': blank_static,
            'x_values': blank_static_values,
            'x_label': 'Number of Blank Spaces',
        },
        {
            'ax': axes[0, 1],
            'title': 'Blank Space - Median Dynamic',
            'data': blank_median,
            'x_values': [1, 2, 5],
            'x_label': 'Multiplier (×median tokens)',
            'x_ticks': ['1x', '2x', '5x'],
        },
        {
            'ax': axes[0, 2],
            'title': 'Blank Space - Max Dynamic',
            'data': blank_max,
            'x_values': [1, 2, 5],
            'x_label': 'Multiplier (×max tokens)',
            'x_ticks': ['1x', '2x', '5x'],
        },
        # Row 1: Incorrect experiments
        {
            'ax': axes[1, 0],
            'title': 'Incorrect Answer - Static',
            'data': incorrect_static,
            'x_values': incorrect_static_values,
            'x_label': 'Number of Repetitions',
        },
        {
            'ax': axes[1, 1],
            'title': 'Incorrect Answer - Median Dynamic',
            'data': incorrect_median,
            'x_values': [1, 2, 5],
            'x_label': 'Multiplier (×median tokens)',
            'x_ticks': ['1x', '2x', '5x'],
        },
        {
            'ax': axes[1, 2],
            'title': 'Incorrect Answer - Max Dynamic',
            'data': incorrect_max,
            'x_values': [1, 2, 5],
            'x_label': 'Multiplier (×max tokens)',
            'x_ticks': ['1x', '2x', '5x'],
        },
    ]
    
    for config in plot_configs:
        ax = config['ax']
        data = config['data']
        x_values = config['x_values']
        
        # Get accuracy values
        y_values = []
        valid_x = []
        for x in x_values:
            if x in data:
                exp = data[x]
                q, s = get_accuracies(exp)
                acc = s if accuracy_type == 'sample' else q
                if acc is not None:
                    y_values.append(acc * 100)
                    valid_x.append(x)
        
        # Plot experiment data
        if valid_x:
            ax.plot(valid_x, y_values, 'b-o', linewidth=2, markersize=8, label='Experiment')
        
        # Plot CoT reference line
        if cot_acc is not None:
            ax.axhline(y=cot_acc * 100, color='red', linestyle='--', linewidth=2, label='CoT')
        
        # Plot NoCoT reference line
        if nocot_acc is not None:
            ax.axhline(y=nocot_acc * 100, color='orange', linestyle='--', linewidth=2, label='NoCoT')
        
        # Configure axes
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel(config['x_label'], fontsize=10)
        ax.set_ylabel(f'{accuracy_type.title()} Accuracy (%)', fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        
        # Custom x-ticks if specified
        if 'x_ticks' in config and valid_x:
            ax.set_xticks(valid_x)
            ax.set_xticklabels(config['x_ticks'])
        
        # Use log scale for static plots with wide range
        if 'Static' in config['title'] and len(valid_x) > 2:
            ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f'qwen_1.7B_{accuracy_type}_accuracy.png'
        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    return fig

#%% Sample Accuracy Graphs
fig_sample = plot_accuracy_grid(accuracy_type='sample', save=SAVE_FIGURES)

#%% Question Accuracy Graphs
fig_question = plot_accuracy_grid(accuracy_type='question', save=SAVE_FIGURES)

#%% Save All Figures (if not already saved)
if SAVE_FIGURES:
    print("\n✓ All figures saved to:", OUTPUT_DIR)

