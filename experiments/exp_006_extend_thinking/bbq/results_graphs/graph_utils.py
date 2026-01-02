"""BBQ Graph Utilities

Simplified utilities for generating graphs for the MATS write-up.
Provides clean functions for:
- Loading experiments by model
- Computing accuracy (with optional category filtering)
- Plotting single graphs (one x-axis, one y-axis per function)
- Interactive exploration: find divergent questions, inspect CoT

Usage (Graphs):
    from graph_utils import plot_blank_static_accuracy, plot_incorrect_static_accuracy, plot_summary_table
    
    plot_blank_static_accuracy(model='8B', save=True)
    plot_incorrect_static_accuracy(model='8B', save=True)
    plot_summary_table(model='8B', save=True)

Usage (Exploration):
    from graph_utils import load_experiment, find_divergent, inspect_question, show_cot
    
    exp1 = load_experiment('8B', 'incorrect', 12)
    exp2 = load_experiment('32B', 'incorrect', 12)
    div = find_divergent(exp1, exp2, '8B', '32B')
    inspect_question(exp1, div['exp1_better'][0]['idx'])
    show_cot(exp1, div['exp1_better'][0]['idx'], sample=0)
"""

import os
import sys
import json
import importlib.util
from glob import glob
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Setup paths - handle interactive sessions where __file__ may not exist
if '__file__' in dir():
    _results_dir = os.path.dirname(os.path.abspath(__file__))
else:
    # Fallback for interactive use - assume cwd is results_graphs/
    _results_dir = os.path.abspath(os.getcwd())
_batch_dir = os.path.join(os.path.dirname(_results_dir), 'batch')
_output_dir = os.path.join(_batch_dir, 'outputs')

if _batch_dir not in sys.path:
    sys.path.insert(0, _batch_dir)

# Load batch_utils
_spec = importlib.util.spec_from_file_location("batch_utils", os.path.join(_batch_dir, 'batch_utils.py'))
_batch_utils = importlib.util.module_from_spec(_spec)
sys.modules["batch_utils"] = _batch_utils
_spec.loader.exec_module(_batch_utils)

# Load analyze_batch_utils
_spec2 = importlib.util.spec_from_file_location("analyze_batch_utils", os.path.join(_batch_dir, 'analyze_batch_utils.py'))
_analyze_utils = importlib.util.module_from_spec(_spec2)
sys.modules["analyze_batch_utils"] = _analyze_utils
_spec2.loader.exec_module(_analyze_utils)


# =============================================================================
# INTERNAL LOADERS
# =============================================================================

def _get_prefix(model: str) -> Optional[str]:
    """Get folder prefix for a model. 8B has no prefix, others use qwen_{model}_."""
    if model == '8B':
        return None
    return f'qwen_{model}'


def _load_experiment(name: str, prefix: Optional[str]) -> Optional[dict]:
    """Load an experiment by name with optional prefix."""
    folders = [f for f in os.listdir(_output_dir) if os.path.isdir(os.path.join(_output_dir, f))]
    
    if prefix:
        # Match prefix_name pattern
        pattern = f'{prefix}_{name}'.lower()
        matches = [f for f in folders if f.lower().startswith(pattern)]
    else:
        # 8B: match name but exclude qwen_* folders
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


def load_experiments(model: str = '8B') -> dict:
    """
    Load all relevant experiments for a model.
    
    Args:
        model: Model size ('8B', '32B', '1.7B')
        
    Returns:
        Dict with keys: 'cot', 'nocot', 'blank_static', 'incorrect_static' (each a dict of {n: exp})
    """
    prefix = _get_prefix(model)
    
    # Load baselines
    cot = _analyze_utils.load_baseline_combined(_batch_utils.load_batch_experiment, prefix=prefix)
    nocot = _load_experiment('force_immediate', prefix)
    
    # Load blank static experiments
    blank_static = {}
    for n in [5, 10, 50, 100, 500]:
        exp = _load_experiment(f'blank_static_{n}_', prefix)
        if exp:
            blank_static[n] = exp
    
    # Load incorrect answer static experiments
    incorrect_static = {}
    for n in [1, 2, 6, 12, 62]:
        exp = _load_experiment(f'incorrect_answer_{n}_', prefix)
        if exp:
            incorrect_static[n] = exp
    
    return {
        'cot': cot,
        'nocot': nocot,
        'blank_static': blank_static,
        'incorrect_static': incorrect_static,
    }


# =============================================================================
# ACCURACY COMPUTATION
# =============================================================================

def get_accuracy(
    exp: dict,
    categories: Optional[List[str]] = None,
    accuracy_type: str = 'sample'
) -> Optional[float]:
    """
    Compute accuracy for an experiment, optionally filtered by categories.
    
    Args:
        exp: Experiment data dict
        categories: Optional list of categories to include (e.g., ['age', 'appearance'])
        accuracy_type: 'sample' or 'question'
        
    Returns:
        Accuracy as float (0.0 to 1.0), or None if no data
    """
    if not exp or not exp.get('results'):
        return None
    
    results = exp.get('results', [])
    
    # Filter by categories if specified
    if categories:
        categories_lower = [c.lower() for c in categories]
        results = [r for r in results if r.get('category', '').lower() in categories_lower]
    
    if not results:
        return None
    
    if accuracy_type == 'question':
        # Question-level: majority vote correct (ties = incorrect)
        correct = 0
        for r in results:
            answer_dist = r.get('answer_distribution', {})
            if answer_dist:
                correct_answer = r.get('correct_answer', '')
                max_count = max(answer_dist.values())
                answers_with_max = [a for a, c in answer_dist.items() if c == max_count]
                if len(answers_with_max) == 1 and answers_with_max[0] == correct_answer:
                    correct += 1
        return correct / len(results)
    else:
        # Sample-level
        total = correct = 0
        for r in results:
            for s in r.get('samples', []):
                total += 1
                if s.get('correct'):
                    correct += 1
        return correct / total if total > 0 else None


def get_category_accuracy(
    exp: dict,
    accuracy_type: str = 'sample'
) -> Dict[str, float]:
    """
    Compute per-category accuracy for an experiment.
    
    Args:
        exp: Experiment data dict
        accuracy_type: 'sample' or 'question'
        
    Returns:
        Dict mapping category -> accuracy
    """
    if not exp or not exp.get('results'):
        return {}
    
    results = exp.get('results', [])
    by_cat: Dict[str, Dict[str, int]] = {}
    
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in by_cat:
            by_cat[cat] = {'correct': 0, 'total': 0}
        
        if accuracy_type == 'question':
            by_cat[cat]['total'] += 1
            answer_dist = r.get('answer_distribution', {})
            if answer_dist:
                correct_answer = r.get('correct_answer', '')
                max_count = max(answer_dist.values())
                answers_with_max = [a for a, c in answer_dist.items() if c == max_count]
                if len(answers_with_max) == 1 and answers_with_max[0] == correct_answer:
                    by_cat[cat]['correct'] += 1
        else:
            for s in r.get('samples', []):
                by_cat[cat]['total'] += 1
                if s.get('correct'):
                    by_cat[cat]['correct'] += 1
    
    return {
        cat: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        for cat, stats in by_cat.items()
    }


# =============================================================================
# PLOTTING FUNCTIONS (Single Graph Each)
# =============================================================================

def plot_blank_static_accuracy(
    model: str = '8B',
    categories: Optional[List[str]] = None,
    accuracy_type: str = 'sample',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Plot accuracy vs number of blank spaces (static experiments only).
    
    Single line graph with CoT/NoCoT reference lines.
    
    Args:
        model: Model size ('8B', '32B', '1.7B')
        categories: Optional list of categories to filter (e.g., ['age', 'appearance'])
        accuracy_type: 'sample' or 'question'
        save: Whether to save the figure
        output_dir: Directory to save figure (defaults to results_graphs/)
        
    Returns:
        matplotlib Figure
    """
    # Load data
    data = load_experiments(model)
    blank_static = data['blank_static']
    
    # Compute accuracies
    cot_acc = get_accuracy(data['cot'], categories, accuracy_type)
    nocot_acc = get_accuracy(data['nocot'], categories, accuracy_type)
    
    x_values = sorted(blank_static.keys())
    y_values = []
    valid_x = []
    
    for x in x_values:
        acc = get_accuracy(blank_static[x], categories, accuracy_type)
        if acc is not None:
            y_values.append(acc * 100)
            valid_x.append(x)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Build title
    cat_suffix = f" ({', '.join(categories)})" if categories else ""
    title = f'Blank Space (Static) - {model}{cat_suffix}'
    
    # Plot experiment data
    if valid_x:
        ax.plot(valid_x, y_values, 'b-o', linewidth=2, markersize=8, label='Experiment')
    
    # Plot reference lines
    if cot_acc is not None:
        ax.axhline(y=cot_acc * 100, color='green', linestyle='--', linewidth=2, label='CoT')
    if nocot_acc is not None:
        ax.axhline(y=nocot_acc * 100, color='red', linestyle='--', linewidth=2, label='NoCoT')
    
    # Configure axes
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Blank Spaces', fontsize=10)
    ax.set_ylabel(f'{accuracy_type.title()} Accuracy (%)', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        if output_dir is None:
            output_dir = _results_dir
        os.makedirs(output_dir, exist_ok=True)
        
        cat_tag = '_' + '_'.join(categories) if categories else '_all_categories'
        filename = f'{model}_blank_static{cat_tag}.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_incorrect_static_accuracy(
    model: str = '8B',
    categories: Optional[List[str]] = None,
    accuracy_type: str = 'sample',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Plot accuracy vs number of incorrect answer repetitions (static experiments only).
    
    Single line graph with CoT/NoCoT reference lines.
    
    Args:
        model: Model size ('8B', '32B', '1.7B')
        categories: Optional list of categories to filter (e.g., ['age', 'appearance'])
        accuracy_type: 'sample' or 'question'
        save: Whether to save the figure
        output_dir: Directory to save figure (defaults to results_graphs/)
        
    Returns:
        matplotlib Figure
    """
    # Load data
    data = load_experiments(model)
    incorrect_static = data['incorrect_static']
    
    # Compute accuracies
    cot_acc = get_accuracy(data['cot'], categories, accuracy_type)
    nocot_acc = get_accuracy(data['nocot'], categories, accuracy_type)
    
    x_values = sorted(incorrect_static.keys())
    y_values = []
    valid_x = []
    
    for x in x_values:
        acc = get_accuracy(incorrect_static[x], categories, accuracy_type)
        if acc is not None:
            y_values.append(acc * 100)
            valid_x.append(x)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Build title
    cat_suffix = f" ({', '.join(categories)})" if categories else ""
    title = f'Incorrect Answer (Static) - {model}{cat_suffix}'
    
    # Plot experiment data
    if valid_x:
        ax.plot(valid_x, y_values, 'b-o', linewidth=2, markersize=8, label='Experiment')
    
    # Plot reference lines
    if cot_acc is not None:
        ax.axhline(y=cot_acc * 100, color='green', linestyle='--', linewidth=2, label='CoT')
    if nocot_acc is not None:
        ax.axhline(y=nocot_acc * 100, color='red', linestyle='--', linewidth=2, label='NoCoT')
    
    # Configure axes
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Incorrect Answer Repetitions', fontsize=10)
    ax.set_ylabel(f'{accuracy_type.title()} Accuracy (%)', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        if output_dir is None:
            output_dir = _results_dir
        os.makedirs(output_dir, exist_ok=True)
        
        cat_tag = '_' + '_'.join(categories) if categories else '_all_categories'
        filename = f'{model}_incorrect_static{cat_tag}.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_summary_table(
    model: str = '8B',
    categories: Optional[List[str]] = None,
    save: bool = False,
    output_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Plot summary table showing category breakdown across experiments.
    
    Args:
        model: Model size ('8B', '32B', '1.7B')
        categories: Optional list of categories to include
        save: Whether to save the figure
        output_dir: Directory to save figure (defaults to results_graphs/)
        
    Returns:
        matplotlib Figure
    """
    # Load data
    data = load_experiments(model)
    
    # Build experiments dict
    experiments = {
        'CoT': data['cot'],
        'NoCoT': data['nocot'],
    }
    
    # Add blank static experiments
    for n, exp in sorted(data['blank_static'].items()):
        experiments[f'B{n}'] = exp
    
    # Get all categories from experiments
    all_cats = set()
    for exp in experiments.values():
        if exp and exp.get('results'):
            for r in exp['results']:
                all_cats.add(r.get('category', 'unknown'))
    
    # Filter categories if specified
    if categories:
        categories_lower = [c.lower() for c in categories]
        all_cats = {c for c in all_cats if c.lower() in categories_lower}
    
    cats = sorted(all_cats)
    
    # Build data for table
    rows = []
    for cat in cats:
        row = [cat]
        for exp_name, exp in experiments.items():
            q_acc = get_accuracy(exp, [cat], 'question')
            s_acc = get_accuracy(exp, [cat], 'sample')
            row.append(f'{q_acc:.0%}' if q_acc is not None else '-')
            row.append(f'{s_acc:.0%}' if s_acc is not None else '-')
        rows.append(row)
    
    # Add overall row
    overall_row = ['OVERALL']
    for exp_name, exp in experiments.items():
        q_acc = get_accuracy(exp, list(all_cats) if categories else None, 'question')
        s_acc = get_accuracy(exp, list(all_cats) if categories else None, 'sample')
        overall_row.append(f'{q_acc:.0%}' if q_acc is not None else '-')
        overall_row.append(f'{s_acc:.0%}' if s_acc is not None else '-')
    rows.append(overall_row)
    
    # Build column labels
    col_labels = ['Category']
    for exp_name in experiments.keys():
        col_labels.append(f'{exp_name}\nQ%')
        col_labels.append(f'{exp_name}\nS%')
    
    # Create figure
    n_cols = len(col_labels)
    n_rows = len(rows)
    fig_width = max(12, n_cols * 0.8)
    fig_height = max(6, n_rows * 0.4 + 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    
    cat_suffix = f" ({', '.join(categories)})" if categories else ""
    ax.set_title(f'BBQ Category Breakdown - {model}{cat_suffix}\nQ% = Question Acc, S% = Sample Acc',
                 fontsize=12, fontweight='bold', pad=10)
    
    # Column widths
    cat_width = 0.12
    val_width = 0.045
    col_widths = [cat_width] + [val_width] * (n_cols - 1)
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    
    # Style header row
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold', fontsize=7)
    
    # Style data rows
    for i in range(1, n_rows + 1):
        is_overall = (i == n_rows)
        base_color = '#D6DCE4' if is_overall else ('#F2F2F2' if i % 2 == 0 else 'white')
        
        for j in range(n_cols):
            table[(i, j)].set_facecolor(base_color)
            if is_overall:
                table[(i, j)].set_text_props(fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        if output_dir is None:
            output_dir = _results_dir
        os.makedirs(output_dir, exist_ok=True)
        
        cat_tag = '_' + '_'.join(categories) if categories else ''
        filename = f'{model}_static_summary_table{cat_tag}.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# EXPLORATION FUNCTIONS
# =============================================================================

def load_experiment(
    model: str,
    exp_type: str,
    n: Optional[int] = None,
) -> Optional[dict]:
    """
    Load a specific experiment by model, type, and parameter.
    
    Args:
        model: Model size ('8B', '32B', '1.7B')
        exp_type: Experiment type:
            - 'cot': CoT baseline (combined across categories)
            - 'nocot': Force immediate (no CoT)
            - 'blank': Blank space static (requires n)
            - 'incorrect': Incorrect answer static (requires n)
        n: Parameter value (required for 'blank' and 'incorrect')
        
    Returns:
        Experiment dict with 'results', 'config', etc.
        
    Examples:
        load_experiment('8B', 'cot')
        load_experiment('32B', 'incorrect', 12)
        load_experiment('8B', 'blank', 100)
    """
    prefix = _get_prefix(model)
    
    if exp_type == 'cot':
        exp = _analyze_utils.load_baseline_combined(_batch_utils.load_batch_experiment, prefix=prefix)
        if exp:
            exp['_model'] = model
            exp['_type'] = 'cot'
        return exp
    
    elif exp_type == 'nocot':
        exp = _load_experiment('force_immediate', prefix)
        if exp:
            exp['_model'] = model
            exp['_type'] = 'nocot'
        return exp
    
    elif exp_type == 'blank':
        if n is None:
            raise ValueError("n is required for 'blank' experiment type")
        exp = _load_experiment(f'blank_static_{n}_', prefix)
        if exp:
            exp['_model'] = model
            exp['_type'] = f'blank_{n}'
        return exp
    
    elif exp_type == 'incorrect':
        if n is None:
            raise ValueError("n is required for 'incorrect' experiment type")
        exp = _load_experiment(f'incorrect_answer_{n}_', prefix)
        if exp:
            exp['_model'] = model
            exp['_type'] = f'incorrect_{n}'
        return exp
    
    else:
        raise ValueError(f"Unknown exp_type: {exp_type}. Use 'cot', 'nocot', 'blank', or 'incorrect'")


def _get_majority_answer(answer_dist: dict) -> Optional[str]:
    """Get majority answer from distribution, or None if tie."""
    if not answer_dist:
        return None
    max_count = max(answer_dist.values())
    answers_with_max = [a for a, c in answer_dist.items() if c == max_count]
    return answers_with_max[0] if len(answers_with_max) == 1 else None


def find_divergent(
    exp1: dict,
    exp2: dict,
    name1: str = 'exp1',
    name2: str = 'exp2',
) -> dict:
    """
    Find questions where experiments disagree (one correct, other wrong).
    
    Matches questions by (category, question_text) so works across models.
    
    Args:
        exp1: First experiment
        exp2: Second experiment
        name1: Name for first experiment (for display)
        name2: Name for second experiment (for display)
        
    Returns:
        Dict with:
            - 'exp1_better': List of questions where exp1 correct, exp2 wrong
            - 'exp2_better': List of questions where exp2 correct, exp1 wrong
            - 'both_correct': Count of questions both got right
            - 'both_wrong': Count of questions both got wrong
            
        Each item in exp1_better/exp2_better contains:
            - 'idx': Index in that experiment's results
            - 'category': Question category
            - 'question': Question text (truncated)
            - 'correct_answer': The correct answer
            - 'exp1_answer': What exp1 answered
            - 'exp2_answer': What exp2 answered
    """
    if not exp1 or not exp2:
        print("❌ One or both experiments are empty")
        return {'exp1_better': [], 'exp2_better': [], 'both_correct': 0, 'both_wrong': 0}
    
    # Build lookup for exp1: (category, question) -> (idx, is_correct, majority_answer)
    exp1_lookup = {}
    for idx, r in enumerate(exp1.get('results', [])):
        key = (r.get('category', '').lower(), r.get('question', ''))
        majority = _get_majority_answer(r.get('answer_distribution', {}))
        correct = r.get('correct_answer', '')
        is_correct = majority == correct if majority else False
        exp1_lookup[key] = {
            'idx': idx,
            'is_correct': is_correct,
            'answer': majority,
            'correct_answer': correct,
            'category': r.get('category', ''),
            'question': r.get('question', ''),
        }
    
    exp1_better = []
    exp2_better = []
    both_correct = 0
    both_wrong = 0
    
    for idx, r in enumerate(exp2.get('results', [])):
        key = (r.get('category', '').lower(), r.get('question', ''))
        
        if key not in exp1_lookup:
            continue
        
        exp1_data = exp1_lookup[key]
        
        # Get exp2's answer
        majority = _get_majority_answer(r.get('answer_distribution', {}))
        correct = r.get('correct_answer', '')
        exp2_correct = majority == correct if majority else False
        
        item = {
            'idx': exp1_data['idx'],  # Index in exp1 (for inspect_question)
            'idx_exp2': idx,  # Index in exp2
            'category': exp1_data['category'],
            'question': exp1_data['question'][:80] + '...' if len(exp1_data['question']) > 80 else exp1_data['question'],
            'question_full': exp1_data['question'],
            'correct_answer': correct,
            f'{name1}_answer': exp1_data['answer'],
            f'{name2}_answer': majority,
        }
        
        if exp1_data['is_correct'] and not exp2_correct:
            exp1_better.append(item)
        elif exp2_correct and not exp1_data['is_correct']:
            exp2_better.append(item)
        elif exp1_data['is_correct'] and exp2_correct:
            both_correct += 1
        else:
            both_wrong += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DIVERGENT QUESTIONS: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"  {name1} correct, {name2} wrong: {len(exp1_better)}")
    print(f"  {name2} correct, {name1} wrong: {len(exp2_better)}")
    print(f"  Both correct: {both_correct}")
    print(f"  Both wrong: {both_wrong}")
    
    return {
        'exp1_better': exp1_better,
        'exp2_better': exp2_better,
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        '_name1': name1,
        '_name2': name2,
    }


def inspect_question(exp: dict, idx: int) -> None:
    """
    Print details of a specific question.
    
    Args:
        exp: Experiment dict
        idx: Index in exp['results']
    """
    results = exp.get('results', [])
    
    if idx >= len(results):
        print(f"❌ Index {idx} out of range (max: {len(results)-1})")
        return
    
    r = results[idx]
    model = exp.get('_model', '?')
    exp_type = exp.get('_type', '?')
    
    print(f"\n{'='*70}")
    print(f"QUESTION {idx} | Model: {model} | Exp: {exp_type}")
    print(f"{'='*70}")
    
    print(f"\nCategory: {r.get('category', '?')}")
    print(f"\nContext:\n  {r.get('context', 'N/A')}")
    print(f"\nQuestion: {r.get('question', 'N/A')}")
    
    print("\nChoices:")
    choices = r.get('choices', [])
    correct = r.get('correct_answer', '?')
    for i, choice in enumerate(choices):
        letter = ['A', 'B', 'C'][i]
        marker = " ← CORRECT" if letter == correct else ""
        print(f"  {letter}. {choice}{marker}")
    
    print(f"\nAnswer Distribution: {r.get('answer_distribution', {})}")
    majority = _get_majority_answer(r.get('answer_distribution', {}))
    is_correct = majority == correct if majority else False
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    print(f"Majority Answer: {majority} ({status})")
    
    # Sample summary
    samples = r.get('samples', [])
    print(f"\nSamples ({len(samples)}):")
    for s in samples[:5]:  # Show first 5
        status = "✓" if s.get('correct') else "✗"
        print(f"  [{s.get('sample_idx', '?')}] {status} {s.get('answer', '?')} | {s.get('tokens', 0)} tokens")
    if len(samples) > 5:
        print(f"  ... and {len(samples) - 5} more")


def show_cot(exp: dict, idx: int, sample: int = 0, max_length: int = 3000) -> None:
    """
    Print Chain-of-Thought reasoning for a specific sample.
    
    Args:
        exp: Experiment dict
        idx: Index in exp['results']
        sample: Sample index (default 0)
        max_length: Max characters to show (default 3000)
    """
    results = exp.get('results', [])
    
    if idx >= len(results):
        print(f"❌ Question index {idx} out of range")
        return
    
    r = results[idx]
    samples = r.get('samples', [])
    
    if sample >= len(samples):
        print(f"❌ Sample index {sample} out of range (max: {len(samples)-1})")
        return
    
    s = samples[sample]
    model = exp.get('_model', '?')
    exp_type = exp.get('_type', '?')
    
    print(f"\n{'='*70}")
    print(f"COT: Question {idx}, Sample {sample} | Model: {model} | Exp: {exp_type}")
    print(f"{'='*70}")
    
    print(f"\nCategory: {r.get('category', '?')}")
    print(f"Question: {r.get('question', 'N/A')[:100]}...")
    
    correct = r.get('correct_answer', '?')
    answer = s.get('answer', '?')
    status = "✓ CORRECT" if s.get('correct') else "✗ WRONG"
    print(f"Answer: {answer} | Correct: {correct} | {status}")
    print(f"Tokens: {s.get('tokens', 0)}")
    
    print(f"\n{'-'*70}")
    print("FULL OUTPUT:")
    print(f"{'-'*70}\n")
    
    output = s.get('full_output', '')
    if max_length and len(output) > max_length:
        print(output[:max_length])
        print(f"\n... [truncated, {len(output) - max_length} more chars]")
    else:
        print(output)


def list_divergent(div: dict, which: str = 'exp1_better', limit: int = 10) -> None:
    """
    Print a formatted list of divergent questions.
    
    Args:
        div: Result from find_divergent()
        which: 'exp1_better' or 'exp2_better'
        limit: Max items to show
    """
    items = div.get(which, [])
    name1 = div.get('_name1', 'exp1')
    name2 = div.get('_name2', 'exp2')
    
    if which == 'exp1_better':
        print(f"\n{name1} correct, {name2} wrong ({len(items)} questions):")
    else:
        print(f"\n{name2} correct, {name1} wrong ({len(items)} questions):")
    
    for i, item in enumerate(items[:limit]):
        print(f"\n  [{i}] {item['category']}: {item['question']}")
        print(f"      Correct: {item['correct_answer']} | {name1}: {item.get(f'{name1}_answer', '?')} | {name2}: {item.get(f'{name2}_answer', '?')}")
        print(f"      idx={item['idx']} (use for inspect_question)")
    
    if len(items) > limit:
        print(f"\n  ... and {len(items) - limit} more")

