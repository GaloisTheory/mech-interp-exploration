#!/usr/bin/env python3
"""BBQ Experiment Analysis - Generalized for Any Model

Visual analysis of CoT insertion experiments with:
- Summary table comparing CoT, NoCoT, best/worst experiments
- Line graphs showing accuracy vs tokens/insertions

Usage:
    # Command line
    python analyze_cot_experiment.py qwen_1.7B
    python analyze_cot_experiment.py qwen_32B
    
    # Or interactively with #%% markers in VS Code/Cursor
    # Set MODEL_PREFIX below and run cells
"""

import os
import sys
import argparse
import importlib.util
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =============================================================================
# Configuration (can be overridden via CLI)
# =============================================================================
MODEL_PREFIX = 'qwen_1.7B'  # Default, override with CLI arg
SAVE_FIGURES = True  # Toggle to save outputs
BATCH_DIR = '/workspace/experiments/exp_006_extend_thinking/bbq/batch'

# =============================================================================
# Setup
# =============================================================================
def setup_analysis(model_prefix):
    """Setup paths and import modules."""
    global MODEL_PREFIX, OUTPUT_DIR, load_batch_experiment, load_baseline_combined, accuracy_by_question
    
    MODEL_PREFIX = model_prefix
    OUTPUT_DIR = os.path.join(BATCH_DIR, 'outputs', f'{model_prefix}_graphs')
    
    # Setup paths
    bbq_dir = os.path.dirname(BATCH_DIR)
    for path in [BATCH_DIR, bbq_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Load batch_utils
    batch_utils_path = os.path.join(BATCH_DIR, 'batch_utils.py')
    spec = importlib.util.spec_from_file_location("batch_utils", batch_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load batch_utils from {batch_utils_path}")
    batch_utils = importlib.util.module_from_spec(spec)
    sys.modules["batch_utils"] = batch_utils
    spec.loader.exec_module(batch_utils)
    
    # Load analyze_batch_utils
    analyze_utils_path = os.path.join(BATCH_DIR, 'analyze_batch_utils.py')
    spec2 = importlib.util.spec_from_file_location("analyze_batch_utils", analyze_utils_path)
    if spec2 is None or spec2.loader is None:
        raise ImportError(f"Could not load analyze_batch_utils from {analyze_utils_path}")
    analyze_utils = importlib.util.module_from_spec(spec2)
    sys.modules["analyze_batch_utils"] = analyze_utils
    spec2.loader.exec_module(analyze_utils)
    
    # Import functions
    load_batch_experiment = batch_utils.load_batch_experiment
    load_baseline_combined = analyze_utils.load_baseline_combined
    accuracy_by_question = analyze_utils.accuracy_by_question
    
    print(f"✓ Setup complete for model: {model_prefix}")
    return OUTPUT_DIR

# =============================================================================
# Data Loading
# =============================================================================
def load_all_experiments(model_prefix):
    """Load all experiments for a given model prefix."""
    print(f"\nLoading all experiments for {model_prefix}...")
    
    # Helper to load with prefix
    def load_exp(name):
        """Load experiment with model prefix."""
        return load_batch_experiment(f'{model_prefix}_{name}')
    
    # Baselines
    cot = load_baseline_combined(load_batch_experiment, prefix=model_prefix)
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
    
    return {
        'cot': cot,
        'nocot': nocot,
        'blank_static': blank_static,
        'blank_static_values': blank_static_values,
        'blank_median': blank_median,
        'blank_max': blank_max,
        'incorrect_static': incorrect_static,
        'incorrect_static_values': incorrect_static_values,
        'incorrect_median': incorrect_median,
        'incorrect_max': incorrect_max,
    }

# =============================================================================
# Analysis Functions
# =============================================================================
def get_accuracies(exp):
    """Get both question and sample accuracy for an experiment."""
    if not exp or not exp.get('results'):
        return None, None
    stats = accuracy_by_question(exp)
    return stats['overall_by_question'], stats['overall_by_sample']

def compute_stats(experiments):
    """Compute accuracy statistics for all experiments."""
    cot_q, cot_s = get_accuracies(experiments['cot'])
    nocot_q, nocot_s = get_accuracies(experiments['nocot'])
    
    print(f"\nBaseline Accuracies:")
    print(f"  CoT:   Question={cot_q:.1%}, Sample={cot_s:.1%}")
    print(f"  NoCoT: Question={nocot_q:.1%}, Sample={nocot_s:.1%}")
    
    # Collect all blank experiment accuracies
    all_blank = {}
    for n, exp in experiments['blank_static'].items():
        q, s = get_accuracies(exp)
        if q is not None:
            all_blank[f'static_{n}'] = {'q': q, 's': s, 'exp': exp}
    
    for mult, exp in experiments['blank_median'].items():
        q, s = get_accuracies(exp)
        if q is not None:
            all_blank[f'median_{mult}x'] = {'q': q, 's': s, 'exp': exp}
    
    for mult, exp in experiments['blank_max'].items():
        q, s = get_accuracies(exp)
        if q is not None:
            all_blank[f'max_{mult}x'] = {'q': q, 's': s, 'exp': exp}
    
    # Collect all incorrect experiment accuracies
    all_incorrect = {}
    for n, exp in experiments['incorrect_static'].items():
        q, s = get_accuracies(exp)
        if q is not None:
            all_incorrect[f'static_{n}'] = {'q': q, 's': s, 'exp': exp}
    
    for mult, exp in experiments['incorrect_median'].items():
        q, s = get_accuracies(exp)
        if q is not None:
            all_incorrect[f'median_{mult}x'] = {'q': q, 's': s, 'exp': exp}
    
    for mult, exp in experiments['incorrect_max'].items():
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
    
    return {
        'cot_q': cot_q, 'cot_s': cot_s,
        'nocot_q': nocot_q, 'nocot_s': nocot_s,
        'all_blank': all_blank,
        'blank_best': blank_best, 'blank_best_key': blank_best_key,
        'blank_worst': blank_worst, 'blank_worst_key': blank_worst_key,
        'all_incorrect': all_incorrect,
        'incorrect_best': incorrect_best, 'incorrect_best_key': incorrect_best_key,
        'incorrect_worst': incorrect_worst, 'incorrect_worst_key': incorrect_worst_key,
    }

def print_summary_table(experiments, stats, model_name):
    """Print text summary table."""
    print("\n" + "=" * 120)
    print(f"SUMMARY: {model_name}")
    print("=" * 120)
    print(f"{'Experiment':<40} {'Question Acc':>15} {'Sample Acc':>15}")
    print("-" * 120)
    print(f"{'CoT (Baseline)':<40} {stats['cot_q']:>14.1%} {stats['cot_s']:>15.1%}")
    print(f"{'NoCoT':<40} {stats['nocot_q']:>14.1%} {stats['nocot_s']:>15.1%}")
    if stats['blank_best']:
        print(f"{'Blank Best (' + stats['blank_best_key'] + ')':<40} {stats['blank_best']['q']:>14.1%} {stats['blank_best']['s']:>15.1%}")
        print(f"{'Blank Worst (' + stats['blank_worst_key'] + ')':<40} {stats['blank_worst']['q']:>14.1%} {stats['blank_worst']['s']:>15.1%}")
    if stats['incorrect_best']:
        print(f"{'Incorrect Best (' + stats['incorrect_best_key'] + ')':<40} {stats['incorrect_best']['q']:>14.1%} {stats['incorrect_best']['s']:>15.1%}")
        print(f"{'Incorrect Worst (' + stats['incorrect_worst_key'] + ')':<40} {stats['incorrect_worst']['q']:>14.1%} {stats['incorrect_worst']['s']:>15.1%}")
    print("=" * 120)

# =============================================================================
# Main Analysis
# =============================================================================
def run_analysis(model_prefix, save_figures=True):
    """Run complete analysis for a model."""
    # Setup
    output_dir = setup_analysis(model_prefix)
    
    # Load experiments
    experiments = load_all_experiments(model_prefix)
    
    # Compute stats
    stats = compute_stats(experiments)
    
    # Get model name from prefix
    model_display = model_prefix.replace('_', '-').replace('qwen-', 'Qwen-').upper()
    
    # Print summary
    print_summary_table(experiments, stats, model_display)
    
    # Create output directory
    if save_figures:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n✓ Output directory: {output_dir}")
    
    print(f"\n✓ Analysis complete for {model_display}")
    print(f"Results saved in: {output_dir}")
    
    return experiments, stats

# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BBQ CoT insertion experiments")
    parser.add_argument("model_prefix", nargs='?', default="qwen_1.7B",
                       help="Model prefix (e.g., qwen_1.7B, qwen_32B)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save figures")
    
    args = parser.parse_args()
    
    run_analysis(args.model_prefix, save_figures=not args.no_save)

