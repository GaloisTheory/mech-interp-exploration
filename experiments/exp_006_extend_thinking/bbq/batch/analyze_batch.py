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
    6. Compare experiments and find divergent questions
"""

#%% Imports
import os
import sys
import importlib.util

# Setup paths
_batch_dir = '/workspace/experiments/exp_006_extend_thinking/bbq/batch'
_bbq_dir = os.path.dirname(_batch_dir)

for path in [_batch_dir, _bbq_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Load batch_utils (core data loading)
_batch_utils_path = os.path.join(_batch_dir, 'batch_utils.py')
_spec = importlib.util.spec_from_file_location("batch_utils", _batch_utils_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load batch_utils from {_batch_utils_path}")
batch_utils = importlib.util.module_from_spec(_spec)
sys.modules["batch_utils"] = batch_utils
_spec.loader.exec_module(batch_utils)

# Load analyze_batch_utils (analysis functions)
_analyze_utils_path = os.path.join(_batch_dir, 'analyze_batch_utils.py')
_spec2 = importlib.util.spec_from_file_location("analyze_batch_utils", _analyze_utils_path)
if _spec2 is None or _spec2.loader is None:
    raise ImportError(f"Could not load analyze_batch_utils from {_analyze_utils_path}")
analyze_utils = importlib.util.module_from_spec(_spec2)
sys.modules["analyze_batch_utils"] = analyze_utils
_spec2.loader.exec_module(analyze_utils)

# Import functions
list_batch_experiments = batch_utils.list_batch_experiments
load_batch_experiment = batch_utils.load_batch_experiment
print_batch_summary = batch_utils.print_batch_summary
inspect_question = batch_utils.inspect_question
show_cot = batch_utils.show_cot
filter_wrong_samples = batch_utils.filter_wrong_samples
accuracy_by_category = batch_utils.accuracy_by_category
compare_experiments = batch_utils.compare_experiments

comparison_table = analyze_utils.comparison_table
accuracy_by_question = analyze_utils.accuracy_by_question
find_divergent_questions = analyze_utils.find_divergent_questions
show_divergent_question = analyze_utils.show_divergent_question
detailed_comparison_table = analyze_utils.detailed_comparison_table
load_baseline_combined = analyze_utils.load_baseline_combined
get_question_from_divergent = analyze_utils.get_question_from_divergent

# =============================================================================
# SINGLE EXPERIMENT EXPLORATION
# =============================================================================

#%% List available experiments
experiments = list_batch_experiments()

#%% Load a single experiment
exp = load_batch_experiment(0)  # Load most recent (or use name like "force_immediate")

#%% Print summary
print_batch_summary(exp)

#%% Accuracy by category
acc_by_cat = accuracy_by_category(exp)

#%% Inspect a specific question
inspect_question(exp, question_idx=0)

#%% Show Chain-of-Thought for a specific sample
show_cot(exp, question_idx=0, sample_idx=0)

#%% Filter to wrong answers
wrong = filter_wrong_samples(exp)
print(f"Found {len(wrong)} wrong samples")
for w in wrong[:5]:
    print(f"  Q{w['question_idx']} S{w['sample_idx']}: {w['category']} | Model: {w['model_answer']} | Correct: {w['correct_answer']}")

#%% Inspect a wrong answer
if wrong:
    w = wrong[0]
    inspect_question(exp, w['question_idx'])
    show_cot(exp, w['question_idx'], w['sample_idx'])

# =============================================================================
# COMPARING EXPERIMENTS
# =============================================================================

#%% Load baseline (combined) and force_immediate
baseline = load_baseline_combined(load_batch_experiment)
force_immediate = load_batch_experiment('force_immediate')

#%% Simple comparison table
if baseline['results'] and force_immediate:
    comparison_table(
        {'Baseline': baseline, 'ForceImm': force_immediate},
        title="Baseline vs Force Immediate - By Sample Accuracy"
    )

#%% Detailed comparison (by question AND by sample)
if baseline['results'] and force_immediate:
    detailed_comparison_table(
        {'Baseline': baseline, 'ForceImm': force_immediate},
        title="Baseline vs Force Immediate - Question & Sample Accuracy"
    )

# =============================================================================
# ALL EXPERIMENTS COMPARISON TABLES
# =============================================================================

#%% Load all experiments for comparison
print("Loading all experiments...")

# Baselines
baseline = load_baseline_combined(load_batch_experiment)
force_immediate = load_batch_experiment('force_immediate')

# Blank space experiments (static) - trailing underscore for exact match
blank_static = {}
for n in [5, 10, 50, 100, 500]:
    exp = load_batch_experiment(f'blank_static_{n}_')
    if exp:
        blank_static[f'{n}sp'] = exp

# Blank space experiments (dynamic)
blank_dynamic = {}
for metric in ['median', 'max']:
    prefix = 'med' if metric == 'median' else 'max'
    for mult in ['1x', '2x', '5x']:
        exp = load_batch_experiment(f'blank_dynamic_{metric}_{mult}_')
        if exp:
            blank_dynamic[f'{prefix}{mult}'] = exp

# Incorrect answer experiments (static) - trailing underscore for exact match
incorrect_static = {}
for n in [1, 2, 6, 12, 62]:
    exp = load_batch_experiment(f'incorrect_answer_{n}_')
    if exp:
        incorrect_static[f'{n}r'] = exp

# Incorrect answer experiments (dynamic)
incorrect_dynamic = {}
for metric in ['median', 'max']:
    prefix = 'med' if metric == 'median' else 'max'
    for mult in ['1x', '2x', '5x']:
        exp = load_batch_experiment(f'incorrect_dynamic_{metric}_{mult}_')
        if exp:
            incorrect_dynamic[f'{prefix}{mult}'] = exp

print("Done loading experiments.")

#%% TABLE 1: Overview - Baselines vs Key Interventions
print("\n" + "="*100)
print("TABLE 1: OVERVIEW")
print("="*100)
overview_exps = {'Base': baseline, 'NoCoT': force_immediate}
if blank_static.get('100sp'):
    overview_exps['Blk100'] = blank_static['100sp']
if incorrect_static.get('1r'):
    overview_exps['Inc1'] = incorrect_static['1r']
if incorrect_static.get('6r'):
    overview_exps['Inc6'] = incorrect_static['6r']

detailed_comparison_table(overview_exps, title="")

#%% TABLE 2: Blank Spaces - All Static and Dynamic
print("\n" + "="*100)
print("TABLE 2: BLANK SPACES (control - neutral tokens)")
print("="*100)
blank_all = {'Base': baseline, 'NoCoT': force_immediate}
blank_all.update(blank_static)
blank_all.update({f'D{k}': v for k, v in blank_dynamic.items()})

detailed_comparison_table(blank_all, title="")

#%% TABLE 3: Incorrect Answer - All Static and Dynamic  
print("\n" + "="*100)
print("TABLE 3: INCORRECT ANSWER (misleading tokens)")
print("="*100)
incorrect_all = {'Base': baseline, 'NoCoT': force_immediate}
incorrect_all.update(incorrect_static)
incorrect_all.update({f'D{k}': v for k, v in incorrect_dynamic.items()})

detailed_comparison_table(incorrect_all, title="")

# =============================================================================
# DIVERGENT QUESTIONS
# =============================================================================

#%% Find questions where experiments disagree
if baseline['results'] and force_immediate:
    divergent = find_divergent_questions(baseline, force_immediate, "Baseline", "ForceImm")
    
    print(f"Baseline correct, ForceImm wrong: {len(divergent['Baseline_better'])}")
    print(f"ForceImm correct, Baseline wrong: {len(divergent['ForceImm_better'])}")
    print()
    
    print("Cases where Baseline (CoT) helped:")
    for i, d in enumerate(divergent['Baseline_better'][:5]):
        print(f"  [{i}] {d['category']}: {d['question'][:50]}...")
        print(f"      Baseline: {d['Baseline_answer']} | ForceImm: {d['ForceImm_answer']} | Correct: {d['correct_answer']}")
#%% Inspect a Baseline_better question in baseline
# Use get_question_from_divergent() to find the question by text (works for combined experiments)
if baseline['results'] and divergent.get('Baseline_better'):
    d = divergent['Baseline_better'][0]
    
    idx = get_question_from_divergent(baseline, d)
    print(f"Found at baseline index: {idx}")
    print(f"Category: {d['category']}, Question: {d['question'][:60]}...")
    
    if idx is not None:
        inspect_question(baseline, idx)
        show_cot(baseline, idx, sample_idx=0)

#%% Inspect same question in force_immediate
if force_immediate and divergent.get('Baseline_better'):
    d = divergent['Baseline_better'][0]
    
    idx = get_question_from_divergent(force_immediate, d)
    print(f"Found at force_immediate index: {idx}")
    
    if idx is not None:
        inspect_question(force_immediate, idx)
        show_cot(force_immediate, idx, sample_idx=0)

#%% Cases where ForceImmediate was better
if baseline['results'] and force_immediate and divergent.get('ForceImm_better'):
    print("Cases where ForceImm was better:")
    for i, d in enumerate(divergent['ForceImm_better'][:5]):
        print(f"  [{i}] {d['category']}: {d['question'][:50]}...")
        print(f"       exp1_idx={d['exp1_idx']}, exp2_idx={d['exp2_idx']}")



#%% Inspect a ForceImm_better question in baseline (using text matching)
# Use get_question_from_divergent() for combined experiments like baseline
if baseline['results'] and divergent.get('ForceImm_better'):
    d = divergent['ForceImm_better'][0]
    
    # Find the question by matching category + question text (works for combined experiments)
    idx = get_question_from_divergent(baseline, d)
    print(f"Found at baseline index: {idx}")
    
    if idx is not None:
        inspect_question(baseline, idx)
        show_cot(baseline, idx, sample_idx=0)

#%% Inspect same question in force_immediate
if force_immediate and divergent.get('ForceImm_better'):
    d = divergent['ForceImm_better'][0]
    
    # For any experiment, get_question_from_divergent matches by category + question text
    idx = get_question_from_divergent(force_immediate, d)
    print(f"Found at force_immediate index: {idx}")
    
    if idx is not None:
        inspect_question(force_immediate, idx)
        show_cot(force_immediate, idx, sample_idx=0)


# =============================================================================
# CUSTOM ANALYSIS EXAMPLES
# =============================================================================

#%% Find questions where model was always wrong
if exp:
    always_wrong = [r for r in exp.get("results", []) if r.get("accuracy", 1) == 0]
    print(f"Questions where model was always wrong: {len(always_wrong)}")
    for q in always_wrong[:5]:
        print(f"  [{q['question_idx']}] {q['category']}: {q['question'][:50]}...")
# %%
divergent 
# %%
#%% Find questions with high variance (split answers)
if exp:
    high_variance = [r for r in exp.get("results", []) if len(r.get("answer_distribution", {})) > 1]
    print(f"Questions with split answers: {len(high_variance)}")
    for q in high_variance[:5]:
        print(f"  [{q['question_idx']}] {q['category']}: {q['answer_distribution']}")

#%% Extract all CoT traces for a category
if exp:
    category = "Age"
    cots = []
    for r in exp.get("results", []):
        if r.get("category") == category:
            for s in r.get("samples", []):
                cots.append({"question": r["question"], "answer": s["answer"], "correct": s["correct"], "cot": s["full_output"]})
    print(f"Extracted {len(cots)} CoT traces for {category}")
