#!/usr/bin/env python3
"""BBQ Question Inspector

Interactive script to inspect questions, answers, and chain-of-thought outputs 
across different experiments.

NOTE: Config.yaml may show wrong model name (legacy issue). Check logs for actual model.
The run_question_batch.py script now saves the correct model name.

Usage:
    Edit the CONFIGURATION section below and run:
    python inspect_questions.py
"""
# %%
import os
import sys
import importlib.util

# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================

# Which experiments to compare (list of experiment name patterns)
EXPERIMENTS = [
    'qwen_32B_baseline_age',           # CoT baseline (age category)
    'qwen_32B_force_immediate',        # NoCoT (all categories)
    'qwen_32B_incorrect_answer_1_',    # 1 incorrect answer repetition
    'qwen_32B_incorrect_answer_2_',    # 2 incorrect answer repetitions
]

# Which question to inspect
CATEGORY = 'age'        # Category name (lowercase)
QUESTION_IDX = 0        # Question index within category (0-based)
SAMPLE_IDX = 0          # Which sample to show (0-9 typically)
NUM_SAMPLES_TO_SHOW = 3 # How many samples to display
# %%
# ============================================================
# SETUP
# ============================================================

_batch_dir = '/workspace/experiments/exp_006_extend_thinking/bbq/batch'
sys.path.insert(0, _batch_dir)

# Load batch_utils
_batch_utils_path = os.path.join(_batch_dir, 'batch_utils.py')
_spec = importlib.util.spec_from_file_location("batch_utils", _batch_utils_path)
batch_utils = importlib.util.module_from_spec(_spec)
sys.modules["batch_utils"] = batch_utils
_spec.loader.exec_module(batch_utils)

load_batch_experiment = batch_utils.load_batch_experiment

# ============================================================
# HELPER FUNCTIONS
# ============================================================
# %%
def find_question_by_category(exp, category, question_idx):
    """Find a question by category and index within that category."""
    category_lower = category.lower().replace('_', '').replace(' ', '')
    
    # Find all questions in this category
    category_questions = []
    for i, result in enumerate(exp.get('results', [])):
        result_cat = result.get('category', '').lower().replace('_', '').replace(' ', '')
        if category_lower in result_cat or result_cat in category_lower:
            category_questions.append((i, result))
    
    if not category_questions:
        return None, None
    
    if question_idx >= len(category_questions):
        print(f"⚠ Question index {question_idx} out of range for category {category} (max: {len(category_questions)-1})")
        return None, None
    
    return category_questions[question_idx]


def display_question(result):
    """Display question information in a nice format."""
    if not result:
        print("No result to display")
        return
    
    print("="*80)
    print(f"CATEGORY: {result.get('category', 'N/A')}")
    print("="*80)
    
    print(f"\nContext:\n{result.get('context', 'N/A')}")
    print(f"\nQuestion:\n{result.get('question', 'N/A')}")
    
    print(f"\nChoices:")
    choices = result.get('choices', [])
    correct = result.get('correct_answer', '')
    for i, choice in enumerate(choices):
        letter = ['A', 'B', 'C'][i]
        marker = " ← CORRECT" if letter == correct else ""
        print(f"  {letter}. {choice}{marker}")
    
    print(f"\nAnswer Distribution:")
    dist = result.get('answer_distribution', {})
    for ans, count in sorted(dist.items()):
        pct = count / sum(dist.values()) * 100 if dist else 0
        marker = " ← CORRECT" if ans == correct else ""
        print(f"  {ans}: {count:2d} ({pct:5.1f}%){marker}")
    
    accuracy = result.get('accuracy', 0)
    print(f"\nAccuracy: {accuracy:.1%}")


def display_cot_output(result, sample_idx=0, max_chars=3000):
    """Display the full output for a specific sample."""
    if not result:
        print("No result to display")
        return
    
    samples = result.get('samples', [])
    if sample_idx >= len(samples):
        print(f"Sample {sample_idx} not found (max: {len(samples)-1})")
        return
    
    sample = samples[sample_idx]
    
    print("="*80)
    print(f"SAMPLE {sample_idx}")
    print("="*80)
    print(f"Answer: {sample.get('answer', 'N/A')}")
    print(f"Correct: {sample.get('correct', False)}")
    print(f"Tokens: {sample.get('tokens', 'N/A')}")
    print(f"Time: {sample.get('time_s', 'N/A'):.2f}s")
    print()
    print("-" * 80)
    print("FULL OUTPUT:")
    print("-" * 80)
    
    output = sample.get('full_output', '')
    if len(output) > max_chars:
        print(output[:max_chars])
        print(f"\n... [truncated, {len(output)-max_chars} more characters] ...")
    else:
        print(output)


# ============================================================
# MAIN
# ============================================================
# %%
print("="*80)
print("BBQ QUESTION INSPECTOR")
print("="*80)
print(f"\n⚠️  IMPORTANT: qwen_32B_* experiments are actually using Qwen3-8B!\n")

# Load experiments
print("\nLoading experiments...")
loaded_experiments = {}

for exp_name in EXPERIMENTS:
    try:
        exp = load_batch_experiment(exp_name)
        if exp:
            loaded_experiments[exp_name] = exp
    except Exception as e:
        print(f"✗ Error loading {exp_name}: {e}")

print(f"\n✓ Loaded {len(loaded_experiments)} experiments\n")

if not loaded_experiments:
    print("No experiments loaded. Check experiment names.")
    return

# Show experiment metadata
print("\n" + "="*80)
print("EXPERIMENT METADATA")
print("="*80)

for name, exp in loaded_experiments.items():
    config = exp.get('config', {})
    summary = exp.get('summary', {})
    
    print(f"\n{name}:")
    print(f"  Model: {config.get('model', {}).get('name', 'N/A')}")
    print(f"  Questions: {summary.get('total_questions', 'N/A')}")
    print(f"  Accuracy: {summary.get('overall_accuracy', 0):.1%}")
    
    override = config.get('override', {})
    if override:
        print(f"  Intervention: ", end="")
        if 'incorrect_answer_repetitions' in override:
            print(f"Incorrect×{override['incorrect_answer_repetitions']}", end="")
        if 'blank_spaces' in override:
            print(f"Blank×{override['blank_spaces']}", end="")
        print()

# Display the question
print("\n" + "="*80)
print(f"QUESTION: {CATEGORY} - Index {QUESTION_IDX}")
print("="*80)

first_exp = list(loaded_experiments.values())[0]
global_idx, result = find_question_by_category(first_exp, CATEGORY, QUESTION_IDX)

if not result:
    print(f"Question not found in category {CATEGORY}")
    return

display_question(result)

# Compare across experiments
print("\n\n" + "="*80)
print("COMPARISON ACROSS EXPERIMENTS")
print("="*80)

for exp_name, exp in loaded_experiments.items():
    global_idx, result = find_question_by_category(exp, CATEGORY, QUESTION_IDX)
    
    if not result:
        print(f"\n{exp_name}: NOT FOUND")
        continue
    
    dist = result.get('answer_distribution', {})
    correct = result.get('correct_answer', '')
    accuracy = result.get('accuracy', 0)
    
    # Find majority answer
    if dist:
        majority = max(dist.items(), key=lambda x: x[1])[0]
        majority_is_correct = (majority == correct)
    else:
        majority = 'N/A'
        majority_is_correct = False
    
    print(f"\n{exp_name}:")
    print(f"  Majority: {majority} {'✓' if majority_is_correct else '✗'}")
    print(f"  Distribution: {dict(dist)}")
    print(f"  Accuracy: {accuracy:.1%}")

# Display Chain-of-Thought outputs
print("\n\n" + "#"*80)
print("CHAIN-OF-THOUGHT OUTPUTS")
print("#"*80)

for exp_name, exp in loaded_experiments.items():
    global_idx, result = find_question_by_category(exp, CATEGORY, QUESTION_IDX)
    
    if not result:
        continue
    
    print(f"\n\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print("="*80)
    
    # Show multiple samples if requested
    samples = result.get('samples', [])
    num_to_show = min(NUM_SAMPLES_TO_SHOW, len(samples))
    
    for i in range(num_to_show):
        if i > 0:
            print("\n" + "-"*80 + "\n")
        display_cot_output(result, i, max_chars=2000)

# List available questions in category
print("\n\n" + "="*80)
print(f"ALL QUESTIONS IN CATEGORY: {CATEGORY}")
print("="*80)

first_exp = list(loaded_experiments.values())[0]
category_lower = CATEGORY.lower().replace('_', '').replace(' ', '')

questions = []
for i, result in enumerate(first_exp.get('results', [])):
    result_cat = result.get('category', '').lower().replace('_', '').replace(' ', '')
    if category_lower in result_cat or result_cat in category_lower:
        questions.append((len(questions), i, result))

print(f"\nFound {len(questions)} questions:")
for cat_idx, global_idx, result in questions:
    question_text = result.get('question', '')[:60]
    accuracy = result.get('accuracy', 0)
    correct = result.get('correct_answer', '')
    marker = " ← CURRENT" if cat_idx == QUESTION_IDX else ""
    print(f"  [{cat_idx}] Acc={accuracy:.0%} {correct}: {question_text}...{marker}")

print("\n" + "="*80)
print("To inspect a different question, edit the CONFIGURATION section")
print("at the top of this script and run again.")
print("="*80)




# %%
