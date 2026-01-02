"""Analysis utilities for BBQ batch experiments.

Functions for comparing experiments, computing accuracy metrics,
and finding divergent questions between experiments.
"""

from typing import Dict, List, Optional


def comparison_table(experiments: dict, title: str = "Experiment Comparison"):
    """
    Generate a comparison table across multiple experiments.
    
    Args:
        experiments: Dict of {name: experiment_data}
        title: Table title
        
    Shows both:
        - By Sample: % of individual samples correct
        - By Question: % of questions where majority answer is correct
    """
    print("=" * 90)
    print(title)
    print("=" * 90)
    print()
    
    # Get all categories across experiments
    all_cats = set()
    for exp in experiments.values():
        if exp:
            all_cats.update(exp.get('summary', {}).get('by_category', {}).keys())
    
    # Header
    exp_names = list(experiments.keys())
    header = f"{'Category':<25}"
    for name in exp_names:
        header += f" {name:>12}"
    print(header)
    print("-" * 90)
    
    # Rows by category
    for cat in sorted(all_cats):
        row = f"{cat:<25}"
        for name, exp in experiments.items():
            if exp:
                acc = exp.get('summary', {}).get('by_category', {}).get(cat, 0)
                row += f" {acc:>11.0%}"
            else:
                row += f" {'N/A':>12}"
        print(row)
    
    # Overall
    print("-" * 90)
    row = f"{'OVERALL':<25}"
    for name, exp in experiments.items():
        if exp:
            acc = exp.get('summary', {}).get('overall_accuracy', 0)
            row += f" {acc:>11.0%}"
        else:
            row += f" {'N/A':>12}"
    print(row)
    print()


def accuracy_by_question(exp: dict) -> dict:
    """
    Compute accuracy by question (majority vote per question).
    
    Ties are counted as INCORRECT (conservative approach for bias research).
    
    Returns dict with:
        - by_question: % of questions where majority answer is correct
        - by_sample: % of individual samples correct (same as summary)
        - by_category_question: per-category question-level accuracy
    """
    results = exp.get('results', [])
    
    by_cat_question = {}
    by_cat_sample = {}
    
    for r in results:
        cat = r.get('category', 'unknown')
        if cat not in by_cat_question:
            by_cat_question[cat] = {'correct': 0, 'total': 0}
            by_cat_sample[cat] = {'correct': 0, 'total': 0}
        
        # Question-level: is majority answer correct?
        # Ties count as incorrect (no clear majority)
        answer_dist = r.get('answer_distribution', {})
        if answer_dist:
            correct_answer = r.get('correct_answer', '')
            max_count = max(answer_dist.values())
            # Check for tie: count how many answers have the max count
            answers_with_max = [a for a, c in answer_dist.items() if c == max_count]
            if len(answers_with_max) == 1 and answers_with_max[0] == correct_answer:
                # Clear majority AND it's correct
                by_cat_question[cat]['correct'] += 1
            # else: tie or wrong majority → counts as incorrect
        by_cat_question[cat]['total'] += 1
        
        # Sample-level
        for s in r.get('samples', []):
            by_cat_sample[cat]['total'] += 1
            if s.get('correct'):
                by_cat_sample[cat]['correct'] += 1
    
    # Compute percentages
    question_acc = {
        cat: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        for cat, stats in by_cat_question.items()
    }
    sample_acc = {
        cat: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        for cat, stats in by_cat_sample.items()
    }
    
    total_q_correct = sum(s['correct'] for s in by_cat_question.values())
    total_q = sum(s['total'] for s in by_cat_question.values())
    total_s_correct = sum(s['correct'] for s in by_cat_sample.values())
    total_s = sum(s['total'] for s in by_cat_sample.values())
    
    return {
        'overall_by_question': total_q_correct / total_q if total_q > 0 else 0,
        'overall_by_sample': total_s_correct / total_s if total_s > 0 else 0,
        'by_category_question': question_acc,
        'by_category_sample': sample_acc,
    }


def find_divergent_questions(exp1: dict, exp2: dict, name1: str = "Exp1", name2: str = "Exp2"):
    """
    Find questions where experiments disagree (one correct, other wrong).
    
    Matches questions by (category, question text).
    
    Returns:
        Dict with '{name1}_better' and '{name2}_better' lists
    """
    # Build lookup by (category, question) for exp1
    exp1_by_q = {}
    for r in exp1.get('results', []):
        key = (r.get('category'), r.get('question'))
        # Get majority answer
        answer_dist = r.get('answer_distribution', {})
        if answer_dist:
            max_count = max(answer_dist.values())
            answers_with_max = [a for a, c in answer_dist.items() if c == max_count]
            majority = answers_with_max[0] if len(answers_with_max) == 1 else None
        else:
            majority = None
        correct = r.get('correct_answer')
        exp1_by_q[key] = {
            'majority': majority,
            'correct': correct,
            'is_correct': majority == correct if majority else False,
            'accuracy': r.get('accuracy', 0),
            'distribution': answer_dist,
            'question_idx': r.get('question_idx'),
            'context': r.get('context'),
            'choices': r.get('choices'),
        }
    
    # Compare with exp2
    exp1_better = []  # exp1 correct, exp2 wrong
    exp2_better = []  # exp2 correct, exp1 wrong
    
    for r in exp2.get('results', []):
        key = (r.get('category'), r.get('question'))
        if key not in exp1_by_q:
            continue
        
        exp1_data = exp1_by_q[key]
        
        # Get exp2 majority
        answer_dist = r.get('answer_distribution', {})
        if answer_dist:
            max_count = max(answer_dist.values())
            answers_with_max = [a for a, c in answer_dist.items() if c == max_count]
            exp2_majority = answers_with_max[0] if len(answers_with_max) == 1 else None
        else:
            exp2_majority = None
        
        correct = r.get('correct_answer')
        exp2_correct = exp2_majority == correct if exp2_majority else False
        
        if exp1_data['is_correct'] and not exp2_correct:
            exp1_better.append({
                'category': r.get('category'),
                'question': r.get('question'),
                'context': exp1_data['context'],
                'choices': exp1_data['choices'],
                'correct_answer': correct,
                f'{name1}_answer': exp1_data['majority'],
                f'{name1}_dist': exp1_data['distribution'],
                f'{name2}_answer': exp2_majority,
                f'{name2}_dist': answer_dist,
                f'{name1}_idx': exp1_data['question_idx'],
                f'{name2}_idx': r.get('question_idx'),
                # Standard keys for easy access
                'exp1_idx': exp1_data['question_idx'],
                'exp2_idx': r.get('question_idx'),
            })
        elif exp2_correct and not exp1_data['is_correct']:
            exp2_better.append({
                'category': r.get('category'),
                'question': r.get('question'),
                'context': exp1_data['context'],
                'choices': exp1_data['choices'],
                'correct_answer': correct,
                f'{name1}_answer': exp1_data['majority'],
                f'{name1}_dist': exp1_data['distribution'],
                f'{name2}_answer': exp2_majority,
                f'{name2}_dist': answer_dist,
                f'{name1}_idx': exp1_data['question_idx'],
                f'{name2}_idx': r.get('question_idx'),
                # Standard keys for easy access
                'exp1_idx': exp1_data['question_idx'],
                'exp2_idx': r.get('question_idx'),
            })
    
    return {
        f'{name1}_better': exp1_better,
        f'{name2}_better': exp2_better,
    }


def show_divergent_question(divergent: dict, idx: int = 0, exp1: dict = None, exp2: dict = None,
                            name1: str = "Exp1", name2: str = "Exp2"):
    """
    Show details of a divergent question.
    
    Args:
        divergent: Single item from find_divergent_questions result
        idx: Index for display
        exp1, exp2: Full experiment data (to show CoT if available)
        name1, name2: Names for display
    """
    print("\n" + "=" * 80)
    print(f"DIVERGENT QUESTION #{idx}")
    print("=" * 80)
    
    print(f"\nCategory: {divergent.get('category')}")
    print(f"Context: {divergent.get('context', 'N/A')[:200]}...")
    print(f"Question: {divergent.get('question')}")
    
    print("\nChoices:")
    choices = divergent.get('choices', [])
    correct = divergent.get('correct_answer')
    for i, c in enumerate(choices):
        letter = ['A', 'B', 'C'][i]
        mark = " ← correct" if letter == correct else ""
        print(f"  {letter}. {c}{mark}")
    
    print(f"\n{name1}: {divergent.get(f'{name1}_answer')} {divergent.get(f'{name1}_dist')}")
    print(f"{name2}: {divergent.get(f'{name2}_answer')} {divergent.get(f'{name2}_dist')}")
    
    # Show CoT if experiments provided
    if exp1 and exp2:
        exp1_idx = divergent.get('exp1_idx') or divergent.get(f'{name1}_idx')
        exp2_idx = divergent.get('exp2_idx') or divergent.get(f'{name2}_idx')
        
        if exp1_idx is not None:
            print(f"\n--- {name1} CoT (question_idx={exp1_idx}, sample 0) ---")
            for r in exp1.get('results', []):
                if r.get('question_idx') == exp1_idx:
                    samples = r.get('samples', [])
                    if samples:
                        output = samples[0].get('full_output', '')[:1000]
                        print(output + ("..." if len(samples[0].get('full_output', '')) > 1000 else ""))
                    break
        
        if exp2_idx is not None:
            print(f"\n--- {name2} CoT (question_idx={exp2_idx}, sample 0) ---")
            for r in exp2.get('results', []):
                if r.get('question_idx') == exp2_idx:
                    samples = r.get('samples', [])
                    if samples:
                        output = samples[0].get('full_output', '')[:1000]
                        print(output + ("..." if len(samples[0].get('full_output', '')) > 1000 else ""))
                    break
        
        # Print indices for easy copy-paste
        print("\n" + "-" * 40)
        print("To inspect further:")
        print(f"  inspect_question(exp1, {exp1_idx})  # {name1}")
        print(f"  show_cot(exp1, {exp1_idx}, sample_idx)")
        print(f"  inspect_question(exp2, {exp2_idx})  # {name2}")
        print(f"  show_cot(exp2, {exp2_idx}, sample_idx)")


def detailed_comparison_table(experiments: dict, title: str = "Detailed Comparison"):
    """
    Generate comparison showing both by-question and by-sample accuracy.
    
    Args:
        experiments: Dict of {name: experiment_data}
    """
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()
    print("Accuracy shown as: by_question% / by_sample%")
    print("  - By Question: % of questions where majority answer is correct")
    print("  - By Sample: % of individual samples correct")
    print()
    
    # Compute stats for each experiment
    stats = {}
    all_cats = set()
    for name, exp in experiments.items():
        if exp:
            stats[name] = accuracy_by_question(exp)
            all_cats.update(stats[name]['by_category_question'].keys())
    
    # Header
    exp_names = list(experiments.keys())
    header = f"{'Category':<25}"
    for name in exp_names:
        header += f" {name:>20}"
    print(header)
    print("-" * 100)
    
    # Rows
    for cat in sorted(all_cats):
        row = f"{cat:<25}"
        for name in exp_names:
            if name in stats:
                q_acc = stats[name]['by_category_question'].get(cat, 0)
                s_acc = stats[name]['by_category_sample'].get(cat, 0)
                row += f" {q_acc:>8.0%} / {s_acc:<7.0%}"
            else:
                row += f" {'N/A':>20}"
        print(row)
    
    # Overall
    print("-" * 100)
    row = f"{'OVERALL':<25}"
    for name in exp_names:
        if name in stats:
            q_acc = stats[name]['overall_by_question']
            s_acc = stats[name]['overall_by_sample']
            row += f" {q_acc:>8.0%} / {s_acc:<7.0%}"
        else:
            row += f" {'N/A':>20}"
    print(row)
    print()


def find_question_in_exp(exp: dict, category: str = None, question: str = None, 
                          question_idx: int = None) -> Optional[int]:
    """
    Find a question's position in an experiment's results list.
    
    Args:
        exp: Experiment data
        category: Category to match (optional)
        question: Question text to match (optional, partial match)
        question_idx: Original question_idx to match (optional)
        
    Returns:
        Index in exp['results'] list, or None if not found
    """
    for i, r in enumerate(exp.get('results', [])):
        if category and r.get('category') != category:
            continue
        if question and question not in r.get('question', ''):
            continue
        if question_idx is not None and r.get('question_idx') != question_idx:
            continue
        return i
    return None


def get_question_from_divergent(exp: dict, divergent_item: dict) -> Optional[int]:
    """
    Find the position of a divergent question in an experiment.
    
    Matches by category and question text (since indices may not match in combined experiments).
    
    Args:
        exp: Experiment data (can be combined)
        divergent_item: Item from find_divergent_questions result
        
    Returns:
        Index in exp['results'] list, or None if not found
    """
    target_cat = divergent_item.get('category')
    target_q = divergent_item.get('question')
    
    for i, r in enumerate(exp.get('results', [])):
        if r.get('category') == target_cat and r.get('question') == target_q:
            return i
    return None


def load_baseline_combined(load_fn, categories: List[str] = None, prefix: str = None) -> dict:
    """
    Load and combine all baseline experiments.
    
    Args:
        load_fn: Function to load experiments (e.g., load_batch_experiment)
        categories: List of categories to load (default: all 11)
        prefix: Optional prefix for experiment names (e.g., 'qwen_1.7B').
                If provided, loads '{prefix}_baseline_{cat}' instead of 'baseline_{cat}'.
        
    Returns:
        Combined experiment dict with all results merged
    """
    import os
    import re
    from glob import glob
    
    if categories is None:
        categories = ['age', 'disability', 'gender', 'nationality', 'appearance', 
                      'race', 'race_ses', 'race_gender', 'religion', 'ses', 'sexual_orientation']
    
    baseline_combined = {'results': [], 'summary': {'by_category': {}}}
    
    # Get output directory from load_fn module
    # Assume load_fn is from batch_utils which has OUTPUT_DIR
    import batch_utils
    output_dir = batch_utils.OUTPUT_DIR
    
    for cat in categories:
        try:
            # Build experiment name pattern with optional prefix
            # Use regex to match exact category name followed by underscore and timestamp
            if prefix:
                pattern = f'{prefix}_baseline_{cat}_[0-9]+'
            else:
                pattern = f'baseline_{cat}_[0-9]+'
            
            # Find matching folders
            if not os.path.exists(output_dir):
                continue
                
            folders = [f for f in os.listdir(output_dir) 
                      if os.path.isdir(os.path.join(output_dir, f))]
            
            matching_folders = [f for f in folders if re.match(pattern, f)]
            
            if not matching_folders:
                continue
            
            # Use most recent matching folder
            matching_folders = sorted(matching_folders, 
                                    key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), 
                                    reverse=True)
            folder = matching_folders[0]
            
            # Load directly instead of using load_fn to avoid substring issues
            folder_path = os.path.join(output_dir, folder)
            results_files = glob(os.path.join(folder_path, "results_*.json"))
            
            if not results_files:
                continue
                
            results_file = max(results_files, key=os.path.getmtime)
            
            import json
            with open(results_file, 'r') as f:
                exp = json.load(f)
            
            print(f"✓ Loaded: {folder}")
            print(f"  Questions: {exp.get('summary', {}).get('total_questions', '?')}")
            print(f"  Samples: {exp.get('summary', {}).get('total_samples', '?')}")
            print(f"  Accuracy: {exp.get('summary', {}).get('overall_accuracy', 0):.1%}")
            
            if exp and exp.get('results'):
                baseline_combined['results'].extend(exp['results'])
                baseline_combined['summary']['by_category'].update(
                    exp.get('summary', {}).get('by_category', {})
                )
        except Exception as e:
            print(f"⚠ Error loading {cat}: {e}")
            pass
    
    # Compute overall accuracy
    if baseline_combined['results']:
        total_s = sum(len(r['samples']) for r in baseline_combined['results'])
        total_correct = sum(sum(1 for s in r['samples'] if s['correct']) for r in baseline_combined['results'])
        baseline_combined['summary']['overall_accuracy'] = total_correct / total_s if total_s > 0 else 0
        baseline_combined['summary']['total_questions'] = len(baseline_combined['results'])
        baseline_combined['summary']['total_samples'] = total_s
    
    return baseline_combined

