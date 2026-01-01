"""Utilities for batch BBQ experiments.

Provides:
- Config loading/saving
- Checkpoint management  
- Analysis utilities for batch results
"""

import os
import sys
import json
import yaml
from glob import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any
from collections import Counter

# Setup paths
_batch_dir = os.path.dirname(os.path.abspath(__file__))
_bbq_dir = os.path.dirname(_batch_dir)
_exp_dir = os.path.dirname(_bbq_dir)

for path in [_batch_dir, _bbq_dir, _exp_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from constants import FEW_SHOT_EXAMPLES
from shared.config import format_bbq_prompt
from data.bbq_dataset import BBQItem

# Output directory
OUTPUT_DIR = os.path.join(_batch_dir, "outputs")


# =============================================================================
# CONFIG LOADING
# =============================================================================

def load_config(yaml_path: str) -> dict:
    """Load experiment config from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, output_dir: str):
    """Save a copy of config to output directory."""
    path = os.path.join(output_dir, "config.yaml")
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# PROMPT BUILDING (reuses shared/config.py)
# =============================================================================

def build_prompt(item: BBQItem, use_few_shot: bool = True) -> str:
    """Build prompt for a BBQ item."""
    base_prompt = format_bbq_prompt(
        context=item.context,
        question=item.question,
        choices=item.choices,
    )
    
    if use_few_shot:
        return FEW_SHOT_EXAMPLES + base_prompt
    return base_prompt


# =============================================================================
# OVERRIDE FUNCTION (adapted from playground.py)
# =============================================================================

def get_override_for_intercept(intercept_num: int, schedule: list) -> str:
    """Get the override text for a given intercept number (1-indexed).
    
    Args:
        intercept_num: Which intercept this is (1, 2, 3, ...)
        schedule: List of [start, end, text] tuples
        
    Returns:
        Override text to inject
    """
    for item in schedule:
        start, end, text = item[0], item[1], item[2]
        if start <= intercept_num <= end:
            return text
    return "\n\nWait, let me reconsider..."


def make_override_fn(schedule: list) -> Callable[[int], str]:
    """Create an override function from a schedule."""
    return lambda n: get_override_for_intercept(n, schedule)


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(output_dir: str, results: list, question_idx: int):
    """Save checkpoint after each question for resumability."""
    checkpoint = {
        "last_completed_idx": question_idx,
        "results": results,
        "saved_at": datetime.now().isoformat(),
    }
    path = os.path.join(output_dir, "checkpoint.json")
    with open(path, 'w') as f:
        json.dump(checkpoint, f)


def load_checkpoint(output_dir: str) -> Tuple[List, int]:
    """Load checkpoint if exists.
    
    Returns:
        (results_so_far, last_completed_idx) or ([], -1) if no checkpoint
    """
    path = os.path.join(output_dir, "checkpoint.json")
    if not os.path.exists(path):
        return [], -1
    
    with open(path, 'r') as f:
        checkpoint = json.load(f)
    
    return checkpoint["results"], checkpoint["last_completed_idx"]


def clear_checkpoint(output_dir: str):
    """Remove checkpoint file after successful completion."""
    path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# RESULTS SAVING
# =============================================================================

def save_results(output_dir: str, config: dict, results: list, 
                 started_at: datetime, completed_at: datetime):
    """Save final results with summary statistics."""
    # Compute summary
    total_samples = sum(len(r["samples"]) for r in results)
    total_correct = sum(
        sum(1 for s in r["samples"] if s["correct"]) 
        for r in results
    )
    total_time = sum(
        sum(s["time_s"] for s in r["samples"]) 
        for r in results
    )
    
    # Per-category accuracy
    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        for s in r["samples"]:
            by_category[cat]["total"] += 1
            if s["correct"]:
                by_category[cat]["correct"] += 1
    
    category_accuracy = {
        cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        for cat, stats in by_category.items()
    }
    
    output = {
        "config": config,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "results": results,
        "summary": {
            "total_questions": len(results),
            "total_samples": total_samples,
            "overall_accuracy": total_correct / total_samples if total_samples > 0 else 0,
            "total_time_s": total_time,
            "by_category": category_accuracy,
        }
    }
    
    # Save with timestamp
    timestamp = completed_at.strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    path = os.path.join(output_dir, filename)
    
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {path}")
    return path


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def list_batch_experiments():
    """List available batch experiment outputs."""
    if not os.path.exists(OUTPUT_DIR):
        print("No experiments found (outputs directory doesn't exist)")
        return []
    
    folders = [f for f in os.listdir(OUTPUT_DIR) 
               if os.path.isdir(os.path.join(OUTPUT_DIR, f))]
    folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
    
    print("Available batch experiments:")
    experiments = []
    for i, folder in enumerate(folders[:15]):
        folder_path = os.path.join(OUTPUT_DIR, folder)
        mtime = datetime.fromtimestamp(os.path.getmtime(folder_path)).strftime("%Y-%m-%d %H:%M")
        
        # Check for results files
        results_files = glob(os.path.join(folder_path, "results_*.json"))
        has_checkpoint = os.path.exists(os.path.join(folder_path, "checkpoint.json"))
        
        status = "✓" if results_files else ("⏸" if has_checkpoint else "?")
        print(f"  [{i}] {status} {folder} ({mtime})")
        experiments.append(folder)
    
    return experiments


def load_batch_experiment(name_or_index) -> Optional[dict]:
    """Load batch experiment results.
    
    Args:
        name_or_index: Experiment name (partial match) or index from list
        
    Returns:
        Loaded experiment dict with config, results, summary
    """
    if not os.path.exists(OUTPUT_DIR):
        print("❌ No outputs directory found")
        return None
    
    folders = [f for f in os.listdir(OUTPUT_DIR) 
               if os.path.isdir(os.path.join(OUTPUT_DIR, f))]
    folders = sorted(folders, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f)), reverse=True)
    
    # Find folder
    if isinstance(name_or_index, int):
        if name_or_index >= len(folders):
            print(f"❌ Index {name_or_index} out of range (max: {len(folders)-1})")
            return None
        folder = folders[name_or_index]
    else:
        matches = [f for f in folders if name_or_index.lower() in f.lower()]
        if not matches:
            print(f"❌ No experiment found matching: {name_or_index}")
            return None
        folder = matches[0]
    
    folder_path = os.path.join(OUTPUT_DIR, folder)
    
    # Find results file
    results_files = glob(os.path.join(folder_path, "results_*.json"))
    if not results_files:
        # Try loading checkpoint
        checkpoint_path = os.path.join(folder_path, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            print(f"⚠ No results file, loading checkpoint...")
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Loaded checkpoint: {folder} ({len(data.get('results', []))} questions)")
            return {"results": data.get("results", []), "config": {}, "_is_checkpoint": True}
        print(f"❌ No results or checkpoint in: {folder}")
        return None
    
    # Load most recent results
    results_file = max(results_files, key=os.path.getmtime)
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded: {folder}")
    print(f"  Questions: {data.get('summary', {}).get('total_questions', '?')}")
    print(f"  Samples: {data.get('summary', {}).get('total_samples', '?')}")
    print(f"  Accuracy: {data.get('summary', {}).get('overall_accuracy', 0):.1%}")
    
    return data


def print_batch_summary(exp: dict):
    """Print summary of batch experiment."""
    if not exp:
        print("No data")
        return
    
    summary = exp.get("summary", {})
    config = exp.get("config", {})
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print(f"\nModel: {config.get('model', {}).get('name', 'N/A')}")
    print(f"Questions: {summary.get('total_questions', '?')}")
    print(f"Samples per question: {config.get('samples', '?')}")
    print(f"Total samples: {summary.get('total_samples', '?')}")
    print(f"Overall accuracy: {summary.get('overall_accuracy', 0):.1%}")
    print(f"Total time: {summary.get('total_time_s', 0):.1f}s")
    
    # By category
    by_cat = summary.get("by_category", {})
    if by_cat:
        print("\nBy category:")
        for cat, acc in sorted(by_cat.items()):
            print(f"  {cat}: {acc:.1%}")


def inspect_question(exp: dict, question_idx: int = 0):
    """Inspect a specific question with all its samples."""
    results = exp.get("results", [])
    
    if question_idx >= len(results):
        print(f"❌ Index {question_idx} out of range (max: {len(results)-1})")
        return
    
    q = results[question_idx]
    
    print("\n" + "=" * 80)
    print(f"QUESTION {question_idx}")
    print("=" * 80)
    
    print(f"\nCategory: {q.get('category', '?')}")
    print(f"Context: {q.get('context', 'N/A')}")
    print(f"Question: {q.get('question', 'N/A')}")
    
    print("\nChoices:")
    choices = q.get("choices", [])
    correct = q.get("correct_answer", "?")
    for i, choice in enumerate(choices):
        letter = ["A", "B", "C"][i]
        marker = " ← correct" if letter == correct else ""
        print(f"  {letter}. {choice}{marker}")
    
    print(f"\nAccuracy: {q.get('accuracy', 0):.0%}")
    print(f"Answer distribution: {q.get('answer_distribution', {})}")
    
    # Sample summary
    samples = q.get("samples", [])
    print(f"\nSamples ({len(samples)}):")
    for s in samples:
        status = "✓" if s.get("correct") else "✗"
        print(f"  [{s.get('sample_idx', '?')}] {status} {s.get('answer', '?')} | {s.get('tokens', 0)} tokens | {s.get('time_s', 0):.1f}s")


def show_cot(exp: dict, question_idx: int = 0, sample_idx: int = 0, max_length: int = 3000):
    """Show full Chain-of-Thought for a specific sample."""
    results = exp.get("results", [])
    
    if question_idx >= len(results):
        print(f"❌ Question index {question_idx} out of range")
        return
    
    q = results[question_idx]
    samples = q.get("samples", [])
    
    if sample_idx >= len(samples):
        print(f"❌ Sample index {sample_idx} out of range")
        return
    
    s = samples[sample_idx]
    
    print("\n" + "=" * 80)
    print(f"COT: Question {question_idx}, Sample {sample_idx}")
    print("=" * 80)
    
    print(f"\nCategory: {q.get('category', '?')}")
    print(f"Question: {q.get('question', 'N/A')[:100]}...")
    print(f"Answer: {s.get('answer', '?')} | Correct: {q.get('correct_answer', '?')} | {'✓' if s.get('correct') else '✗'}")
    
    print("\n" + "-" * 80)
    print("FULL OUTPUT:")
    print("-" * 80)
    
    output = s.get("full_output", "")
    if max_length and len(output) > max_length:
        output = output[:max_length] + "\n... [truncated]"
    print(output)


def filter_wrong_samples(exp: dict) -> List[dict]:
    """Get all wrong samples with their question context."""
    results = exp.get("results", [])
    wrong = []
    
    for q_idx, q in enumerate(results):
        for s_idx, s in enumerate(q.get("samples", [])):
            if not s.get("correct"):
                wrong.append({
                    "question_idx": q_idx,
                    "sample_idx": s_idx,
                    "category": q.get("category"),
                    "question": q.get("question"),
                    "correct_answer": q.get("correct_answer"),
                    "model_answer": s.get("answer"),
                })
    
    return wrong


def accuracy_by_category(exp: dict) -> dict:
    """Get accuracy breakdown by category."""
    results = exp.get("results", [])
    by_cat = {}
    
    for q in results:
        cat = q.get("category", "unknown")
        if cat not in by_cat:
            by_cat[cat] = {"correct": 0, "total": 0}
        
        for s in q.get("samples", []):
            by_cat[cat]["total"] += 1
            if s.get("correct"):
                by_cat[cat]["correct"] += 1
    
    print("\nAccuracy by category:")
    for cat, stats in sorted(by_cat.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {cat}: {acc:.1%} ({stats['correct']}/{stats['total']})")
    
    return {cat: stats["correct"]/stats["total"] if stats["total"] > 0 else 0 
            for cat, stats in by_cat.items()}


def compare_experiments(experiments: List[dict], names: List[str] = None):
    """Compare multiple experiments side-by-side."""
    if not experiments:
        print("No experiments to compare")
        return
    
    if names is None:
        names = [f"Exp {i}" for i in range(len(experiments))]
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    
    # Overall accuracy
    print("\nOverall Accuracy:")
    for name, exp in zip(names, experiments):
        acc = exp.get("summary", {}).get("overall_accuracy", 0)
        print(f"  {name}: {acc:.1%}")
    
    # By category
    print("\nBy Category:")
    all_cats = set()
    for exp in experiments:
        all_cats.update(exp.get("summary", {}).get("by_category", {}).keys())
    
    for cat in sorted(all_cats):
        print(f"\n  {cat}:")
        for name, exp in zip(names, experiments):
            acc = exp.get("summary", {}).get("by_category", {}).get(cat, 0)
            print(f"    {name}: {acc:.1%}")

