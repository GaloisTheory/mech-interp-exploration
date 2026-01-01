#!/usr/bin/env python3
"""BBQ Batch Experiment Runner

Run batch experiments with configurable models, prompts, overrides, and questions.
Results include full Chain-of-Thought traces for each sample.

Usage:
    python run_question_batch.py configs/my_experiment.yaml
    python run_question_batch.py configs/my_experiment.yaml --resume
    python run_question_batch.py configs/my_experiment.yaml --dry-run
"""

import os
import sys
import time
import argparse
from datetime import datetime
from collections import Counter
from tqdm import tqdm

# =============================================================================
# SETUP PATHS
# =============================================================================
_batch_dir = os.path.dirname(os.path.abspath(__file__))
_bbq_dir = os.path.dirname(_batch_dir)
_exp_dir = os.path.dirname(_bbq_dir)

for path in [_batch_dir, _bbq_dir, _exp_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(_bbq_dir)

# =============================================================================
# IMPORTS
# =============================================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.generation import generate_with_custom_override
from data.bbq_dataset import load_bbq_items, ALL_CATEGORIES

from batch_utils import (
    load_config,
    save_config,
    build_prompt,
    make_override_fn,
    save_checkpoint,
    load_checkpoint,
    clear_checkpoint,
    save_results,
    OUTPUT_DIR,
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run BBQ batch experiment")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Show questions without running")
    return parser.parse_args()


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================
def run_experiment(config_path: str, resume: bool = False, dry_run: bool = False):
    """Run a batch experiment from config file."""
    
    # Load config
    print("=" * 60)
    print("BBQ BATCH EXPERIMENT")
    print("=" * 60)
    
    # Resolve config path relative to batch dir if not absolute
    if not os.path.isabs(config_path):
        config_path = os.path.join(_batch_dir, config_path)
    
    config = load_config(config_path)
    exp_name = config.get("name", "unnamed")
    
    print(f"\nExperiment: {exp_name}")
    print(f"Config: {config_path}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, f"{exp_name}_{timestamp}")
    
    if resume:
        # Find existing output dir for this experiment
        existing = [d for d in os.listdir(OUTPUT_DIR) if d.startswith(exp_name)]
        if existing:
            output_dir = os.path.join(OUTPUT_DIR, sorted(existing)[-1])
            print(f"Resuming from: {output_dir}")
        else:
            print("No existing experiment to resume, starting fresh")
            resume = False
    
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        save_config(config, output_dir)
    
    # Load questions
    print("\n" + "-" * 60)
    print("LOADING QUESTIONS")
    print("-" * 60)
    
    questions_cfg = config.get("questions", {})
    source = questions_cfg.get("source", "bbq")
    
    if source == "bbq":
        categories = questions_cfg.get("categories", ALL_CATEGORIES)
        n_per_category = questions_cfg.get("n_per_category", 10)
        seed = questions_cfg.get("seed", 42)
        
        items = load_bbq_items(
            categories=categories,
            n_per_category=n_per_category,
            ambiguous_only=True,
            seed=seed,
        )
        print(f"Loaded {len(items)} BBQ questions from {len(categories)} categories")
    else:
        # Custom questions (future extension)
        raise NotImplementedError("Custom questions not yet supported")
    
    if dry_run:
        print("\n" + "-" * 60)
        print("DRY RUN - Questions to be processed:")
        print("-" * 60)
        for i, item in enumerate(items[:20]):
            print(f"[{i}] {item.category}: {item.question[:60]}...")
        if len(items) > 20:
            print(f"... and {len(items) - 20} more")
        print(f"\nTotal: {len(items)} questions × {config.get('samples', 10)} samples = {len(items) * config.get('samples', 10)} generations")
        return
    
    # Load model
    print("\n" + "-" * 60)
    print("LOADING MODEL")
    print("-" * 60)
    
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "Qwen/Qwen3-8B")
    
    print(f"Loading: {model_name}")
    print("This may take a minute...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    print(f"Model loaded on {model.device}")
    
    # Setup generation config
    gen_cfg = config.get("generation", {})
    temperature = gen_cfg.get("temperature", 0.6)
    top_p = gen_cfg.get("top_p", 0.95)
    max_tokens = gen_cfg.get("max_tokens", 1000)
    enable_thinking = gen_cfg.get("enable_thinking", True)
    use_few_shot = gen_cfg.get("use_few_shot", True)
    
    # Setup override config
    override_cfg = config.get("override", {})
    token_to_match = override_cfg.get("token_to_match", "</think>")
    intercept_count = override_cfg.get("intercept_count", 0)
    schedule = override_cfg.get("schedule", [])
    position_overrides = override_cfg.get("position_overrides", [])
    
    override_fn = make_override_fn(schedule)
    
    num_samples = config.get("samples", 10)
    
    print("\n" + "-" * 60)
    print("GENERATION CONFIG")
    print("-" * 60)
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print(f"Max tokens: {max_tokens}")
    print(f"Enable thinking: {enable_thinking}")
    print(f"Few-shot: {use_few_shot}")
    print(f"Token to match: {token_to_match}")
    print(f"Intercept count: {intercept_count}")
    print(f"Samples per question: {num_samples}")
    
    # Load checkpoint if resuming
    results = []
    start_idx = 0
    
    if resume:
        results, last_idx = load_checkpoint(output_dir)
        if last_idx >= 0:
            start_idx = last_idx + 1
            print(f"\nResuming from question {start_idx} ({len(results)} completed)")
    
    # Run experiment
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT")
    print("=" * 60)
    
    started_at = datetime.now()
    
    total_items = len(items)
    
    for q_idx in tqdm(range(start_idx, total_items), desc="Questions", initial=start_idx, total=total_items):
        item = items[q_idx]
        
        # Build prompt
        prompt = build_prompt(item, use_few_shot=use_few_shot)
        
        # Run samples
        samples = []
        for s_idx in range(num_samples):
            start_time = time.time()
            
            result = generate_with_custom_override(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                token_to_match=token_to_match,
                override_text=override_fn,
                max_tokens=max_tokens,
                intercept_count=intercept_count,
                temperature=temperature,
                top_p=top_p,
                streaming=False,
                token_position_overrides=position_overrides,
                model_name=model_name,
                enable_thinking=enable_thinking,
            )
            
            elapsed = time.time() - start_time
            
            samples.append({
                "sample_idx": s_idx,
                "answer": result.answer,
                "correct": result.answer == item.correct_letter,
                "tokens": result.token_count,
                "time_s": round(elapsed, 2),
                "full_output": result.full_output,
            })
        
        # Compute question-level stats
        answers = [s["answer"] for s in samples]
        answer_dist = dict(Counter(answers))
        correct_count = sum(1 for s in samples if s["correct"])
        accuracy = correct_count / len(samples)
        
        question_result = {
            "question_idx": q_idx,
            "category": item.category,
            "context": item.context,
            "question": item.question,
            "choices": item.choices,
            "correct_answer": item.correct_letter,
            "samples": samples,
            "accuracy": accuracy,
            "answer_distribution": answer_dist,
        }
        
        results.append(question_result)
        
        # Save checkpoint
        save_checkpoint(output_dir, results, q_idx)
    
    completed_at = datetime.now()
    
    # Save final results
    print("\n" + "-" * 60)
    print("SAVING RESULTS")
    print("-" * 60)
    
    results_path = save_results(output_dir, config, results, started_at, completed_at)
    clear_checkpoint(output_dir)
    
    # Print summary
    total_samples = sum(len(r["samples"]) for r in results)
    total_correct = sum(sum(1 for s in r["samples"] if s["correct"]) for r in results)
    total_time = (completed_at - started_at).total_seconds()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nQuestions: {len(results)}")
    print(f"Total samples: {total_samples}")
    print(f"Overall accuracy: {total_correct}/{total_samples} ({100*total_correct/total_samples:.1f}%)")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg time per sample: {total_time/total_samples:.2f}s")
    print(f"\nResults saved to: {results_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.config, resume=args.resume, dry_run=args.dry_run)

