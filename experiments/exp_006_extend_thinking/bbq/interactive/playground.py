#!/usr/bin/env python3
"""Interactive BBQ Playground - Fast Config Iteration

Run this script once to load the model, then press Enter to re-run
with updated config. Edit config.py between runs.

Usage:
    python playground.py

Workflow:
    1. Script loads model (once)
    2. Reads config.py and runs generation
    3. Waits for Enter keypress
    4. Hot-reloads config.py and runs again
    5. Ctrl+C to exit
"""

import os
import sys
import time
import importlib
from collections import Counter

# =============================================================================
# SETUP PATHS
# =============================================================================
_interactive_dir = os.path.dirname(os.path.abspath(__file__))
_bbq_dir = os.path.dirname(_interactive_dir)
_shared_dir = os.path.dirname(_bbq_dir)

for path in [_interactive_dir, _bbq_dir, _shared_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(_bbq_dir)

# =============================================================================
# IMPORTS
# =============================================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.config import format_bbq_prompt
from shared.generation import generate_with_custom_override
from data.bbq_dataset import load_bbq_items, ALL_CATEGORIES
from constants import FEW_SHOT_EXAMPLES

import config as cfg

# =============================================================================
# LOAD MODEL (runs once)
# =============================================================================
print(f"Loading model: {cfg.MODEL_NAME}")
print("This may take a minute...")

tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg.MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
)
print(f"Model loaded on {model.device}")
print("=" * 60)


# =============================================================================
# GENERATION FUNCTION
# =============================================================================
def get_override_for_intercept(intercept_num: int, schedule: list) -> str:
    """Get the override text for a given intercept number (1-indexed)."""
    for start, end, text in schedule:
        if start <= intercept_num <= end:
            return text
    return "\n\nWait, let me reconsider..."


def run_single(cfg, prompt, bbq_item, override_fn, sample_num=None, show_thinking=True):
    """Run a single generation. Returns (result, elapsed_time)."""
    if sample_num is not None:
        print(f"\n--- Sample {sample_num} ---")
    
    start_time = time.time()
    result = generate_with_custom_override(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        token_to_match=cfg.TOKEN_TO_MATCH,
        override_text=override_fn,
        max_tokens=cfg.MAX_TOKENS,
        intercept_count=cfg.INTERCEPT_COUNT,
        temperature=cfg.TEMPERATURE,
        top_p=cfg.TOP_P,
        streaming=cfg.STREAMING,
        token_position_overrides=cfg.TOKEN_POSITION_OVERRIDES,
        model_name=cfg.MODEL_NAME,
        enable_thinking=cfg.ENABLE_THINKING,
    )
    elapsed = time.time() - start_time
    if show_thinking: 
        print("Chain of Thought:")
        print("-" * 60)
        print(result.full_output)
        print("-" * 60)
    # Display results
    tokens_per_sec = result.token_count / elapsed if elapsed > 0 else 0
    print(f"Answer: {result.answer} | Tokens: {result.token_count} | Time: {elapsed:.1f}s ({tokens_per_sec:.1f} tok/s)")
    
    if bbq_item:
        print(f"Correct: {bbq_item.correct_letter} | {'✓' if result.answer == bbq_item.correct_letter else '✗'}")
    
    return result, elapsed


def run(cfg):
    """Run generation with current config (supports multiple samples)."""
    num_samples = getattr(cfg, 'NUM_SAMPLES', 1)
    
    # Build prompt
    if cfg.USE_BBQ:
        items = load_bbq_items(categories=[cfg.BBQ_CATEGORY], n_per_category=max(10, cfg.BBQ_INDEX + 1))
        if cfg.BBQ_INDEX >= len(items):
            print(f"Warning: BBQ_INDEX {cfg.BBQ_INDEX} out of range, using 0")
            bbq_item = items[0]
        else:
            bbq_item = items[cfg.BBQ_INDEX]
        
        base_prompt = format_bbq_prompt(
            context=bbq_item.context,
            question=bbq_item.question,
            choices=bbq_item.choices,
        )
        
        print(f"\n=== BBQ Question ({cfg.BBQ_CATEGORY} #{cfg.BBQ_INDEX}) ===")
        print(f"Context: {bbq_item.context}")
        print(f"Question: {bbq_item.question}")
        print(f"Options: A. {bbq_item.choices[0]} | B. {bbq_item.choices[1]} | C. {bbq_item.choices[2]}")
        print(f"Correct: {bbq_item.correct_letter}")
    else:
        base_prompt = cfg.CUSTOM_PROMPT
        bbq_item = None
        print("\n=== Custom Prompt ===")
        print(base_prompt)
    
    # Add few-shot examples
    prompt = FEW_SHOT_EXAMPLES + base_prompt if cfg.USE_FEW_SHOT else base_prompt
    
    print("\n" + "=" * 60)
    print("GENERATION CONFIG")
    print("=" * 60)
    print(f"Model: {cfg.MODEL_NAME}")
    print(f"Temperature: {cfg.TEMPERATURE}, Top-p: {cfg.TOP_P}")
    print(f"Intercept count: {cfg.INTERCEPT_COUNT}")
    print(f"Max tokens: {cfg.MAX_TOKENS}")
    print(f"Num samples: {num_samples}")
    print(f"Streaming: {cfg.STREAMING}")
    
    # Create override function with current schedule
    override_fn = lambda n: get_override_for_intercept(n, cfg.OVERRIDE_SCHEDULE)
    
    # Run generation(s)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    show_thinking = getattr(cfg, 'SHOW_THINKING', False)
    results = []
    times = []
    for i in range(num_samples):
        sample_num = i + 1 if num_samples > 1 else None
        result, elapsed = run_single(cfg, prompt, bbq_item, override_fn, sample_num, show_thinking)
        results.append(result)
        times.append(elapsed)
    
    # Show summary if multiple samples
    if num_samples > 1:
        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        answers = [r.answer for r in results]
        counts = Counter(answers)
        print(f"Answer distribution: {dict(counts)}")
        if bbq_item:
            correct_count = sum(1 for a in answers if a == bbq_item.correct_letter)
            print(f"Accuracy: {correct_count}/{num_samples} ({100*correct_count/num_samples:.1f}%)")
        
        # Timing stats
        total_tokens = sum(r.token_count for r in results)
        total_time = sum(times)
        avg_time = total_time / num_samples
        avg_tokens = total_tokens / num_samples
        print(f"Timing: {total_time:.1f}s total | {avg_time:.1f}s avg | {total_tokens/total_time:.1f} tok/s avg")
    
    result = results[-1]  # Return last result for compatibility
    
    # Optional comparison run
    if cfg.RUN_COMPARISON:
        print("\n" + "=" * 60)
        print("COMPARISON (without override)")
        print("=" * 60)
        
        result_no_override = generate_with_custom_override(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            token_to_match=cfg.TOKEN_TO_MATCH,
            override_text=override_fn,
            max_tokens=cfg.MAX_TOKENS,
            intercept_count=0,  # No override
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            streaming=False,
            model_name=cfg.MODEL_NAME,
            enable_thinking=cfg.ENABLE_THINKING,
            token_position_overrides=[],
        )
        
        print(f"Without Override: Answer = {result_no_override.answer} ({result_no_override.token_count} tokens)")
        print(f"With Override:    Answer = {result.answer} ({result.token_count} tokens)")
        
        if bbq_item:
            print(f"Correct:          {bbq_item.correct_letter}")
            print(f"\nNo Override: {'✓ Correct' if result_no_override.answer == bbq_item.correct_letter else '✗ Wrong'}")
            print(f"With Override: {'✓ Correct' if result.answer == bbq_item.correct_letter else '✗ Wrong'}")
    
    return result


# =============================================================================
# INTERACTIVE LOOP
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INTERACTIVE PLAYGROUND")
    print("=" * 60)
    print("Edit config.py to change settings, then press Enter to re-run.")
    print("Press Ctrl+C to exit.\n")
    
    while True:
        try:
            # Hot-reload config
            importlib.reload(cfg)
            print(f"\n[Config reloaded from config.py]")
            
            # Run generation
            run(cfg)
            
            # Wait for user
            input("\n>>> Press Enter to re-run with updated config (Ctrl+C to exit)...")
            print("\n" + "=" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            print("Fix the error in config.py and press Enter to retry.")
            input("\n>>> Press Enter to retry...")

