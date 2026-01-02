#!/usr/bin/env python3
"""Blank Spaces vs Accuracy Analysis

Systematically varies the number of blank space repetitions in the thinking
override and measures accuracy across 100 samples per setting.

Usage:
    python blank_vs_accuracy.py
"""

import os
import sys
import time
from collections import Counter

# =============================================================================
# SETUP PATHS
# =============================================================================
_interactive_dir = os.path.dirname(os.path.abspath(__file__))
_bbq_dir = os.path.dirname(_interactive_dir)
_shared_dir = os.path.dirname(_bbq_dir)
_results_dir = os.path.join(_bbq_dir, "results_graphs")

for path in [_interactive_dir, _bbq_dir, _shared_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(_bbq_dir)

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.config import format_bbq_prompt
from shared.generation import generate_with_custom_override
from data.bbq_dataset import load_bbq_items
from constants import FEW_SHOT_EXAMPLES

import config as cfg

# =============================================================================
# CONFIGURATION
# =============================================================================
BLANK_COUNTS = list(range(0, 51, 5))  # [0, 5, 10, ..., 95, 100]
NUM_SAMPLES = 300
OUTPUT_PATH = os.path.join(_results_dir, "blank_vs_accuracy.png")

# =============================================================================
# LOAD MODEL
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
# LOAD BBQ QUESTION
# =============================================================================
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
prompt = FEW_SHOT_EXAMPLES + base_prompt

print(f"\n=== BBQ Question ({cfg.BBQ_CATEGORY} #{cfg.BBQ_INDEX}) ===")
print(f"Context: {bbq_item.context}")
print(f"Question: {bbq_item.question}")
print(f"Options: A. {bbq_item.choices[0]} | B. {bbq_item.choices[1]} | C. {bbq_item.choices[2]}")
print(f"Correct: {bbq_item.correct_letter}")
print("=" * 60)


# =============================================================================
# RUN SWEEP
# =============================================================================
def run_samples(blank_count: int) -> dict:
    """Run NUM_SAMPLES generations with given blank count."""
    override_text = "<think>" + " " * blank_count + "</think>"
    
    answers = []
    for _ in range(NUM_SAMPLES):
        result = generate_with_custom_override(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            token_to_match="<think>",
            override_text=override_text,
            max_tokens=cfg.MAX_TOKENS,
            intercept_count=1,
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            streaming=False,
            token_position_overrides=[],
            model_name=cfg.MODEL_NAME,
            enable_thinking=cfg.ENABLE_THINKING,
        )
        answers.append(result.answer)
    
    correct_count = sum(1 for a in answers if a == bbq_item.correct_letter)
    accuracy = correct_count / NUM_SAMPLES
    dist = dict(Counter(answers))
    
    return {
        "blank_count": blank_count,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "distribution": dist,
    }


print(f"\n{'='*60}")
print(f"RUNNING SWEEP: {len(BLANK_COUNTS)} blank counts x {NUM_SAMPLES} samples each")
print(f"{'='*60}\n")

results = []
start_time = time.time()

for i, blank_count in enumerate(BLANK_COUNTS):
    iter_start = time.time()
    result = run_samples(blank_count)
    iter_elapsed = time.time() - iter_start
    
    results.append(result)
    
    print(f"[{i+1}/{len(BLANK_COUNTS)}] blanks={blank_count:3d} | "
          f"accuracy={result['accuracy']*100:5.1f}% | "
          f"dist={result['distribution']} | "
          f"time={iter_elapsed:.1f}s")

total_elapsed = time.time() - start_time
print(f"\nTotal time: {total_elapsed/60:.1f} minutes")


# =============================================================================
# PLOT RESULTS
# =============================================================================
print(f"\n{'='*60}")
print("GENERATING PLOT")
print(f"{'='*60}")

x_values = [r["blank_count"] for r in results]
y_values = [r["accuracy"] * 100 for r in results]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_values, y_values, 'b-o', linewidth=2, markersize=6)

ax.set_xlabel('Number of Blank Spaces', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title(f'Blank Spaces vs Accuracy\n{cfg.MODEL_NAME} {NUM_SAMPLES} samples each', 
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_xlim(-2, 102)
ax.grid(True, alpha=0.3)

# Add value labels on points
for x, y in zip(x_values, y_values):
    ax.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                xytext=(0, 8), ha='center', fontsize=8)

plt.tight_layout()

os.makedirs(_results_dir, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {OUTPUT_PATH}")

plt.show()

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Blanks':>8} | {'Accuracy':>10} | {'Correct':>8}")
print("-" * 35)
for r in results:
    print(f"{r['blank_count']:>8} | {r['accuracy']*100:>9.1f}% | {r['correct_count']:>8}/{NUM_SAMPLES}")

