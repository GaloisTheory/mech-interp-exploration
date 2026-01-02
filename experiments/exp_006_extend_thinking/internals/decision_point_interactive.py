#!/usr/bin/env python3
"""Interactive Decision Point Analysis

Analyze where ABC probability mass appears for different blank space counts.
Model loads once, then you can test different N values interactively.

Usage:
    python decision_point_interactive.py
"""

import os
import sys

# Setup paths
_internals_dir = os.path.dirname(os.path.abspath(__file__))
_exp_dir = os.path.dirname(_internals_dir)
_bbq_dir = os.path.join(_exp_dir, "bbq")

for path in [_internals_dir, _exp_dir, _bbq_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(_internals_dir)

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

import torch
from transformer_lens import HookedTransformer

from data.bbq_dataset import load_bbq_items
from shared.config import format_bbq_prompt
from constants import FEW_SHOT_EXAMPLES

# =============================================================================
# CONFIG - Edit these
# =============================================================================
MODEL_NAME = "Qwen/Qwen3-8B"
BBQ_CATEGORY = "appearance"
BBQ_INDEX = 1

# =============================================================================
# Load Model (once)
# =============================================================================
print(f"Loading model: {MODEL_NAME}")
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(f"Model loaded: {model.cfg.n_layers} layers")

# Load BBQ question
items = load_bbq_items(categories=[BBQ_CATEGORY], n_per_category=max(10, BBQ_INDEX + 1))
bbq_item = items[BBQ_INDEX]

print(f"\n=== BBQ Question ({BBQ_CATEGORY} #{BBQ_INDEX}) ===")
print(f"Context: {bbq_item.context}")
print(f"Question: {bbq_item.question}")
print(f"Choices: A. {bbq_item.choices[0]} | B. {bbq_item.choices[1]} | C. {bbq_item.choices[2]}")
print(f"Correct: {bbq_item.correct_letter}")

# Prepare prompt template
bbq_prompt = format_bbq_prompt(
    context=bbq_item.context,
    question=bbq_item.question,
    choices=bbq_item.choices,
)
full_prompt = FEW_SHOT_EXAMPLES + bbq_prompt


# =============================================================================
# Helper Functions
# =============================================================================
def get_answer_token_ids():
    """Get token IDs for A, B, C (both with and without space)."""
    A_sp = model.to_single_token(" A")
    B_sp = model.to_single_token(" B")
    C_sp = model.to_single_token(" C")
    A_id = model.to_single_token("A")
    B_id = model.to_single_token("B")
    C_id = model.to_single_token("C")
    return A_id, B_id, C_id, A_sp, B_sp, C_sp


def find_decision_point(cache, tokens, n_positions=15):
    """Find where ABC probability mass is highest."""
    A_id, B_id, C_id, A_sp, B_sp, C_sp = get_answer_token_ids()
    all_answer_ids = [A_id, B_id, C_id, A_sp, B_sp, C_sp]
    
    final_layer = model.cfg.n_layers - 1
    resid = cache["resid_post", final_layer][0]
    resid = model.ln_final(resid)
    logits = resid @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    
    token_strs = model.to_str_tokens(tokens[0])
    seq_len = len(token_strs)
    start_pos = max(0, seq_len - n_positions)
    
    print(f"\n{'Pos':<5} {'Token':<15} {'ABC Mass':<10} {'P(A)':<8} {'P(B)':<8} {'P(C)':<8} {'Top Token':<15}")
    print("-" * 80)
    
    for pos in range(start_pos, seq_len):
        pos_logits = logits[pos]
        probs = torch.softmax(pos_logits.float(), dim=-1)
        
        p_a = max(probs[A_id].item(), probs[A_sp].item())
        p_b = max(probs[B_id].item(), probs[B_sp].item())
        p_c = max(probs[C_id].item(), probs[C_sp].item())
        abc_mass = sum(probs[tid].item() for tid in all_answer_ids)
        
        top_idx = pos_logits.argmax().item()
        top_logit = pos_logits[top_idx].item()
        top_token = model.to_single_str_token(top_idx)
        top_token_escaped = repr(top_token)[1:-1][:12]
        
        tok_escaped = repr(token_strs[pos])[1:-1][:13]
        
        marker = "<<<" if abc_mass > 0.3 else ""
        print(f"{pos:<5} {tok_escaped:<15} {abc_mass:<10.4f} {p_a:<8.4f} {p_b:<8.4f} {p_c:<8.4f} {top_token_escaped:<15} {marker}")
    
    print("-" * 80)
    return logits[-1]  # Return last position logits


def analyze_n(n_blanks, show_table=True):
    """Analyze decision point for N blank spaces."""
    # Build prompt with N blanks
    prompt = (
        f"<|im_start|>user\n{full_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>{' '*n_blanks}</think>\n"
    )
    
    tokens = model.to_tokens(prompt)
    
    print(f"\n{'='*60}")
    print(f"N = {n_blanks} blank spaces ({tokens.shape[1]} tokens)")
    print(f"{'='*60}")
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    if show_table:
        find_decision_point(cache, tokens, n_positions=10)
    
    # Get final position probs
    A_id, B_id, C_id, A_sp, B_sp, C_sp = get_answer_token_ids()
    all_ids = [A_id, B_id, C_id, A_sp, B_sp, C_sp]
    
    resid = cache["resid_post", model.cfg.n_layers - 1][0, -1]
    resid = model.ln_final(resid)
    logits = resid @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    
    probs = torch.softmax(logits.float(), dim=-1)
    p_a = max(probs[A_id].item(), probs[A_sp].item())
    p_b = max(probs[B_id].item(), probs[B_sp].item())
    p_c = max(probs[C_id].item(), probs[C_sp].item())
    abc_mass = sum(probs[tid].item() for tid in all_ids)
    
    top_idx = logits.argmax().item()
    top_token = model.to_single_str_token(top_idx)
    
    correct = "✓" if top_token == bbq_item.correct_letter else "✗"
    print(f"\nResult: Top={top_token} {correct}  |  P(A)={p_a:.2%}  P(B)={p_b:.2%}  P(C)={p_c:.2%}  |  ABC={abc_mass:.2%}")
    
    return top_token, p_a, p_b, p_c, abc_mass


def sweep(n_values, show_tables=False):
    """Run analysis for multiple N values."""
    print(f"\n{'N':<6} {'Top':<5} {'P(A)':<10} {'P(B)':<10} {'P(C)':<10} {'ABC Mass':<10} {'Correct'}")
    print("-" * 65)
    
    for n in n_values:
        prompt = (
            f"<|im_start|>user\n{full_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>{' '*n}</think>\n"
        )
        tokens = model.to_tokens(prompt)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        A_id, B_id, C_id, A_sp, B_sp, C_sp = get_answer_token_ids()
        all_ids = [A_id, B_id, C_id, A_sp, B_sp, C_sp]
        
        resid = cache["resid_post", model.cfg.n_layers - 1][0, -1]
        resid = model.ln_final(resid)
        logits = resid @ model.W_U
        if model.b_U is not None:
            logits = logits + model.b_U
        
        probs = torch.softmax(logits.float(), dim=-1)
        p_a = max(probs[A_id].item(), probs[A_sp].item())
        p_b = max(probs[B_id].item(), probs[B_sp].item())
        p_c = max(probs[C_id].item(), probs[C_sp].item())
        abc_mass = sum(probs[tid].item() for tid in all_ids)
        
        top_idx = logits.argmax().item()
        top_token = model.to_single_str_token(top_idx)
        correct = "✓" if top_token == bbq_item.correct_letter else "✗"
        
        print(f"{n:<6} {top_token:<5} {p_a:<10.4f} {p_b:<10.4f} {p_c:<10.4f} {abc_mass:<10.4f} {correct}")


# =============================================================================
# Interactive Loop
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DECISION POINT INTERACTIVE ANALYSIS")
    print("=" * 60)
    print("\nCommands:")
    print("  <number>     - Analyze single N value (e.g., '20')")
    print("  sweep        - Run sweep for [0,5,10,15,20,25,30,35,40,45,50]")
    print("  sweep 0-50   - Custom range sweep")
    print("  q            - Quit")
    print()
    
    while True:
        try:
            cmd = input(">>> ").strip()
            
            if not cmd:
                continue
            elif cmd.lower() == 'q':
                print("Goodbye!")
                break
            elif cmd.lower() == 'sweep':
                sweep([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
            elif cmd.lower().startswith('sweep '):
                # Parse range like "sweep 0-50" or "sweep 0,10,20"
                range_str = cmd[6:].strip()
                if '-' in range_str:
                    start, end = map(int, range_str.split('-'))
                    sweep(list(range(start, end + 1, 5)))
                elif ',' in range_str:
                    vals = [int(x.strip()) for x in range_str.split(',')]
                    sweep(vals)
                else:
                    print("Usage: sweep 0-50 or sweep 0,10,20,30")
            else:
                try:
                    n = int(cmd)
                    analyze_n(n)
                except ValueError:
                    print(f"Unknown command: {cmd}")
                    
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")



