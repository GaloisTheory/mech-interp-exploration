#!/usr/bin/env python3
"""Quick Internals Analysis - Interactive Cells

Run cells with Shift+Enter in VS Code/Cursor.
Model loads once, then tweak inline config and re-run analysis cells.
"""

#%% Model Loading (run once) ================================================
import os
import sys
import importlib

# Setup paths
_internals_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
os.chdir(_internals_dir)

# Import everything from logit_lens (this loads the model)
from logit_lens import (
    model,
    get_answer_token_ids,
    build_prompt,
    find_decision_point,
    run_logit_lens,
    plot_heatmap,
    plot_final_layer,
    plot_by_layer,
    run,
    load_bbq_items,
    format_bbq_prompt,
    FEW_SHOT_EXAMPLES,
    torch,
)
from transformers import AutoTokenizer
import config as cfg

# Load HF tokenizer to use the same chat template as HuggingFace generation
hf_tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)

print("Ready! Model loaded.")


#%% Inline Config (edit these) ==============================================
# Quick tweaks - edit and re-run cells below

CATEGORY = "appearance"
INDEX = 1
N_VALUES = [1, 10, 20, 25, 30]  # Blank space counts to test
FORCED_TEXT = " "  # What to inject (space, or e.g. "The answer is A! ")
USE_FEW_SHOT = True

#%% N-Sweep Analysis (table output) =========================================
# Loops over N_VALUES and prints ABC probabilities

# %%
items = load_bbq_items(categories=[CATEGORY], n_per_category=max(10, INDEX + 1))
bbq_item = items[INDEX]

print(f"\nQuestion: {bbq_item.question}")
print(f"Correct: {bbq_item.correct_letter}")
print(f"Choices: A. {bbq_item.choices[0]} | B. {bbq_item.choices[1]} | C. {bbq_item.choices[2]}")

bbq_prompt = format_bbq_prompt(context=bbq_item.context, question=bbq_item.question, choices=bbq_item.choices)
if USE_FEW_SHOT:
    bbq_prompt = FEW_SHOT_EXAMPLES + bbq_prompt

A_sp, B_sp, C_sp = get_answer_token_ids(model)
A_id, B_id, C_id = model.to_single_token("A"), model.to_single_token("B"), model.to_single_token("C")
all_answer_ids = [A_id, B_id, C_id, A_sp, B_sp, C_sp]

# Trailing \n\n after </think> - Model naturally generates this before answering
# The model predicts '\n\n' after </think>, then predicts the answer letter
#
# NOTE: We discovered a BUG in shared/generation.py's generate_with_custom_override!
# After processing override tokens, it continues from override_ids[:, -1:] which
# processes the last token TWICE, corrupting the KV cache and producing wrong answers.
# This is why blank_vs_accuracy.py showed 0% accuracy at low blank counts.
#
# TransformerLens (this script) correctly encodes prompt+override in one pass.
# HuggingFace and TransformerLens produce identical outputs (cosine sim ~0.98)
# when the prompt is constructed correctly.
INCLUDE_TRAILING_NEWLINE = True

# Use apply_chat_template with enable_thinking=True to match HuggingFace generation
messages = [{"role": "user", "content": bbq_prompt}]
base_prompt = hf_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
print(f"Base prompt ends with: {repr(base_prompt[-50:])}")

print(f"\n{'N':<5} {'ABC Mass':<10} {'P(A)':<10} {'P(B)':<10} {'P(C)':<10} {'Top':<6} {'Correct?'}")
print("-" * 70)

for N in N_VALUES:
    suffix = "\n\n" if INCLUDE_TRAILING_NEWLINE else ""
    # Append <think> + blanks + </think> + newlines to the template-formatted prompt
    prompt = base_prompt + "<think>" + FORCED_TEXT * N + "</think>" + suffix
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    # Get final layer logits
    resid = cache["resid_post", model.cfg.n_layers - 1][0]
    resid = model.ln_final(resid)
    logits = resid @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    
    pos_logits = logits[-1]
    probs = torch.softmax(pos_logits.float(), dim=-1)
    
    p_a = probs[A_id].item() + probs[A_sp].item()
    p_b = probs[B_id].item() + probs[B_sp].item()
    p_c = probs[C_id].item() + probs[C_sp].item()
    abc_mass = sum(probs[tid].item() for tid in all_answer_ids)
    
    top_idx = pos_logits.argmax().item()
    top_token = model.to_single_str_token(top_idx)
    correct = "✓" if top_token.strip() == bbq_item.correct_letter else "✗"
    
    print(f"{N:<5} {abc_mass:<10.4f} {p_a:<10.4f} {p_b:<10.4f} {p_c:<10.4f} {top_token:<6} {correct}")


#%% Heatmap Visualization (full logit lens) =================================
# Uses config.py settings - reload first if you changed it
print(prompt)
# %%

importlib.reload(cfg)

# %%
print(f"Running with: {cfg.BBQ_CATEGORY} #{cfg.BBQ_INDEX}, '{cfg.FORCED_COT_TEXT}' x{cfg.FORCED_COT_REPEATS}")
run(cfg)


#%% Reload Config ===========================================================
# Run this cell to hot-reload config.py after editing it

importlib.reload(cfg)
print(f"Config reloaded:")
print(f"  Category: {cfg.BBQ_CATEGORY}, Index: {cfg.BBQ_INDEX}")
print(f"  Forced CoT: '{cfg.FORCED_COT_TEXT}' x {cfg.FORCED_COT_REPEATS}")
print(f"  Few-shot: {cfg.USE_FEW_SHOT}")

