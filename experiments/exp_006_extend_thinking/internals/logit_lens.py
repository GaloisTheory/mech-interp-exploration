#!/usr/bin/env python3
"""Logit Lens Analysis for Unfaithful Reasoning

Analyze WHERE (which layer) and WHEN (which token position) the model's
internal representation diverges from forced reasoning text.

Usage:
    python logit_lens.py

Workflow:
    1. Script loads model (once)
    2. Reads config.py and runs logit lens analysis
    3. Generates visualizations
    4. Waits for Enter keypress
    5. Hot-reloads config.py and runs again
    6. Ctrl+C to exit
"""

#%% Imports and Setup
import os
import sys
import importlib

# Set up paths
_internals_dir = os.path.dirname(os.path.abspath(__file__))
_exp_dir = os.path.dirname(_internals_dir)
_bbq_dir = os.path.join(_exp_dir, "bbq")

for path in [_internals_dir, _exp_dir, _bbq_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(_internals_dir)

# Set HuggingFace cache
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

from data.bbq_dataset import load_bbq_items
from shared.config import format_bbq_prompt
from constants import FEW_SHOT_EXAMPLES

import config as cfg

# Create outputs directory
_outputs_dir = os.path.join(_internals_dir, "outputs")
os.makedirs(_outputs_dir, exist_ok=True)

#%% Load Model (runs once)
print(f"Loading model: {cfg.MODEL_NAME}")
print("This may take a minute...")

model = HookedTransformer.from_pretrained(
    cfg.MODEL_NAME,
    dtype=torch.bfloat16,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dim")
print("=" * 60)


#%% Helper Functions
def get_answer_token_ids(model):
    """Get token IDs for A, B, C answers."""
    # Try with space prefix (common for answer tokens)
    try:
        A_id = model.to_single_token(" A")
        B_id = model.to_single_token(" B")
        C_id = model.to_single_token(" C")
        return A_id, B_id, C_id
    except Exception:
        # Fallback: try without space
        A_id = model.to_single_token("A")
        B_id = model.to_single_token("B")
        C_id = model.to_single_token("C")
        return A_id, B_id, C_id


def build_prompt(bbq_item, forced_cot_text, forced_cot_repeats, use_few_shot=True):
    """Build prompt with forced CoT injection."""
    # Format the BBQ question
    bbq_prompt = format_bbq_prompt(
        context=bbq_item.context,
        question=bbq_item.question,
        choices=bbq_item.choices,
    )
    
    # Add few-shot examples (forces model to output A, B, or C)
    if use_few_shot:
        bbq_prompt = FEW_SHOT_EXAMPLES + bbq_prompt
    
    # Build the forced CoT
    forced_cot = forced_cot_text * forced_cot_repeats
    
    # Qwen3 chat template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>...
    prompt = (
        f"<|im_start|>user\n{bbq_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>{forced_cot}</think>\n"
    )
    
    return prompt, bbq_prompt, forced_cot


def find_think_start_idx(token_strs):
    """Find the token index where <think> starts."""
    for i, tok in enumerate(token_strs):
        if "<think>" in tok or tok == "<think>":
            return i
    return None


def find_decision_point(cache, model, tokens, n_positions=15):
    """Find where ABC probability mass is highest in the last N positions.
    
    Helps identify the TRUE decision point where the model commits to A/B/C.
    """
    # Get both variants: with and without space prefix
    A_sp, B_sp, C_sp = get_answer_token_ids(model)  # " A", " B", " C"
    A_id = model.to_single_token("A")
    B_id = model.to_single_token("B") 
    C_id = model.to_single_token("C")
    all_answer_ids = [A_id, B_id, C_id, A_sp, B_sp, C_sp]
    
    final_layer = model.cfg.n_layers - 1
    resid = cache["resid_post", final_layer][0]  # [seq_len, d_model]
    
    # Apply ln_final and unembed
    resid = model.ln_final(resid)
    logits = resid @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    
    # Get token strings
    token_strs = model.to_str_tokens(tokens[0])
    seq_len = len(token_strs)
    start_pos = max(0, seq_len - n_positions)
    
    print(f"\n=== Decision Point Analysis (last {n_positions} positions) ===")
    print(f"{'Pos':<5} {'Token':<15} {'ABC Mass':<10} {'P(A)':<8} {'P(B)':<8} {'P(C)':<8} {'Top Token (logit)':<25}")
    print("-" * 90)
    
    for pos in range(start_pos, seq_len):
        pos_logits = logits[pos]
        probs = torch.softmax(pos_logits.float(), dim=-1)
        
        # ABC probabilities (max of with/without space variants)
        p_a = max(probs[A_id].item(), probs[A_sp].item())
        p_b = max(probs[B_id].item(), probs[B_sp].item())
        p_c = max(probs[C_id].item(), probs[C_sp].item())
        abc_mass = sum(probs[tid].item() for tid in all_answer_ids)
        
        # Top predicted token
        top_idx = pos_logits.argmax().item()
        top_logit = pos_logits[top_idx].item()
        top_token = model.to_single_str_token(top_idx)
        top_token_escaped = repr(top_token)[1:-1]  # escape special chars
        
        # Escape token string
        tok_str = token_strs[pos]
        tok_escaped = repr(tok_str)[1:-1][:13]  # truncate for display
        
        marker = "<<<" if abc_mass > 0.3 else ""
        print(f"{pos:<5} {tok_escaped:<15} {abc_mass:<10.4f} {p_a:<8.4f} {p_b:<8.4f} {p_c:<8.4f} {top_token_escaped} ({top_logit:.1f}) {marker}")
    
    print("-" * 90)


def run_logit_lens(model, tokens):
    """Run model and extract logit lens data for all layers.
    
    Returns:
        all_probs: [n_layers, seq_len, 3] - P(A), P(B), P(C) normalized over ABC only
        all_abc_mass: [n_layers, seq_len] - Total probability mass on A+B+C (before normalization)
        all_raw_diff: [n_layers, seq_len] - Raw logit(C) - logit(A) (before softmax)
        cache: Activation cache
    """
    # Run with cache
    _, cache = model.run_with_cache(tokens)
    
    # Get answer token IDs
    A_id, B_id, C_id = get_answer_token_ids(model)
    answer_ids = torch.tensor([A_id, B_id, C_id])
    
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[-1]
    
    # Storage
    all_probs = torch.zeros(n_layers, seq_len, 3)
    all_abc_mass = torch.zeros(n_layers, seq_len)
    all_raw_diff = torch.zeros(n_layers, seq_len)
    
    for layer in range(n_layers):
        # Get residual stream at this layer
        resid = cache["resid_post", layer]  # [1, seq_len, d_model]
        resid = resid[0]  # [seq_len, d_model] - remove batch dim
        
        # Apply final LayerNorm for more accurate early layer readings
        if hasattr(model, 'ln_final'):
            resid = model.ln_final(resid)
        
        # Apply unembedding: residual -> logits
        layer_logits = resid @ model.W_U  # [seq_len, vocab]
        if model.b_U is not None:
            layer_logits = layer_logits + model.b_U
        
        # Full vocab softmax to check ABC mass
        full_probs = torch.softmax(layer_logits.float(), dim=-1)
        abc_mass = full_probs[:, answer_ids].sum(dim=-1)  # [seq_len]
        all_abc_mass[layer] = abc_mass.cpu().detach()
        
        # Extract ABC logits
        abc_logits = layer_logits[:, answer_ids]  # [seq_len, 3]
        
        # Raw logit difference (before softmax) - avoids saturation
        raw_diff = abc_logits[:, 2] - abc_logits[:, 0]  # logit(C) - logit(A)
        all_raw_diff[layer] = raw_diff.cpu().detach()
        
        # Normalized probs over ABC only
        abc_probs = torch.softmax(abc_logits.float(), dim=-1)  # [seq_len, 3]
        all_probs[layer] = abc_probs.cpu().detach()
    
    return all_probs, all_abc_mass, all_raw_diff, cache


#%% Visualization Functions
def plot_heatmap(all_probs, token_strs, cfg, think_start_idx=None, save_path=None):
    """Plot heatmap: x=position, y=layer, color=P(C)-P(A)."""
    # P(C) - P(A) for each layer and position
    diff = all_probs[:, :, 2] - all_probs[:, :, 0]  # [n_layers, seq_len]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    vmin = cfg.HEATMAP_VMIN if cfg.HEATMAP_VMIN is not None else -1
    vmax = cfg.HEATMAP_VMAX if cfg.HEATMAP_VMAX is not None else 1
    
    im = ax.imshow(
        diff.numpy(),
        aspect='auto',
        cmap='RdBu',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
    )
    
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title("Logit Lens: P(C) - P(A) by Layer and Position\n(Blue = C winning, Red = A winning)")
    
    # Mark where forced CoT starts
    if think_start_idx is not None:
        ax.axvline(x=think_start_idx, color='yellow', linestyle='--', 
                   linewidth=2, label=f'<think> at pos {think_start_idx}')
        ax.legend(loc='upper left')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(C) - P(A)")
    
    # Add token labels on x-axis (sample every N tokens to avoid crowding)
    n_tokens = len(token_strs)
    if n_tokens > 50:
        step = max(1, n_tokens // 30)
        tick_positions = list(range(0, n_tokens, step))
        tick_labels = [token_strs[i][:8] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_final_layer(all_probs, token_strs, think_start_idx=None, save_path=None):
    """Plot P(A), P(B), P(C) trajectory at final layer."""
    final_layer_probs = all_probs[-1].numpy()  # [seq_len, 3]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    positions = np.arange(len(token_strs))
    
    ax.plot(positions, final_layer_probs[:, 0], label='P(A)', color='red', alpha=0.8)
    ax.plot(positions, final_layer_probs[:, 1], label='P(B)', color='green', alpha=0.8)
    ax.plot(positions, final_layer_probs[:, 2], label='P(C)', color='blue', alpha=0.8)
    
    # Mark where forced CoT starts
    if think_start_idx is not None:
        ax.axvline(x=think_start_idx, color='gray', linestyle='--', 
                   linewidth=1.5, label=f'<think> start')
    
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Probability")
    ax.set_title("Final Layer: Answer Probabilities Across Token Positions")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_by_layer(all_probs, save_path=None):
    """Plot P(C)-P(A) at last position for each layer."""
    # Get probs at last position for each layer
    last_pos_probs = all_probs[:, -1, :]  # [n_layers, 3]
    diff = (last_pos_probs[:, 2] - last_pos_probs[:, 0]).numpy()  # P(C) - P(A)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = np.arange(len(diff))
    colors = ['red' if d < 0 else 'blue' for d in diff]
    
    ax.bar(layers, diff, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel("Layer")
    ax.set_ylabel("P(C) - P(A)")
    ax.set_title("P(C) - P(A) at Last Token Position by Layer\n(Blue = C winning, Red = A winning)")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


#%% Main Run Function
def run(cfg):
    """Run logit lens analysis with current config."""
    print("\n" + "=" * 60)
    print("LOGIT LENS ANALYSIS")
    print("=" * 60)
    
    # Load BBQ question
    items = load_bbq_items(
        categories=[cfg.BBQ_CATEGORY],
        n_per_category=max(10, cfg.BBQ_INDEX + 1),
    )
    
    if cfg.BBQ_INDEX >= len(items):
        print(f"Warning: BBQ_INDEX {cfg.BBQ_INDEX} out of range, using 0")
        bbq_item = items[0]
    else:
        bbq_item = items[cfg.BBQ_INDEX]
    
    # Display question info
    print(f"\n=== BBQ Question ({cfg.BBQ_CATEGORY} #{cfg.BBQ_INDEX}) ===")
    print(f"Context: {bbq_item.context}")
    print(f"Question: {bbq_item.question}")
    print(f"Options: A. {bbq_item.choices[0]} | B. {bbq_item.choices[1]} | C. {bbq_item.choices[2]}")
    print(f"Correct Answer: {bbq_item.correct_letter}")
    
    # Build prompt
    use_few_shot = getattr(cfg, 'USE_FEW_SHOT', True)
    prompt, bbq_prompt, forced_cot = build_prompt(
        bbq_item,
        cfg.FORCED_COT_TEXT,
        cfg.FORCED_COT_REPEATS,
        use_few_shot=use_few_shot,
    )
    
    print(f"\n=== Forced CoT ===")
    print(f"Injected text: {repr(cfg.FORCED_COT_TEXT)} x {cfg.FORCED_COT_REPEATS}")
    print(f"Few-shot: {use_few_shot}")
    
    # Tokenize
    tokens = model.to_tokens(prompt)
    token_strs = model.to_str_tokens(prompt)
    
    print(f"\n=== Tokenization ===")
    print(f"Total tokens: {len(token_strs)}")
    
    # Verify tokenization looks right (first 20 tokens)
    print(f"First 20 tokens: {token_strs[:20]}")
    
    # Find <think> position
    think_start_idx = find_think_start_idx(token_strs)
    if think_start_idx:
        print(f"<think> found at position: {think_start_idx}")
    
    # Run logit lens
    print(f"\n=== Running Logit Lens ===")
    with torch.no_grad():
        all_probs, all_abc_mass, all_raw_diff, cache = run_logit_lens(model, tokens)
    
    # Print diagnostics
    print(f"\n=== Diagnostics ===")
    A_id, B_id, C_id = get_answer_token_ids(model)
    print(f"Token IDs: A={A_id}, B={B_id}, C={C_id}")
    
    # ABC mass diagnostic (is model confident in A/B/C?)
    final_abc_mass = all_abc_mass[-1, -1]
    print(f"\nP(A)+P(B)+P(C) at final position: {final_abc_mass:.4f}")
    if final_abc_mass < 0.5:
        print("  ⚠️  WARNING: Model may be planning different output!")
    
    # Raw logit diff (avoids softmax saturation)
    final_raw_diff = all_raw_diff[-1, -1]
    print(f"Raw logit diff [logit(C) - logit(A)] at final pos: {final_raw_diff:.2f}")
    
    # Final position probabilities (last layer)
    final_probs = all_probs[-1, -1]  # [3]
    print(f"\nFinal position (layer {model.cfg.n_layers-1}):")
    print(f"  P(A) = {final_probs[0]:.4f}")
    print(f"  P(B) = {final_probs[1]:.4f}")
    print(f"  P(C) = {final_probs[2]:.4f}")
    print(f"  Winner: {['A', 'B', 'C'][final_probs.argmax()]}")
    
    # Show some token context
    print(f"\nToken context (last 10 tokens):")
    for i in range(max(0, len(token_strs)-10), len(token_strs)):
        print(f"  [{i}] {repr(token_strs[i])}")
    
    # Find decision point
    find_decision_point(cache, model, tokens)
    
    # Generation verification
    print(f"\n=== Generation Verification ===")
    with torch.no_grad():
        generated = model.generate(tokens, max_new_tokens=5, temperature=0)
        generated_str = model.to_string(generated[0])
        print(f"Actual generation (last 50 chars): ...{generated_str[-50:]}")
    
    # Generate visualizations
    print(f"\n=== Generating Visualizations ===")
    
    fig1 = plot_heatmap(
        all_probs, token_strs, cfg,
        think_start_idx=think_start_idx,
        save_path=os.path.join(_outputs_dir, "logit_lens_heatmap.png") if cfg.SAVE_PLOTS else None
    )
    
    fig2 = plot_final_layer(
        all_probs, token_strs,
        think_start_idx=think_start_idx,
        save_path=os.path.join(_outputs_dir, "logit_lens_final_layer.png") if cfg.SAVE_PLOTS else None
    )
    
    fig3 = plot_by_layer(
        all_probs,
        save_path=os.path.join(_outputs_dir, "logit_lens_by_layer.png") if cfg.SAVE_PLOTS else None
    )
    
    if cfg.SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return all_probs, all_abc_mass, all_raw_diff, cache, bbq_item


#%% Interactive Loop
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LOGIT LENS INTERACTIVE ANALYSIS")
    print("=" * 60)
    print("Edit config.py to change BBQ question or injection settings.")
    print("Press Enter to re-run, Ctrl+C to exit.\n")
    
    while True:
        try:
            # Hot-reload config
            importlib.reload(cfg)
            print(f"\n[Config reloaded from config.py]")
            
            # Run analysis
            run(cfg)
            
            # Wait for user
            input("\n>>> Press Enter to re-run with updated config (Ctrl+C to exit)...")
            print("\n" + "=" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
            print("\nFix the error in config.py and press Enter to retry.")
            input("\n>>> Press Enter to retry...")
