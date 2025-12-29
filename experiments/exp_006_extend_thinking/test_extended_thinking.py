#!/usr/bin/env python3
"""test_extended_thinking.py

Extended Thinking Intervention Test

Tests whether intercepting </think> and forcing continued reasoning produces
coherent output or garbage in DeepSeek-R1-Distill-Qwen-1.5B.

Two conditions:
- NORMAL: Generate until natural completion
- EXTENDED: Intercept first </think>, inject continuation, allow second </think>

Usage:
    python test_extended_thinking.py
"""

import os

# Set HF cache before imports
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

import torch
import torch.nn.functional as F
from nnsight import LanguageModel

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Known token IDs (discovered via diagnostics)
# <think> = 151648, </think> = 151649
THINK_END_TOKEN = "</think>"
THINK_END_ID = 151649  # Fallback if tokenizer lookup fails

# Sampling parameters (DeepSeek recommended)
TEMPERATURE = 0.6
TOP_P = 0.95

# Set to True to run on CPU (slow but works on any hardware)
USE_CPU = False

# Token limits
MAX_TOKENS_NORMAL = 800
MAX_TOKENS_EXTENDED = 1200

# Continuation text to inject when intercepting </think>
CONTINUATION_TEXT = "\n\nWait, let me reconsider this step by step..."

# Test prompt
TEST_PROMPT = """<|User|>Is the population of France greater than the population of Germany? Think step by step, then answer Yes or No.
<|Assistant|><think>
"""


# =============================================================================
# Tokenizer Diagnostics
# =============================================================================

def run_diagnostics(tokenizer):
    """Investigate tokenizer to find the actual think tokens."""
    print("=" * 70)
    print("TOKENIZER DIAGNOSTICS")
    print("=" * 70)
    
    print("\n[Special tokens map]")
    print(tokenizer.special_tokens_map)
    
    print("\n[Added tokens]")
    print(tokenizer.added_tokens_encoder)
    
    print("\n[All special tokens]")
    print(tokenizer.all_special_tokens)
    
    print("\n[Tokens containing 'think']")
    think_tokens = {}
    for token, idx in tokenizer.get_vocab().items():
        if "think" in token.lower():
            print(f"  {idx}: {repr(token)}")
            think_tokens[token] = idx
    
    print(f"\n[EOS token id]: {tokenizer.eos_token_id}")
    print(f"[EOS token]: {repr(tokenizer.eos_token)}")
    
    # Try to identify the think-end token from added_tokens_encoder
    think_end_id = None
    think_end_token = None
    
    # Check added_tokens_encoder first (most reliable)
    if THINK_END_TOKEN in tokenizer.added_tokens_encoder:
        think_end_id = tokenizer.added_tokens_encoder[THINK_END_TOKEN]
        think_end_token = THINK_END_TOKEN
    # Search in discovered think_tokens
    elif think_tokens:
        for token, idx in think_tokens.items():
            if token == "</think>" or ("/" in token and "think" in token.lower()):
                think_end_id = idx
                think_end_token = token
                break
    
    # Fallback to known ID
    if think_end_id is None:
        print(f"[WARNING] Could not find think-end token dynamically, using fallback ID={THINK_END_ID}")
        think_end_id = THINK_END_ID
        think_end_token = THINK_END_TOKEN
    
    print(f"\n[Detected think-end token]: {repr(think_end_token)} (id={think_end_id})")
    print("=" * 70)
    
    return think_end_id, think_end_token


# =============================================================================
# Generation Functions
# =============================================================================

def sample_token(logits, temperature, top_p):
    """Sample next token using temperature and top-p (nucleus) sampling."""
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-p filtering
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    # Sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


def generate_with_intervention(model, tokenizer, prompt, max_tokens, 
                                think_end_id, extend_thinking=False):
    """
    Generate tokens one at a time with optional </think> intervention.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate
        think_end_id: Token ID for </think>
        extend_thinking: If True, intercept first </think> and inject continuation
    
    Returns:
        tuple: (full_output_string, token_count, think_end_positions)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_length = input_ids.shape[1]
    
    think_end_count = 0
    think_end_positions = []
    generated_tokens = 0
    
    while generated_tokens < max_tokens:
        # Forward pass
        with torch.no_grad():
            with model.trace(input_ids):
                logits = model.lm_head.output.save()
        
        # Get logits for last position
        last_logits = logits[0, -1].float()
        
        # Sample next token
        next_token_id = sample_token(last_logits, TEMPERATURE, TOP_P)
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break
        
        # Check for </think>
        if next_token_id == think_end_id:
            think_end_positions.append(generated_tokens)
            think_end_count += 1
            
            # Intervention: on first </think> in extended mode, inject continuation
            if extend_thinking and think_end_count == 1:
                print(f"  [INTERVENTION] Intercepted </think> at position {generated_tokens}")
                print(f"  [INTERVENTION] Injecting: {repr(CONTINUATION_TEXT[:50])}...")
                
                # Encode continuation and append (skip the </think> token)
                continuation_ids = tokenizer.encode(CONTINUATION_TEXT, 
                                                    add_special_tokens=False,
                                                    return_tensors="pt").to(model.device)
                input_ids = torch.cat([input_ids, continuation_ids], dim=1)
                generated_tokens += continuation_ids.shape[1]
                continue
        
        # Append token
        input_ids = torch.cat([input_ids, 
                               torch.tensor([[next_token_id]], device=model.device)], dim=1)
        generated_tokens += 1
    
    # Decode full output
    full_output = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    generated_output = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=False)
    
    return generated_output, generated_tokens, think_end_positions


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("EXTENDED THINKING INTERVENTION TEST")
    print("=" * 70)
    
    # Print GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {gpu_mem:.1f} GB")
    
    # Load model
    print(f"\n[Loading model: {MODEL_NAME}]")
    if USE_CPU:
        print("[Running on CPU - this will be slow]")
        model = LanguageModel(MODEL_NAME, dispatch=True, device_map="cpu", 
                              dtype=torch.float32)
    else:
        model = LanguageModel(MODEL_NAME, dispatch=True, device_map="auto", 
                              dtype=torch.float16)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.do_sample = False
    tokenizer = model.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded successfully.")
    
    # Run diagnostics
    think_end_id, think_end_token = run_diagnostics(tokenizer)
    
    if think_end_id is None:
        print("\n[ERROR] Could not find think-end token! Aborting.")
        return
    
    # Run NORMAL condition
    print("\n" + "=" * 70)
    print("NORMAL CONDITION")
    print(f"Max tokens: {MAX_TOKENS_NORMAL}, No intervention")
    print("=" * 70)
    
    output_normal, count_normal, positions_normal = generate_with_intervention(
        model, tokenizer, TEST_PROMPT, MAX_TOKENS_NORMAL, 
        think_end_id, extend_thinking=False
    )
    
    print(f"\n[Output]:\n{output_normal}")
    print(f"\n[Token count]: {count_normal}")
    print(f"[</think> positions]: {positions_normal}")
    
    # Run EXTENDED condition
    print("\n" + "=" * 70)
    print("EXTENDED CONDITION")
    print(f"Max tokens: {MAX_TOKENS_EXTENDED}, Intercept first </think>")
    print("=" * 70)
    
    output_extended, count_extended, positions_extended = generate_with_intervention(
        model, tokenizer, TEST_PROMPT, MAX_TOKENS_EXTENDED,
        think_end_id, extend_thinking=True
    )
    
    print(f"\n[Output]:\n{output_extended}")
    print(f"\n[Token count]: {count_extended}")
    print(f"[</think> positions]: {positions_extended}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Check if model re-emitted </think> immediately after injection
    immediate_reemit = False
    if len(positions_extended) >= 2:
        # Continuation text is roughly 10-15 tokens
        continuation_tokens = len(tokenizer.encode(CONTINUATION_TEXT, add_special_tokens=False))
        gap = positions_extended[1] - positions_extended[0]
        immediate_reemit = gap <= continuation_tokens + 5
    
    print(f"\nNormal token count: {count_normal}")
    print(f"Extended token count: {count_extended}")
    print(f"Additional tokens from extension: {count_extended - count_normal}")
    print(f"\nDid model re-emit </think> immediately after injection: {'YES' if immediate_reemit else 'NO'}")
    print(f"Number of </think> in extended output: {len(positions_extended)}")
    
    print("\n[Manual inspection needed]")
    print("  - Is the extended reasoning coherent?")
    print("  - Does the model continue reasoning logically after injection?")
    print("  - Is the final answer consistent between conditions?")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()


