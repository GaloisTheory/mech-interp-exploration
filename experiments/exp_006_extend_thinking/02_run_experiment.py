#!/usr/bin/env python3
"""02_run_experiment.py

Extended Thinking Intervention Experiment

Tests whether intercepting </think> and forcing continued reasoning produces
coherent output or garbage in DeepSeek-R1-Distill-Qwen-1.5B.

Run cells interactively with #%% markers in VS Code/Cursor.
"""

#%% Imports and cache setup
import os

# Set HF cache before any HuggingFace imports
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

import torch
import torch.nn.functional as F
from nnsight import LanguageModel

#%% Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Think token ID (discovered in 01_find_think_tokens.py)
THINK_END_ID = 151649  # </think>

# Sampling parameters (DeepSeek recommended)
TEMPERATURE = 0.6
TOP_P = 0.95

# Token limits
MAX_TOKENS_NORMAL = 800
MAX_TOKENS_EXTENDED = 1200

# Continuation text to inject when intercepting </think>
CONTINUATION_TEXT = "\n\nWait, let me reconsider this step by step..."

#%% Test prompt
TEST_PROMPT = """<|User|>Is the population of France greater than the population of Germany? Think step by step, then answer Yes or No.
<|Assistant|><think>
"""

print("Test prompt:")
print("-" * 40)
print(TEST_PROMPT)
print("-" * 40)

#%% Load model
print(f"Loading model: {MODEL_NAME}")

if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {gpu_mem:.1f} GB")
    model = LanguageModel(MODEL_NAME, dispatch=True, device_map="auto", dtype=torch.float16)
else:
    print("No GPU available, using CPU (will be slow)")
    model = LanguageModel(MODEL_NAME, dispatch=True, device_map="cpu", dtype=torch.float32)

# Disable default sampling config (we do our own)
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.do_sample = False

tokenizer = model.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id

print("Model loaded!")

#%% Verify think token ID
print("\nVerifying </think> token ID...")
think_end_id_from_tokenizer = tokenizer.added_tokens_encoder.get("</think>")
print(f"  From added_tokens_encoder: {think_end_id_from_tokenizer}")
print(f"  Using configured value: {THINK_END_ID}")

if think_end_id_from_tokenizer != THINK_END_ID:
    print("  ⚠️ WARNING: Mismatch! Updating to tokenizer value.")
    THINK_END_ID = think_end_id_from_tokenizer
else:
    print("  ✓ Match confirmed")

#%% Sampling function
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

print("Sampling function defined ✓")

#%% Generation function with intervention
def generate_with_intervention(model, tokenizer, prompt, max_tokens, 
                                think_end_id, extend_thinking=False,
                                continuation_text=CONTINUATION_TEXT,
                                verbose=True):
    """
    Generate tokens one at a time with optional </think> intervention.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        max_tokens: Maximum tokens to generate
        think_end_id: Token ID for </think>
        extend_thinking: If True, intercept first </think> and inject continuation
        continuation_text: Text to inject on intervention
        verbose: Print progress
    
    Returns:
        tuple: (generated_output_string, token_count, think_end_positions)
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
            if verbose:
                print(f"  [EOS reached at position {generated_tokens}]")
            break
        
        # Check for </think>
        if next_token_id == think_end_id:
            think_end_positions.append(generated_tokens)
            think_end_count += 1
            
            # Intervention: on first </think> in extended mode, inject continuation
            if extend_thinking and think_end_count == 1:
                if verbose:
                    print(f"  [INTERVENTION] Intercepted </think> at position {generated_tokens}")
                    print(f"  [INTERVENTION] Injecting: {repr(continuation_text[:50])}...")
                
                # Encode continuation and append (skip the </think> token)
                continuation_ids = tokenizer.encode(continuation_text, 
                                                    add_special_tokens=False,
                                                    return_tensors="pt").to(model.device)
                input_ids = torch.cat([input_ids, continuation_ids], dim=1)
                generated_tokens += continuation_ids.shape[1]
                continue
        
        # Append token
        input_ids = torch.cat([input_ids, 
                               torch.tensor([[next_token_id]], device=model.device)], dim=1)
        generated_tokens += 1
        
        # Progress indicator
        if verbose and generated_tokens % 100 == 0:
            print(f"  [Generated {generated_tokens} tokens...]")
    
    # Decode generated output (excluding prompt)
    generated_output = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=False)
    
    return generated_output, generated_tokens, think_end_positions

print("Generation function defined ✓")

#%% Run NORMAL condition
print("=" * 70)
print("NORMAL CONDITION")
print(f"Max tokens: {MAX_TOKENS_NORMAL}, No intervention")
print("=" * 70)

output_normal, count_normal, positions_normal = generate_with_intervention(
    model, tokenizer, TEST_PROMPT, MAX_TOKENS_NORMAL, 
    THINK_END_ID, extend_thinking=False
)

print(f"\n[Token count]: {count_normal}")
print(f"[</think> positions]: {positions_normal}")

#%% Display NORMAL output
print("=" * 70)
print("NORMAL OUTPUT")
print("=" * 70)
print(output_normal)

#%% Run EXTENDED condition
print("=" * 70)
print("EXTENDED CONDITION")
print(f"Max tokens: {MAX_TOKENS_EXTENDED}, Intercept first </think>")
print("=" * 70)

output_extended, count_extended, positions_extended = generate_with_intervention(
    model, tokenizer, TEST_PROMPT, MAX_TOKENS_EXTENDED,
    THINK_END_ID, extend_thinking=True
)

print(f"\n[Token count]: {count_extended}")
print(f"[</think> positions]: {positions_extended}")

#%% Display EXTENDED output
print("=" * 70)
print("EXTENDED OUTPUT")
print("=" * 70)
print(output_extended)

#%% Analysis
print("=" * 70)
print("ANALYSIS")
print("=" * 70)

# Check if model re-emitted </think> immediately after injection
immediate_reemit = False
if len(positions_extended) >= 2:
    # Continuation text is roughly 10-15 tokens
    continuation_tokens = len(tokenizer.encode(CONTINUATION_TEXT, add_special_tokens=False))
    gap = positions_extended[1] - positions_extended[0]
    immediate_reemit = gap <= continuation_tokens + 5
    print(f"\nGap between 1st and 2nd </think>: {gap} tokens")
    print(f"Continuation was: {continuation_tokens} tokens")

print(f"\nNormal token count: {count_normal}")
print(f"Extended token count: {count_extended}")
print(f"Additional tokens from extension: {count_extended - count_normal}")
print(f"\nDid model re-emit </think> immediately after injection: {'YES' if immediate_reemit else 'NO'}")
print(f"Number of </think> in extended output: {len(positions_extended)}")

#%% Manual inspection checklist
print("\n" + "=" * 70)
print("MANUAL INSPECTION CHECKLIST")
print("=" * 70)
print("""
Questions to consider:

1. Is the extended reasoning coherent?
   - Does it read like natural language?
   - Is there logical flow?

2. Does the model continue reasoning logically after injection?
   - Did it pick up from where it was thinking?
   - Did it genuinely reconsider or just repeat?

3. Is the final answer consistent between conditions?
   - Normal answer: ___
   - Extended answer: ___
   - Did extension change the answer?

4. Did extension lead to self-correction?
   - Were any errors caught?
   - Did the model revise any claims?
""")

#%% Try different continuation prompts
# Uncomment and modify to test different interventions

# ALTERNATIVE_CONTINUATIONS = [
#     "\n\nWait, let me reconsider this step by step...",
#     "\n\nActually, I should double-check my reasoning...",
#     "\n\nHmm, I'm not sure about that. Let me think again...",
#     "\n\nBut wait, is that correct?",
# ]
# 
# for cont in ALTERNATIVE_CONTINUATIONS:
#     print(f"\n{'='*70}")
#     print(f"Testing: {repr(cont[:40])}...")
#     print("="*70)
#     
#     output, count, positions = generate_with_intervention(
#         model, tokenizer, TEST_PROMPT, MAX_TOKENS_EXTENDED,
#         THINK_END_ID, extend_thinking=True,
#         continuation_text=cont
#     )
#     print(f"Tokens: {count}, </think> at: {positions}")
#     print(output[:500] + "...")

#%% Try different prompts
# Uncomment and modify to test on different questions

# OTHER_PROMPTS = [
#     """<|User|>What is 17 * 23? Think step by step.
# <|Assistant|><think>
# """,
#     """<|User|>Is a tomato a fruit or a vegetable? Explain your reasoning.
# <|Assistant|><think>
# """,
# ]
# 
# for prompt in OTHER_PROMPTS:
#     print(f"\n{'='*70}")
#     print(f"Prompt: {prompt[:50]}...")
#     print("="*70)
#     
#     # Normal
#     out_n, cnt_n, pos_n = generate_with_intervention(
#         model, tokenizer, prompt, 400, THINK_END_ID, extend_thinking=False, verbose=False
#     )
#     print(f"Normal: {cnt_n} tokens, </think> at {pos_n}")
#     
#     # Extended
#     out_e, cnt_e, pos_e = generate_with_intervention(
#         model, tokenizer, prompt, 600, THINK_END_ID, extend_thinking=True, verbose=False
#     )
#     print(f"Extended: {cnt_e} tokens, </think> at {pos_e}")

