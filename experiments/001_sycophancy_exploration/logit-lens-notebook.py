#!/usr/bin/env python3
"""
Interactive example demonstrating logit lens analysis.

This script loads a model, runs inference, and demonstrates
the logit lens utility functions.
"""

import torch
import config  # Must import before transformer_lens to set up cache directory
from transformer_lens import HookedTransformer
from logit_lens_utils import get_top5_by_layer, display_top5_table, display_probability_by_layer

# %%
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("\n" + "="*80)
print("Loading Llama-3-8B-Instruct with TransformerLens...")
print("="*80)

# Load model with specified parameters
model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device=device, 
    local_files_only=True
)
print("✅ Model loaded successfully!")

# %%
# Example prompt
example_prompt = "The capital of Australia is"
print(f"\nExample prompt: '{example_prompt}'")

# Tokenize and run with cache
tokens = model.to_tokens(example_prompt)
print(f"Tokenized prompt shape: {tokens.shape}")

# Run with cache
logits, cache = model.run_with_cache(tokens)
print(f"Logits shape: {logits.shape}")

n_layers = model.cfg.n_layers
print(f"Model has {n_layers} layers")

# %%
# Example 1: Get top-5 tokens by layer
print("\n" + "="*80)
print("EXAMPLE 1: Top-5 Tokens by Layer")
print("="*80)
print(f"Prompt: '{example_prompt}'")
print(f"Analyzing how the model's prediction evolves across {n_layers} layers...")
print()

logit_lens_data = get_top5_by_layer(model, cache, n_layers)
display_top5_table(logit_lens_data)

# %%
# Example 2: Display probability evolution for a specific token
print("\n" + "="*80)
print("EXAMPLE 2: Probability Evolution by Layer")
print("="*80)

# This will automatically track the final layer's top prediction
target_probs, target_token_id, target_token_str = display_probability_by_layer(
    model=model,
    cache=cache,
    n_layers=n_layers,
    prompt=example_prompt
)

# %%
# Example 3: Track a specific token (if you know its ID)
# For example, if you want to track a specific token:
# target_token_id = model.to_single_token(" Canberra")
# display_probability_by_layer(
#     model=model,
#     cache=cache,
#     n_layers=n_layers,
#     prompt=example_prompt,
#     target_token_id=target_token_id
# )

print("\n" + "="*80)
print("Done!")
print("="*80)

