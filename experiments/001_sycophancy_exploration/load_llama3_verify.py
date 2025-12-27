#!/usr/bin/env python3
# %% [markdown]
# Load Llama-3-8B-Instruct via TransformerLens and verify interpretability access.
#
# This script tests:
# 1. Model loading with TransformerLens
# 2. Text generation
# 3. Cache access for interpretability work
#
# **AUTHENTICATION REQUIRED:**
# This model is gated on HuggingFace. Before running this script, authenticate:
#     huggingface-cli login
#
# You'll need a HuggingFace account with access to Meta-Llama-3-8B-Instruct.

# %%
import torch
import torch.nn.functional as F

import config 
from transformer_lens import HookedTransformer

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
# Test generation
print("\n" + "="*80)
print("Testing generation...")
print("="*80)
test_prompt = "The capital of France is"
generated_text = model.generate(
    test_prompt,
    max_new_tokens=11,
    temperature=0.7,
    verbose=False
)
print(f"Prompt: {test_prompt}")
print(f"Generated: {generated_text}")
# %%
# Check for suspicious output
if "nan" in generated_text.lower() or "error" in generated_text.lower():
    print("⚠️  WARNING: Suspicious generation output detected!")
else:
    print("✅ Generation looks reasonable")

print("\n" + "="*80)
print("Testing cache access...")
print("="*80)
# %%
# Test cache access
# Tokenize the prompt
tokens = model.to_tokens(test_prompt)
print(f"Tokenized prompt shape: {tokens.shape}")

# Run with cache
logits, cache = model.run_with_cache(tokens)
print(f"Logits shape: {logits.shape}")

print("tokens: ", tokens)
print("logits: ", logits)
print("cache: ", cache)

# %%
#Explore cache keys a bit 

cache_keys = list(cache.keys())
cache_keys_splot = [key.split(".") for key in cache_keys]
print("cache_keys_splot: ", cache_keys_splot)

#number of keys
print("number of keys: ", len(cache_keys))
#First words
first_words = [key.split(".")[0] for key in cache_keys]
print("first_words: ", set(first_words))

#Second words
second_words = [key.split(".")[1] for key in cache_keys if len(key.split(".")) > 1]
print("second_words: ", set(second_words))

#Third words
third_words = [key.split(".")[2] for key in cache_keys if len(key.split(".")) > 2]
print("third_words: ", set(third_words))

#Fourth words
fourth_words = [key.split(".")[3] for key in cache_keys if len(key.split(".")) > 3]
print("fourth_words: ", set(fourth_words))



# %%
# Get cache keys
cache_keys = list(cache.keys())
print(f"\nTotal cache keys: {len(cache_keys)}")

# Filter for resid_post keys
resid_post_keys = [key for key in cache_keys if "resid_post" in key]
print(f"resid_post keys found: {len(resid_post_keys)}")

print("\nresid_post keys:")
for key in sorted(resid_post_keys):
    print(f"  - {key}")

# %%
# Get final layer residual stream at final token position
n_layers = model.cfg.n_layers
final_layer_idx = n_layers - 1

# Try different possible key formats
final_resid_key = "blocks.0.hook_resid_post"
if final_resid_key in cache:
    print('final_resid_key: ', final_resid_key)
else:
    print(f"⚠️  WARNING: {final_resid_key} not found in cache!")
 # %%

if final_resid_key:
    final_resid = cache[final_resid_key]
    print(f"\n✅ Found final layer residual stream: {final_resid_key}")
    print(f"   Shape: {final_resid.shape}")
    
    # Extract final token position (last position in sequence)
    final_token_resid = final_resid[0, -1, :]  # [batch=0, final_pos, hidden_dim]
    print(f"   Final token residual shape: {final_token_resid.shape}")
    print(f"   Final token residual dtype: {final_token_resid.dtype}")
    # print("final_token_resid: ", final_token_resid)
    # print("to string", model.to_string(final_token_resid))
    # Check for NaN
    if torch.isnan(final_token_resid).any():
        print("   ⚠️  WARNING: NaN values detected in residual stream!")
    else:
        print("   ✅ No NaN values detected")
        
    # Check expected shape (should be (1, seq_len, 4096) for Llama-3-8B)
    expected_hidden_dim = 4096
    if final_resid.shape[-1] == expected_hidden_dim:
        print(f"   ✅ Hidden dimension matches expected: {expected_hidden_dim}")
    else:
        print(f"   ⚠️  Hidden dimension is {final_resid.shape[-1]}, expected {expected_hidden_dim}")
        
else:
    print(f"\n⚠️  WARNING: Could not find final layer residual stream!")
    print(f"   Looked for layer {final_layer_idx} (model has {n_layers} layers)")
    print(f"   Available resid_post keys: {resid_post_keys}")

# %%
# LOGIT LENS VISUALIZATION
print("\n" + "="*80)
print("LOGIT LENS ANALYSIS")
print("="*80)
print(f"Prompt: '{test_prompt}'")
print(f"Analyzing how the model's prediction evolves across {n_layers} layers...")
print()

# Collect logit lens data for each layer
logit_lens_data = []

for layer_idx in range(n_layers):
    # Get residual stream at this layer's output (after attention + MLP)
    resid_key = f"blocks.{layer_idx}.hook_resid_post"
    
    if resid_key not in cache:
        print(f"⚠️  Layer {layer_idx}: Key '{resid_key}' not found, skipping")
        continue
    
    # Get residual at final token position: shape [hidden_dim]
    resid = cache[resid_key][0, -1, :]  # batch=0, final token, all hidden dims
    
    # Apply layer norm (model.ln_final) then unembedding (model.unembed)
    # This mimics what the model does at the final layer
    resid_normed = model.ln_final(resid)
    logits = model.unembed(resid_normed.unsqueeze(0).unsqueeze(0))  # [1, 1, vocab_size]
    logits = logits.squeeze()  # [vocab_size]
    
    # Get probabilities
    probs = F.softmax(logits.float(), dim=-1)
    
    # Get top-5 tokens
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    # Convert to strings
    top5_tokens = []
    for idx in top5_indices:
        token_str = model.to_string(idx.unsqueeze(0))
        # Clean up the string for display
        token_str = repr(token_str)[1:-1]  # Show escape chars clearly
        top5_tokens.append(token_str)
    
    logit_lens_data.append({
        'layer': layer_idx,
        'top5_tokens': top5_tokens,
        'top5_probs': top5_probs.tolist(),
        'top1_prob': top5_probs[0].item(),
    })

# Print table
print("Layer │ Top-1 Prob │ Top-5 Predicted Tokens")
print("──────┼────────────┼" + "─" * 60)

for data in logit_lens_data:
    layer = data['layer']
    top1_prob = data['top1_prob']
    tokens_with_probs = []
    for tok, prob in zip(data['top5_tokens'], data['top5_probs']):
        # Truncate long tokens for display
        tok_display = tok[:15] + "..." if len(tok) > 18 else tok
        tokens_with_probs.append(f"{tok_display}")
    
    tokens_str = " | ".join(tokens_with_probs)
    print(f"{layer:5d} │ {top1_prob:10.4f} │ {tokens_str}")

# %%
# ASCII Visualization: Show how "Paris" (or similar) emerges
print("\n" + "="*80)
print("LOGIT LENS VISUALIZATION: Token Probability Evolution")
print("="*80)

# Get the top prediction from the final layer to track
final_resid = cache[f"blocks.{n_layers-1}.hook_resid_post"][0, -1, :]
final_normed = model.ln_final(final_resid)
final_logits = model.unembed(final_normed.unsqueeze(0).unsqueeze(0)).squeeze()
target_token_id = torch.argmax(final_logits).item()
target_token_str = model.to_string(torch.tensor([target_token_id]))
print(f"Tracking final layer's top prediction: '{target_token_str}' (token_id={target_token_id})")

# Get probability of target token at each layer
target_probs = []
for layer_idx in range(n_layers):
    resid_key = f"blocks.{layer_idx}.hook_resid_post"
    if resid_key not in cache:
        target_probs.append(0.0)
        continue
    
    resid = cache[resid_key][0, -1, :]
    resid_normed = model.ln_final(resid)
    logits = model.unembed(resid_normed.unsqueeze(0).unsqueeze(0)).squeeze()
    probs = F.softmax(logits.float(), dim=-1)
    target_probs.append(probs[target_token_id].item())

# ASCII bar chart with dynamic scaling
print(f"\nProbability of '{target_token_str.strip()}' at each layer:")
print()
max_bar_width = 50
max_prob_in_data = max(target_probs) if target_probs else 1.0
# Use log scale indicator for better visibility at low probs
use_linear_scale = max_prob_in_data > 0.1

if use_linear_scale:
    print(f"(Linear scale, max={max_prob_in_data:.3f})")
else:
    print(f"(Values shown as-is, max={max_prob_in_data:.4f})")
print()

for layer_idx, prob in enumerate(target_probs):
    # Scale bar width based on max probability in data
    if max_prob_in_data > 0:
        bar_width = int((prob / max_prob_in_data) * max_bar_width)
    else:
        bar_width = 0
    bar = "█" * bar_width + "░" * (max_bar_width - bar_width)
    
    # Highlight key layers
    if layer_idx < 5:
        layer_label = f"L{layer_idx:2d} (early)"
    elif layer_idx >= n_layers - 5:
        layer_label = f"L{layer_idx:2d} (late) "
    else:
        layer_label = f"L{layer_idx:2d}       "
    
    print(f"{layer_label} │{bar}│ {prob:.4f}")

# Summary analysis
print("\n" + "-"*80)
early_avg = sum(target_probs[:5]) / 5 if len(target_probs) >= 5 else 0
late_avg = sum(target_probs[-5:]) / 5 if len(target_probs) >= 5 else 0
max_prob = max(target_probs)
max_layer = target_probs.index(max_prob)

print(f"Token tracked: '{target_token_str}' (top prediction at final layer)")
print(f"Early layers (0-4) average prob: {early_avg:.4f}")
print(f"Late layers ({n_layers-5}-{n_layers-1}) average prob: {late_avg:.4f}")
print(f"Max probability: {max_prob:.4f} at layer {max_layer}")

# Analysis of the pattern
if max_prob > 0.5:
    print("✅ HEALTHY: Strong prediction emerges (>50% at peak)")
    if max_layer > n_layers // 2:
        print("✅ Peak occurs in later half of network (expected)")
    else:
        print("⚠️  Peak occurs early - unusual pattern")
elif max_prob > 0.1:
    print("⚠️  MODERATE: Prediction reaches moderate confidence")
else:
    print("⚠️  WEAK: Token never gets high probability")

if late_avg > early_avg * 2:
    print("✅ Late layers have higher prob than early (expected)")
elif early_avg > late_avg * 2 and early_avg > 0.01:
    print("⚠️  Early layers have higher prob than late (unusual)")

# Show what the top prediction is at the final layer
final_data = logit_lens_data[-1] if logit_lens_data else None
if final_data:
    print(f"\nFinal layer top prediction: '{final_data['top5_tokens'][0]}' with prob {final_data['top1_prob']:.4f}")
    if "Paris" in final_data['top5_tokens'][0] or "paris" in final_data['top5_tokens'][0].lower():
        print("✅ 'Paris' is the top prediction - logit lens working correctly!")
    elif any("Paris" in tok or "paris" in tok.lower() for tok in final_data['top5_tokens']):
        print("⚠️  'Paris' is in top-5 but not top-1")
    else:
        print(f"ℹ️  Top prediction is '{final_data['top5_tokens'][0]}' (for this prompt)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"1. Generated text: {generated_text}")
print(f"2. Cache keys available: {len(cache_keys)} total, {len(resid_post_keys)} resid_post keys")
if final_resid_key:
    print(f"3. Final layer residual stream shape: {cache[final_resid_key].shape}")
    print(f"   Final token position shape: {cache[final_resid_key][0, -1, :].shape}")
else:
    print(f"3. Final layer residual stream: NOT FOUND")
print(f"4. Logit lens: Analyzed {len(logit_lens_data)} layers")

print("\n" + "="*80)
print("Done!")
print("="*80)


# %%
