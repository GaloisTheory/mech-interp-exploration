#!/usr/bin/env python3
"""01_find_think_tokens.py

Tokenizer diagnostics for finding <think> and </think> tokens in DeepSeek-R1-Distill models.

Run cells interactively with #%% markers in VS Code/Cursor.
"""

#%% Imports and cache setup
import os

# Set HF cache before any HuggingFace imports
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

import torch
from transformers import AutoTokenizer

#%% Configuration
MODEL_NAME = "Qwen/QwQ-32B"
#"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Known token IDs (for reference/verification)
KNOWN_THINK_START_ID = 151648  # <think>
KNOWN_THINK_END_ID = 151649    # </think>

#%% Load tokenizer only (faster than loading full model)
print(f"Loading tokenizer for: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded!")

#%% Check special tokens map
print("=" * 70)
print("SPECIAL TOKENS MAP")
print("=" * 70)
for key, value in tokenizer.special_tokens_map.items():
    print(f"  {key}: {repr(value)}")

#%% Check all special tokens
print("\n" + "=" * 70)
print("ALL SPECIAL TOKENS")
print("=" * 70)
for i, token in enumerate(tokenizer.all_special_tokens):
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(f"  {i}: {repr(token)} (id={token_id})")

#%% Check added tokens encoder (KEY - this is where think tokens live!)
print("\n" + "=" * 70)
print("ADDED TOKENS ENCODER")
print("=" * 70)
print("This is where special tokens like <think>/<think> are stored!\n")

for token, idx in sorted(tokenizer.added_tokens_encoder.items(), key=lambda x: x[1]):
    print(f"  {idx}: {repr(token)}")

#%% Search for all tokens containing "think"
print("\n" + "=" * 70)
print("TOKENS CONTAINING 'think'")
print("=" * 70)

vocab = tokenizer.get_vocab()
think_tokens = {}

for token, idx in vocab.items():
    if "think" in token.lower():
        think_tokens[token] = idx
        print(f"  {idx}: {repr(token)}")

if not think_tokens:
    print("  (none found in regular vocab)")

#%% Find think tokens programmatically
print("\n" + "=" * 70)
print("THINK TOKEN DETECTION")
print("=" * 70)

think_start_token = "<think>"
think_end_token = "</think>"

# Method 1: Check added_tokens_encoder (CORRECT way)
think_start_id = tokenizer.added_tokens_encoder.get(think_start_token)
think_end_id = tokenizer.added_tokens_encoder.get(think_end_token)

print(f"\nFrom added_tokens_encoder:")
print(f"  {think_start_token}: {think_start_id}")
print(f"  {think_end_token}: {think_end_id}")

# Verify against known values
if think_start_id == KNOWN_THINK_START_ID:
    print(f"\n✓ <think> ID matches known value: {KNOWN_THINK_START_ID}")
else:
    print(f"\n✗ <think> ID mismatch! Got {think_start_id}, expected {KNOWN_THINK_START_ID}")

if think_end_id == KNOWN_THINK_END_ID:
    print(f"✓ </think> ID matches known value: {KNOWN_THINK_END_ID}")
else:
    print(f"✗ </think> ID mismatch! Got {think_end_id}, expected {KNOWN_THINK_END_ID}")

#%% WARNING: tokenizer.encode() doesn't work reliably for special tokens!
print("\n" + "=" * 70)
print("WARNING: encode() vs added_tokens_encoder")
print("=" * 70)

# This may NOT give correct results for special tokens
encoded_think_end = tokenizer.encode(think_end_token, add_special_tokens=False)
print(f"\ntokenizer.encode('{think_end_token}'): {encoded_think_end}")

# Compare with added_tokens_encoder
print(f"added_tokens_encoder['{think_end_token}']: {think_end_id}")

if len(encoded_think_end) > 1:
    print("\n⚠️  encode() tokenized as multiple pieces - DON'T USE THIS!")
    print("   Always use added_tokens_encoder for special tokens.")
elif encoded_think_end[0] != think_end_id:
    print("\n⚠️  encode() gave different ID - DON'T USE THIS!")
    print("   Always use added_tokens_encoder for special tokens.")
else:
    print("\n✓ encode() happened to give correct result, but added_tokens_encoder is more reliable.")

#%% Test decoding the think tokens
print("\n" + "=" * 70)
print("DECODE TEST")
print("=" * 70)

# Decode individual tokens
decoded_start = tokenizer.decode([think_start_id])
decoded_end = tokenizer.decode([think_end_id])

print(f"\nDecode ID {think_start_id}: {repr(decoded_start)}")
print(f"Decode ID {think_end_id}: {repr(decoded_end)}")

# Decode a sequence with think tags
test_sequence = [think_start_id, 40, 1079, 358, 1744, 911, 419, 13, think_end_id]
decoded_seq = tokenizer.decode(test_sequence)
print(f"\nTest sequence: {test_sequence}")
print(f"Decoded: {repr(decoded_seq)}")

#%% EOS token info
print("\n" + "=" * 70)
print("EOS TOKEN INFO")
print("=" * 70)

print(f"\nEOS token: {repr(tokenizer.eos_token)}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

# Check if there are multiple EOS-like tokens
eos_candidates = []
for token, idx in tokenizer.added_tokens_encoder.items():
    if "end" in token.lower() or "eos" in token.lower() or token.startswith("<|"):
        eos_candidates.append((token, idx))

if eos_candidates:
    print("\nPotential EOS-like tokens:")
    for token, idx in eos_candidates:
        print(f"  {idx}: {repr(token)}")

#%% Summary
print("\n" + "=" * 70)
print("SUMMARY - Copy these values for the experiment")
print("=" * 70)

print(f"""
# Token IDs for {MODEL_NAME}
THINK_START_ID = {think_start_id}  # {think_start_token}
THINK_END_ID = {think_end_id}      # {think_end_token}
EOS_TOKEN_ID = {tokenizer.eos_token_id}  # {repr(tokenizer.eos_token)}

# Access method (use in experiment):
# think_end_id = tokenizer.added_tokens_encoder["</think>"]
""")

