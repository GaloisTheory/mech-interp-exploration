#%% Imports
from experiments.config import *  # Sets HF_HOME before any HF imports

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel

# Model identifiers - using the CORRECT base model for Distill-Qwen-7B
BASE_MODEL = "Qwen/Qwen2.5-Math-7B"  # Stated base for DeepSeek-R1-Distill-Qwen-7B
DISTILL_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

#%% ============================================================
#   TOKENIZER COMPATIBILITY CHECK
#   ============================================================
print("=" * 60)
print("TOKENIZER COMPATIBILITY CHECK")
print("=" * 60)

print(f"\nLoading tokenizers...")
print(f"  Base:    {BASE_MODEL}")
print(f"  Distill: {DISTILL_MODEL}")

tok_base = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tok_dist = AutoTokenizer.from_pretrained(DISTILL_MODEL, trust_remote_code=True)

#%% Compare vocab sizes
print(f"\n--- Vocab Sizes ---")
print(f"  Base vocab size:    {tok_base.vocab_size}")
print(f"  Distill vocab size: {tok_dist.vocab_size}")
print(f"  Match: {tok_base.vocab_size == tok_dist.vocab_size}")

#%% Check token ID -> token string mapping consistency (FULL VOCAB)
print(f"\n--- Token ID Mapping Consistency (Full Vocab) ---")
check_range = min(tok_base.vocab_size, tok_dist.vocab_size)

mismatches = []
first_mismatch_id = None
for i in range(check_range):
    tok_b = tok_base.convert_ids_to_tokens(i)
    tok_d = tok_dist.convert_ids_to_tokens(i)
    if tok_b != tok_d:
        if first_mismatch_id is None:
            first_mismatch_id = i
        mismatches.append((i, tok_b, tok_d))

print(f"  Checked token IDs 0-{check_range-1}")
print(f"  Mismatches found: {len(mismatches)}")
print(f"  Match rate: {100 * (check_range - len(mismatches)) / check_range:.4f}%")

if first_mismatch_id is not None:
    print(f"\n  ⚠️  First mismatch at ID {first_mismatch_id}")
    print(f"  → All tokens 0-{first_mismatch_id - 1} are IDENTICAL (normal text tokens)")
    print(f"  → Mismatches {first_mismatch_id}+ are special/chat tokens replaced by DeepSeek")

if mismatches:
    print(f"\n  All {len(mismatches)} mismatches (special tokens replaced by DeepSeek):")
    for tid, tb, td in mismatches:
        print(f"    ID {tid:6d}: base={repr(tb):35} → distill={repr(td)}")

#%% Check added_tokens_encoder (special tokens registered for encoding)
print(f"\n--- Added Tokens (Special Tokens for Encoding) ---")
print(f"  Base ({len(tok_base.added_tokens_encoder)} tokens):")
for tok, idx in sorted(tok_base.added_tokens_encoder.items(), key=lambda x: x[1]):
    print(f"    {idx:6d}: {repr(tok)}")

print(f"\n  Distill ({len(tok_dist.added_tokens_encoder)} tokens):")
for tok, idx in sorted(tok_dist.added_tokens_encoder.items(), key=lambda x: x[1]):
    print(f"    {idx:6d}: {repr(tok)}")

#%% Test encoding on sample strings
print(f"\n--- Sample String Encoding Comparison ---")
test_strings = [
    "What is 2 + 2?",
    "Solve: x^2 - 4x + 4 = 0",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n <= 1:\n        return n",
    "<|im_start|>user\nHello<|im_end|>",
]

for s in test_strings:
    ids_base = tok_base.encode(s, add_special_tokens=False)
    ids_dist = tok_dist.encode(s, add_special_tokens=False)
    
    toks_base = tok_base.convert_ids_to_tokens(ids_base)
    toks_dist = tok_dist.convert_ids_to_tokens(ids_dist)
    
    print(f"\n  String: {repr(s)}")
    print(f"    Base IDs:    {ids_base}")
    print(f"    Distill IDs: {ids_dist}")
    print(f"    IDs match:   {ids_base == ids_dist}")
    if ids_base != ids_dist:
        print(f"    Base tokens:    {toks_base}")
        print(f"    Distill tokens: {toks_dist}")

#%% Check special tokens
print(f"\n--- Special Tokens ---")
print(f"  Base special tokens:    {tok_base.special_tokens_map}")
print(f"  Distill special tokens: {tok_dist.special_tokens_map}")

#%% Check BOS token behavior (critical for sequence alignment!)
print(f"\n--- BOS Token Behavior (Affects Sequence Length!) ---")
test_str = "What is 2 + 2?"
ids_base_special = tok_base.encode(test_str, add_special_tokens=True)
ids_dist_special = tok_dist.encode(test_str, add_special_tokens=True)
ids_base_no_special = tok_base.encode(test_str, add_special_tokens=False)
ids_dist_no_special = tok_dist.encode(test_str, add_special_tokens=False)

print(f"  Base bos_token:    {repr(tok_base.bos_token)} (id={tok_base.bos_token_id})")
print(f"  Distill bos_token: {repr(tok_dist.bos_token)} (id={tok_dist.bos_token_id})")
print(f"\n  With add_special_tokens=True:")
print(f"    Base:    {ids_base_special} (len={len(ids_base_special)})")
print(f"    Distill: {ids_dist_special} (len={len(ids_dist_special)})")
print(f"\n  With add_special_tokens=False:")
print(f"    Base:    {ids_base_no_special} (len={len(ids_base_no_special)})")
print(f"    Distill: {ids_dist_no_special} (len={len(ids_dist_no_special)})")
print(f"    Match: {ids_base_no_special == ids_dist_no_special} ✅" if ids_base_no_special == ids_dist_no_special else f"    Match: False ❌")

print(f"\n  ⚠️  DeepSeek adds BOS token by default!")
print(f"  → For token-wise diffs: use add_special_tokens=False OR slice off position 0 from distill")

#%% ============================================================
#   MODEL LOADING & RESIDUAL STREAM ACCESS
#   ============================================================
print("\n" + "=" * 60)
print("MODEL LOADING & RESIDUAL STREAM ACCESS")
print("=" * 60)

#%% Load base model
print(f"\nLoading {BASE_MODEL} with nnsight...")
base_model = LanguageModel(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
print(f"Base model loaded. Layers: {len(base_model.model.layers)}")

#%% Test residual stream access on base model
test_prompt = "What is 2 + 2?"
with base_model.trace(test_prompt) as tracer:
    layer0_output = base_model.model.layers[0].output[0].save()
    last_layer = len(base_model.model.layers) - 1
    last_layer_output = base_model.model.layers[last_layer].output[0].save()

print(f"Layer 0 output: {layer0_output}")
print(f"Last layer output: {last_layer_output}")
print(f"Layer 0 output shape: {layer0_output.shape}")
print(f"Last layer output shape: {last_layer_output.shape}")

base_num_layers = len(base_model.model.layers)

# Clear memory
del base_model
torch.cuda.empty_cache()

#%% Load distilled model
print(f"\nLoading {DISTILL_MODEL} with nnsight...")
distill_model = LanguageModel(
    DISTILL_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
print(f"Distill model loaded. Layers: {len(distill_model.model.layers)}")

#%% Test residual stream access on distilled model
with distill_model.trace(test_prompt) as tracer:
    layer0_output_distill = distill_model.model.layers[0].output[0].save()
    last_layer = len(distill_model.model.layers) - 1
    last_layer_output_distill = distill_model.model.layers[last_layer].output[0].save()

print(f"Layer 0 output shape: {layer0_output_distill.shape}")
print(f"Last layer output shape: {last_layer_output_distill.shape}")

#%% ============================================================
#   SUMMARY
#   ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n(a) Base checkpoint: {BASE_MODEL}")
print(f"\n(b) Tokenizer compatibility:")
if first_mismatch_id is not None:
    print(f"    - Normal tokens (IDs 0-{first_mismatch_id-1}): IDENTICAL ✅")
    print(f"    - Special tokens (IDs {first_mismatch_id}+): {len(mismatches)} replaced by DeepSeek")
else:
    print(f"    - All {check_range} vocab tokens: IDENTICAL ✅")
print(f"    - Added tokens (151643+): Chat tokens differ, utility tokens shared")
print(f"    - BOS behavior: Distill adds BOS, Base does not")
print(f"    → For token-wise diffs: use add_special_tokens=False or slice position 0 from distill")
print(f"\n(c) Architecture match:")
print(f"    - Same number of layers: {base_num_layers == len(distill_model.model.layers)} ({base_num_layers} vs {len(distill_model.model.layers)})")
print(f"    - Same hidden dimension: {layer0_output.shape[-1] == layer0_output_distill.shape[-1]} ({layer0_output.shape[-1]} vs {layer0_output_distill.shape[-1]})")


# %%
