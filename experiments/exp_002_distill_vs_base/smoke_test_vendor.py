#%% Imports & Config
"""
Smoke test for model loading and generation.

Verifies that both DeepSeek-R1-Distill-Qwen-7B and Qwen2.5-Math-7B
load correctly and can generate completions.
"""
from experiments.config import *  # Sets HF env vars before any HF imports

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#%% Configuration
PROMPT = "What is 2 + 2? Think step by step."
MAX_NEW_TOKENS = 64

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#%% Helper function
def generate_completion(model_id: str, prompt: str, max_new_tokens: int = 64) -> str:
    """Load a model and generate a single completion."""
    print(f"\n{'=' * 60}")
    print(f"Loading: {model_id}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Cleanup to free VRAM for next model
    del model
    torch.cuda.empty_cache()

    return completion

#%% Test DeepSeek-R1-Distill-Qwen-7B
distill_output = generate_completion(DISTILL_MODEL, PROMPT, MAX_NEW_TOKENS)
print(f"\nOutput:\n{distill_output}")

#%% Test Qwen2.5-Math-7B (base model)
base_output = generate_completion(BASE_MODEL, PROMPT, MAX_NEW_TOKENS)
print(f"\nOutput:\n{base_output}")

#%% Compare outputs
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"\nDistill model uses <think> tags: {'<think>' in distill_output or 'think' in distill_output.lower()}")
print(f"Base model output length: {len(base_output)} chars")
print(f"Distill model output length: {len(distill_output)} chars")
