#!/usr/bin/env python3
"""Minimal test to isolate Jupyter vs GPU issues."""
# %%
import os
import sys
import time

print("Step 1: Basic imports...")
start = time.time()

_bbq_dir = '/workspace/experiments/exp_006_extend_thinking/bbq'
_shared_dir = '/workspace/experiments/exp_006_extend_thinking'
if _bbq_dir not in sys.path:
    sys.path.insert(0, _bbq_dir)
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)
os.chdir(_bbq_dir)

# %%
print(f"  Done ({time.time()-start:.1f}s)")

print("Step 2: PyTorch import...")
start = time.time()
import torch
print(f"  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"  Done ({time.time()-start:.1f}s)")
# %%
print("Step 3: Transformers import...")
start = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"  Done ({time.time()-start:.1f}s)")

print("Step 4: Local imports...")
start = time.time()
from shared.config import get_think_end_id, format_bbq_prompt
from shared.generation import generate_with_custom_override
from data.bbq_dataset import load_bbq_items
from constants import FEW_SHOT_EXAMPLES
print(f"  Done ({time.time()-start:.1f}s)")
# %%
print("Step 5: Load tokenizer...")
start = time.time()
MODEL_NAME = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"  Done ({time.time()-start:.1f}s)")

print("Step 6: Load model (this takes a minute)...")
start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",  # Blackwell compatibility
)
print(f"  Model loaded on {model.device} ({time.time()-start:.1f}s)")
# %%
print("Step 7: Quick inference test...")
start = time.time()
test_input = tokenizer("Hello, world", return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**test_input, max_new_tokens=20, do_sample=False)
result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"  Output: {result[:100]}...")
print(f"  Done ({time.time()-start:.1f}s)")

print("\n✅ All steps completed successfully!")
print("If this works but Jupyter hangs, the issue is Jupyter-specific.")
# %%







