# %% [markdown]
# # Steering Vector Interactive Playground
# 
# Fast iteration script - modify CONFIG and run cells.
# Swap model_name to test cross-model transfer.

# %% Imports
import sys

# Add workspace root to path for imports (works in both script and notebook mode)
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

from experiments.config import *  # Sets HF_HOME before any HF imports

from experiments.exp_003_steering_playground.steering_utils import (
    load_model_and_vectors,
    compare_generations,
    sanity_check_steering,
    interactive_demo,
    STEERING_CONFIG
)

# %% ============== CONFIG - MODIFY FOR EXPERIMENTS ==============

# Model to steer (swap these to test transfer!)
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"  # Base model - uncomment to test transfer

# Path to steering vectors (extracted from distill model)
VECTOR_PATH = "/workspace/third_party/steering-thinking-llms/train-steering-vectors/results/vars/mean_vectors_deepseek-r1-distill-qwen-1.5b.pt"

# Default steering settings
STEERING_LABEL = "backtracking"  # backtracking, uncertainty-estimation, example-testing, adding-knowledge, etc.
COEFFICIENT = 0.5  # Safe range: 0.3-0.6. >0.7 may cause instability
MAX_NEW_TOKENS = 300

# %% Load Model and Vectors
print(f"Loading model: {MODEL_NAME}")
model, tokenizer, feature_vectors = load_model_and_vectors(MODEL_NAME, VECTOR_PATH)
print(f"Available steering vectors: {list(feature_vectors.keys())}")

# %% Sanity Check - Verify steering modifies outputs
if feature_vectors:
    sanity_check_steering(model, tokenizer, feature_vectors)

# %% Quick Test - Simple prompt
prompt = "What is 15% of 80?"
compare_generations(
    prompt, model, tokenizer, feature_vectors,
    coefficient=COEFFICIENT,
    steering_label=STEERING_LABEL,
    max_new_tokens=MAX_NEW_TOKENS
)

# %% Math Problem - Good for testing backtracking
prompt = """A train leaves station A at 2:00 PM traveling at 60 mph. Another train leaves station B at 3:00 PM traveling at 80 mph towards station A. The distance between stations A and B is 300 miles. At what time will the two trains meet?"""
compare_generations(
    prompt, model, tokenizer, feature_vectors,
    coefficient=0.5,
    steering_label="backtracking",
    max_new_tokens=500
)

# %% Test Different Coefficients
prompt = "Explain why the sky is blue."
for coef in [0.3, 0.5, 0.7]:
    print(f"\n{'='*60}\nCOEFFICIENT: {coef}\n{'='*60}")
    compare_generations(
        prompt, model, tokenizer, feature_vectors,
        coefficient=coef,
        steering_label=STEERING_LABEL,
        max_new_tokens=200
    )

# %% Test Different Steering Vectors
prompt = "What is the capital of France?"
for label in ["backtracking", "uncertainty-estimation", "adding-knowledge"]:
    print(f"\n{'='*60}\nSTEERING: {label}\n{'='*60}")
    compare_generations(
        prompt, model, tokenizer, feature_vectors,
        coefficient=0.5,
        steering_label=label,
        max_new_tokens=200
    )

# %% Custom Experiment Cell - Copy and modify as needed
prompt = "YOUR PROMPT HERE"
compare_generations(
    prompt, model, tokenizer, feature_vectors,
    coefficient=0.5,
    steering_label="backtracking",
    max_new_tokens=300
)

# %% Interactive Demo (optional)
interactive_demo(model, tokenizer, feature_vectors)

