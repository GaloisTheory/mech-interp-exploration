"""Configuration for vLLM-based IPHR experiment with extended thinking."""

import os

# Set cache before any HF imports
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

# Model configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# </think> token IDs vary by model family
THINK_END_IDS = {
    # DeepSeek R1 distillations (based on older Qwen tokenizer)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 151649,
    # Qwen QwQ models (newer tokenizer with additional special tokens)
    "Qwen/QwQ-32B": 151668,
    "Qwen/QwQ-32B-Preview": 151668,
}


def get_think_end_id(model_name: str) -> int:
    """Get the </think> token ID for a given model."""
    if model_name in THINK_END_IDS:
        return THINK_END_IDS[model_name]
    # Fallback: try to detect from model name pattern
    if "QwQ" in model_name:
        return 151668
    if "DeepSeek-R1-Distill" in model_name:
        return 151649
    raise ValueError(f"Unknown model {model_name}. Add to THINK_END_IDS in config.py")


THINK_END_ID = get_think_end_id(MODEL_NAME)

# Continuation text injected when we intercept </think>
CONTINUATION_TEXT = "\n\nWait, let me reconsider this step by step..."

# Condition settings: max_tokens and number of intercepts
CONDITION_SETTINGS = {
    "normal": {"max_tokens": 1200, "intercept_count": 0},
    "extended_1x": {"max_tokens": 2000, "intercept_count": 1},
    "extended_2x": {"max_tokens": 3000, "intercept_count": 2},
    "extended_5x": {"max_tokens": 8000, "intercept_count": 5},
    "extended_10x": {"max_tokens": 16000, "intercept_count": 10},
}

# Generation parameters
TEMPERATURE = 0.6
TOP_P = 0.95

