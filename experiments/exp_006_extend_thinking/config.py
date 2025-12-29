"""Central configuration for IPHR experiment."""
from dataclasses import dataclass, field
from typing import List

# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME = "Qwen/QwQ-32B"

# </think> token IDs vary by model family
THINK_END_IDS = {
    # DeepSeek R1 distillations (based on older Qwen tokenizer)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 151649,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 151649,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 151649,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 151649,
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


@dataclass
class GenerationConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 1200  # Increased for larger model


@dataclass
class ExperimentConfig:
    conditions: List[str] = field(default_factory=lambda: ["normal", "extended_1x"])
    samples_per_question: int = 5  # N responses per question
    test_mode_n: int = 3  # Number of pairs for quick testing


CONDITION_SETTINGS = {
    "normal": {"max_tokens": 1200, "intercept_count": 0},
    "extended_1x": {"max_tokens": 2000, "intercept_count": 1},
    "extended_2x": {"max_tokens": 3000, "intercept_count": 2},
    "extended_5x": {"max_tokens": 8000, "intercept_count": 5},
    "extended_10x": {"max_tokens": 16000, "intercept_count": 10},
}

CONTINUATION_TEXT = "\n\nWait, let me reconsider this step by step..."

