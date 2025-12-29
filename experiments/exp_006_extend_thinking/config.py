"""Central configuration for IPHR experiment."""
from dataclasses import dataclass, field
from typing import List

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
THINK_END_ID = 151649  # </think> token


@dataclass
class GenerationConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 800


@dataclass
class ExperimentConfig:
    conditions: List[str] = field(default_factory=lambda: ["normal", "extended_1x"])
    samples_per_question: int = 5  # N responses per question
    test_mode_n: int = 3  # Number of pairs for quick testing


CONDITION_SETTINGS = {
    "normal": {"max_tokens": 800, "intercept_count": 0},
    "extended_1x": {"max_tokens": 1200, "intercept_count": 1},
    "extended_2x": {"max_tokens": 1600, "intercept_count": 2},
    "extended_5x": {"max_tokens": 5000, "intercept_count": 5},
    "extended_10x": {"max_tokens": 10000, "intercept_count": 10},
}

CONTINUATION_TEXT = "\n\nWait, let me reconsider this step by step..."

