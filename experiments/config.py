"""
Project configuration.

Import this module FIRST (before any HuggingFace/transformers imports) to ensure
the cache is directed to /workspace (which has ample storage).
"""
import os
from pathlib import Path

# === Paths ===
WORKSPACE = Path("/workspace")
HF_CACHE_DIR = WORKSPACE / ".cache/huggingface"
VENDORED_REPO = WORKSPACE / "third_party/thinking-llms-interp"

# === HuggingFace environment (must be set before HF imports) ===
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")

# === Model identifiers ===
DISTILL_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
BASE_MODEL = "Qwen/Qwen2.5-Math-7B"
