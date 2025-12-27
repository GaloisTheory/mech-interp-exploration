"""
Project configuration for sycophancy exploration.

Import this module before any HuggingFace/transformers imports to ensure
the cache is directed to /workspace (which has ample storage).
"""
import os

# Redirect HuggingFace cache to /workspace (root filesystem has limited space)
HF_CACHE_DIR = "/workspace/.cache/huggingface"
os.environ["HF_HOME"] = HF_CACHE_DIR

