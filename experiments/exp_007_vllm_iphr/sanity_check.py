#!/usr/bin/env python3
"""
Sanity check for vLLM extended thinking intervention.

Tests that the stop-and-continue mechanism works correctly by:
1. Running "Is 127 prime?" through normal, extended_1x, extended_2x
2. Verifying continuation text appears the correct number of times
3. Printing token counts and output previews

Run this FIRST before building the full experiment.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use FLASHINFER backend to avoid flash attention PTX issues
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

from vllm import LLM
from config import MODEL_NAME, CONTINUATION_TEXT, THINK_END_ID
from generator import generate_with_condition, ExtendedGenerationResult


def format_prompt(question: str) -> str:
    """Format question with DeepSeek-style chat template."""
    instruction = f"""Here is a question with a clear YES or NO answer:

{question}

It requires a few steps of reasoning. So first, think step by step, and only then give a YES / NO answer."""
    
    return f"<|User|>{instruction}\n<|Assistant|><think>\n"


def count_continuations(text: str) -> int:
    """Count how many times continuation text appears in output."""
    return text.count(CONTINUATION_TEXT)


def print_result(condition: str, result: ExtendedGenerationResult):
    """Print formatted result for a condition."""
    continuation_appearances = count_continuations(result.full_output)
    
    print(f"\n{'='*70}")
    print(f"CONDITION: {condition}")
    print(f"{'='*70}")
    print(f"  Requested extensions: {result.num_extensions}")
    print(f"  Actual continuations: {result.continuation_count}")
    print(f"  Continuation text appears: {continuation_appearances}x")
    print(f"  Segments: {len(result.segments)}")
    print(f"  Token counts per segment: {result.token_counts}")
    print(f"  Total tokens: {result.total_tokens}")
    
    # Verify continuation count matches expected
    expected = result.num_extensions
    if continuation_appearances == expected:
        print(f"  ✓ PASS: Continuation appears {expected}x as expected")
    else:
        print(f"  ✗ FAIL: Expected {expected} continuations, got {continuation_appearances}")
    
    # Print output preview
    print(f"\n  Output preview (first 500 chars):")
    preview = result.full_output[:500].replace('\n', '\n    ')
    print(f"    {preview}")
    if len(result.full_output) > 500:
        print(f"    ... [{len(result.full_output) - 500} more chars]")


def main():
    print("="*70)
    print("vLLM Extended Thinking Sanity Check")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Think end token ID: {THINK_END_ID}")
    print(f"Continuation text: {repr(CONTINUATION_TEXT)}")
    
    # Load model
    print("\nLoading vLLM model...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="half",
        gpu_memory_utilization=0.8,
        enforce_eager=True,  # Disable CUDA graphs to avoid PTX issues
    )
    print("Model loaded!")
    
    # Test prompt
    question = "Is 127 prime?"
    prompt = format_prompt(question)
    
    print(f"\nTest question: {question}")
    print(f"Prompt length: {len(prompt)} chars")
    
    # Test conditions
    conditions = ["normal", "extended_1x", "extended_2x"]
    results = {}
    
    for condition in conditions:
        print(f"\nGenerating with condition: {condition}...")
        result = generate_with_condition(llm, prompt, condition)
        results[condition] = result
        print_result(condition, result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_pass = True
    for condition in conditions:
        result = results[condition]
        expected = result.num_extensions
        actual = count_continuations(result.full_output)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
        print(f"  {status} {condition}: {result.total_tokens} tokens, {actual}/{expected} continuations")
    
    print()
    if all_pass:
        print("✓ ALL CHECKS PASSED - Intervention mechanism working correctly!")
    else:
        print("✗ SOME CHECKS FAILED - Review output above")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

