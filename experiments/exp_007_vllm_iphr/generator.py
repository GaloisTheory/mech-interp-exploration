"""vLLM-based generation with stop-and-continue for extended thinking."""

from dataclasses import dataclass
from typing import List, Optional
from vllm import LLM, SamplingParams

from config import (
    THINK_END_ID,
    CONTINUATION_TEXT,
    CONDITION_SETTINGS,
    TEMPERATURE,
    TOP_P,
)


@dataclass
class ExtendedGenerationResult:
    """Result of extended generation."""
    full_output: str
    segments: List[str]  # Output segments between continuations
    token_counts: List[int]  # Token count per segment
    total_tokens: int
    num_extensions: int
    continuation_count: int  # How many times continuation text was added


def generate_extended(
    llm: LLM,
    prompt: str,
    num_extensions: int = 0,
    max_tokens_per_segment: int = 2000,
) -> ExtendedGenerationResult:
    """
    Generate with stop-and-continue for extended thinking.
    
    Uses vLLM's stop_token_ids to halt at </think>, then appends
    continuation text and continues generating.
    
    Args:
        llm: vLLM LLM instance
        prompt: The input prompt (should end with <think> or similar)
        num_extensions: Number of times to extend (0 = normal generation)
        max_tokens_per_segment: Max tokens per generation segment
        
    Returns:
        ExtendedGenerationResult with full output and metadata
    """
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens_per_segment,
        stop_token_ids=[THINK_END_ID],
    )
    
    segments: List[str] = []
    token_counts: List[int] = []
    current_prompt = prompt
    continuation_count = 0
    
    # Generate num_extensions + 1 times total
    # (1 initial generation + num_extensions continuations)
    for i in range(num_extensions + 1):
        # Generate until </think> or max_tokens
        result = llm.generate([current_prompt], sampling_params)
        output = result[0].outputs[0]
        
        segment_text = output.text
        segment_tokens = len(output.token_ids)
        
        segments.append(segment_text)
        token_counts.append(segment_tokens)
        
        # Check if we stopped due to </think> (vs max_tokens or EOS)
        stopped_at_think = (
            output.stop_reason == "stop" or 
            (output.token_ids and output.token_ids[-1] == THINK_END_ID)
        )
        
        # If more extensions to do and we stopped at </think>
        if i < num_extensions and stopped_at_think:
            # Append continuation and update prompt for next iteration
            current_prompt = current_prompt + segment_text + CONTINUATION_TEXT
            continuation_count += 1
        else:
            # Either done with extensions or didn't hit </think>
            break
    
    # Build full output
    full_output = ""
    for i, segment in enumerate(segments):
        full_output += segment
        if i < continuation_count:
            full_output += CONTINUATION_TEXT
    
    return ExtendedGenerationResult(
        full_output=full_output,
        segments=segments,
        token_counts=token_counts,
        total_tokens=sum(token_counts),
        num_extensions=num_extensions,
        continuation_count=continuation_count,
    )


def generate_with_condition(
    llm: LLM,
    prompt: str,
    condition: str,
) -> ExtendedGenerationResult:
    """
    Generate using a named condition from CONDITION_SETTINGS.
    
    Args:
        llm: vLLM LLM instance
        prompt: The input prompt
        condition: Condition name (e.g., "normal", "extended_1x")
        
    Returns:
        ExtendedGenerationResult
    """
    if condition not in CONDITION_SETTINGS:
        raise ValueError(f"Unknown condition: {condition}. Available: {list(CONDITION_SETTINGS.keys())}")
    
    settings = CONDITION_SETTINGS[condition]
    return generate_extended(
        llm=llm,
        prompt=prompt,
        num_extensions=settings["intercept_count"],
        max_tokens_per_segment=settings["max_tokens"],
    )

