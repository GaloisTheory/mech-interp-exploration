"""Generation logic for normal and extended thinking conditions.

Supports both IPHR (YES/NO) and BBQ (A/B/C) answer formats.
"""

import gc
import re
import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Literal, Union
from dataclasses import dataclass
from tqdm import tqdm

from .config import (
    MODEL_NAME,
    get_think_end_id,
    CONTINUATION_TEXT,
    CONDITION_SETTINGS,
    GenerationConfig,
)


def clear_gpu_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Generation Result
# =============================================================================

@dataclass
class GenerationResult:
    """Result of a single generation."""
    full_output: str
    answer: str  # "YES", "NO", "A", "B", "C", or "INVALID"
    token_count: int
    think_end_positions: list
    condition: str


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_answer_yesno(text: str) -> str:
    """Extract YES or NO from model output (for IPHR).
    
    Only searches AFTER </think> token to avoid parsing incomplete reasoning.
    Uses word boundary matching to avoid false positives like 'NO' in 'NORTH'.
    """
    text_upper = text.upper()
    
    # Only parse after </think>
    if "</THINK>" in text_upper:
        after_think = text_upper.split("</THINK>")[-1]
    else:
        # No </think> found - likely truncated, return INVALID
        return "INVALID"
    
    after_think = after_think.strip()
    
    # Use word boundaries to find standalone YES/NO
    yes_matches = list(re.finditer(r'\bYES\b', after_think))
    no_matches = list(re.finditer(r'\bNO\b', after_think))
    
    has_yes = len(yes_matches) > 0
    has_no = len(no_matches) > 0
    
    if has_yes and not has_no:
        return "YES"
    elif has_no and not has_yes:
        return "NO"
    elif has_yes and has_no:
        # Find the first occurrence of each (answer usually comes first after </think>)
        yes_pos = yes_matches[0].start()
        no_pos = no_matches[0].start()
        return "YES" if yes_pos < no_pos else "NO"
    else:
        return "INVALID"


def extract_answer_mcq(text: str) -> str:
    """Extract A, B, or C from model output (for BBQ).
    
    Only searches AFTER </think> token to avoid parsing incomplete reasoning.
    Returns the first standalone letter found.
    """
    text_upper = text.upper()
    
    # Only parse after </think>
    if "</THINK>" in text_upper:
        after_think = text_upper.split("</THINK>")[-1]
    else:
        # No </think> found - likely truncated, return INVALID
        return "INVALID"
    
    after_think = after_think.strip()
    
    # Look for standalone A, B, or C
    # Match: start of string or non-letter, then A/B/C, then non-letter or end
    match = re.search(r'(?:^|[^A-Z])([ABC])(?:[^A-Z]|$)', after_think)
    
    if match:
        return match.group(1)
    
    # Fallback: check if the entire response is just the letter
    if after_think in ["A", "B", "C"]:
        return after_think
    
    # Check for "A.", "B.", "C." patterns
    for letter in ["A", "B", "C"]:
        if after_think.startswith(f"{letter}.") or after_think.startswith(f"{letter}:"):
            return letter
    
    return "INVALID"


# =============================================================================
# KV Cache Utilities
# =============================================================================

def clone_kv_cache(past_key_values):
    """Clone KV cache for forking generation paths.
    
    Handles both legacy tuple format and new DynamicCache format.
    Uses to_legacy_cache/from_legacy_cache for DynamicCache to ensure
    compatibility across transformers versions.
    """
    if past_key_values is None:
        return None
    
    # Check if it's a DynamicCache object (new HuggingFace format)
    if hasattr(past_key_values, 'to_legacy_cache'):
        from transformers.cache_utils import DynamicCache
        # Convert to legacy format (tuple of (key, value) tuples)
        legacy = past_key_values.to_legacy_cache()
        # Clone all tensors
        cloned_legacy = tuple(
            (k.clone(), v.clone()) for k, v in legacy
        )
        # Convert back to DynamicCache
        return DynamicCache.from_legacy_cache(cloned_legacy)
    
    # Legacy tuple format
    return tuple(
        (k.clone(), v.clone()) for k, v in past_key_values
    )


# =============================================================================
# Token Sampling
# =============================================================================

def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample next token with temperature and nucleus sampling."""
    logits = logits / temperature
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# =============================================================================
# Generation Functions
# =============================================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    condition: str,
    answer_type: Literal["yesno", "mcq"] = "mcq",
    gen_config: GenerationConfig = None,
    model_name: str = None,
    verbose: bool = False
) -> GenerationResult:
    """
    Generate response with proper O(n) KV caching for extended conditions.
    
    Uses token-by-token generation with persistent KV cache to avoid
    recomputing attention after each intercept.
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
    if model_name is None:
        model_name = MODEL_NAME
    
    think_end_id = get_think_end_id(model_name)
    settings = CONDITION_SETTINGS[condition]
    max_tokens = settings["max_tokens"]
    intercept_count = settings["intercept_count"]
    
    # Get underlying HuggingFace model
    hf_model = getattr(model, '_model', model)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(hf_model.device)
    
    think_end_encounters = 0
    think_end_positions = []
    generated_ids = []
    
    # KV cache persists across tokens - this is the key to O(n)
    past_key_values = None
    
    for step in range(max_tokens):
        with torch.no_grad():
            outputs = hf_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        # Get logits for last position
        logits = outputs.logits[0, -1, :]
        
        # Sample next token
        next_token_id = sample_token(logits, gen_config.temperature, gen_config.top_p)
        
        # Update KV cache
        past_key_values = outputs.past_key_values
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            if verbose:
                print(f"  [EOS at position {len(generated_ids)}]")
            break
        
        # Check for </think>
        if next_token_id == think_end_id:
            think_end_positions.append(len(generated_ids))
            think_end_encounters += 1
            
            if think_end_encounters <= intercept_count:
                # INTERCEPT: replace </think> with continuation
                if verbose:
                    print(f"  [INTERCEPT #{think_end_encounters} at position {len(generated_ids)}]")
                
                # Encode continuation text and add to sequence
                continuation_ids = tokenizer.encode(
                    CONTINUATION_TEXT,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(hf_model.device)
                
                # Add continuation tokens to our tracking
                generated_ids.extend(continuation_ids[0].tolist())
                
                # Process continuation tokens through model to update KV cache
                with torch.no_grad():
                    cont_outputs = hf_model(
                        input_ids=continuation_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = cont_outputs.past_key_values
                
                # Next iteration will generate from last continuation token
                input_ids = continuation_ids[:, -1:]
                continue
        
        # Normal token: add to sequence
        generated_ids.append(next_token_id)
        input_ids = torch.tensor([[next_token_id]], device=hf_model.device)
        
        if verbose and len(generated_ids) % 500 == 0:
            print(f"  [Generated {len(generated_ids)} tokens...]")
    
    # Decode
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # Extract answer based on type
    if answer_type == "yesno":
        answer = extract_answer_yesno(full_output)
    else:
        answer = extract_answer_mcq(full_output)
    
    return GenerationResult(
        full_output=full_output,
        answer=answer,
        token_count=len(generated_ids),
        think_end_positions=think_end_positions,
        condition=condition,
    )


# =============================================================================
# Incremental Extension Generation
# =============================================================================

# Map conditions to their intercept counts
CONDITION_INTERCEPTS = {
    "normal": 0,
    "extended_1x": 1,
    "extended_2x": 2,
    "extended_5x": 5,
}


def _generate_answer_from_checkpoint(
    hf_model,
    tokenizer,
    generated_ids: List[int],
    past_key_values,
    think_end_id: int,
    think_end_positions: List[int],
    condition: str,
    answer_type: Literal["yesno", "mcq"],
    gen_config: GenerationConfig,
    max_answer_tokens: int = 500,
    verbose: bool = False,
) -> GenerationResult:
    """Generate answer after forking from a checkpoint.
    
    Takes the state at a </think> boundary and:
    1. Adds the </think> token
    2. Generates until EOS or max tokens
    3. Returns the complete result
    """
    # Clone the lists so we don't modify the originals
    answer_ids = generated_ids.copy()
    answer_think_positions = think_end_positions.copy()
    
    # Add </think> token
    answer_ids.append(think_end_id)
    answer_think_positions.append(len(answer_ids) - 1)
    
    # Process </think> through model to update KV cache
    think_end_tensor = torch.tensor([[think_end_id]], device=hf_model.device)
    with torch.no_grad():
        outputs = hf_model(
            input_ids=think_end_tensor,
            past_key_values=past_key_values,
            use_cache=True,
        )
    kv = outputs.past_key_values
    input_ids = think_end_tensor
    
    # Generate answer tokens
    for _ in range(max_answer_tokens):
        with torch.no_grad():
            outputs = hf_model(
                input_ids=input_ids,
                past_key_values=kv,
                use_cache=True,
            )
        
        logits = outputs.logits[0, -1, :]
        next_token_id = sample_token(logits, gen_config.temperature, gen_config.top_p)
        kv = outputs.past_key_values
        
        if next_token_id == tokenizer.eos_token_id:
            if verbose:
                print(f"    [{condition}] EOS at position {len(answer_ids)}")
            break
        
        answer_ids.append(next_token_id)
        input_ids = torch.tensor([[next_token_id]], device=hf_model.device)
    
    # Decode and extract answer
    full_output = tokenizer.decode(answer_ids, skip_special_tokens=False)
    
    if answer_type == "yesno":
        answer = extract_answer_yesno(full_output)
    else:
        answer = extract_answer_mcq(full_output)
    
    return GenerationResult(
        full_output=full_output,
        answer=answer,
        token_count=len(answer_ids),
        think_end_positions=answer_think_positions,
        condition=condition,
    )


def generate_with_extensions(
    model,
    tokenizer,
    prompt: str,
    conditions: List[str],
    answer_type: Literal["yesno", "mcq"] = "mcq",
    gen_config: GenerationConfig = None,
    model_name: str = None,
    verbose: bool = False,
) -> Dict[str, GenerationResult]:
    """
    Generate all conditions in a single pass with shared prefixes.
    
    This is more efficient than generating each condition from scratch because:
    1. The shared thinking prefix is only computed once
    2. ext1x extends normal's thinking, ext2x extends ext1x, etc.
    
    At each </think> token, we:
    - Fork for conditions that end at this intercept count
    - Inject continuation and continue for remaining conditions
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt: The input prompt
        conditions: List of conditions to generate (e.g., ["normal", "extended_1x"])
        answer_type: "yesno" for IPHR, "mcq" for BBQ
        gen_config: Generation parameters
        model_name: Model name for token ID lookup
        verbose: Print debug info
    
    Returns:
        Dict mapping condition name to GenerationResult
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
    if model_name is None:
        model_name = MODEL_NAME
    
    think_end_id = get_think_end_id(model_name)
    
    # Get underlying HuggingFace model
    hf_model = getattr(model, '_model', model)
    
    # Map conditions to intercept counts and find max
    condition_to_intercept = {c: CONDITION_INTERCEPTS[c] for c in conditions}
    max_intercepts = max(condition_to_intercept.values())
    
    # Find max tokens needed (use the highest condition's limit)
    max_tokens = max(CONDITION_SETTINGS[c]["max_tokens"] for c in conditions)
    
    # Initialize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(hf_model.device)
    generated_ids = []
    think_end_positions = []
    past_key_values = None
    intercept_count = 0
    
    results = {}
    
    for step in range(max_tokens):
        with torch.no_grad():
            outputs = hf_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        logits = outputs.logits[0, -1, :]
        next_token_id = sample_token(logits, gen_config.temperature, gen_config.top_p)
        past_key_values = outputs.past_key_values
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            if verbose:
                print(f"  [EOS at position {len(generated_ids)}, intercept_count={intercept_count}]")
            # Generate results for any conditions that haven't been captured yet
            for cond in conditions:
                if cond not in results:
                    # This condition didn't get enough intercepts before EOS
                    # Generate result from current state
                    result = _generate_answer_from_checkpoint(
                        hf_model, tokenizer,
                        generated_ids, clone_kv_cache(past_key_values),
                        think_end_id, think_end_positions,
                        cond, answer_type, gen_config, verbose=verbose,
                    )
                    results[cond] = result
            break
        
        # Check for </think>
        if next_token_id == think_end_id:
            if verbose:
                print(f"  [</think> at position {len(generated_ids)}, intercept #{intercept_count}]")
            
            # Check which conditions end at this intercept count
            for cond, intercepts in condition_to_intercept.items():
                if intercepts == intercept_count and cond not in results:
                    # This condition ends here - fork and generate answer
                    if verbose:
                        print(f"    Forking for {cond}")
                    result = _generate_answer_from_checkpoint(
                        hf_model, tokenizer,
                        generated_ids, clone_kv_cache(past_key_values),
                        think_end_id, think_end_positions,
                        cond, answer_type, gen_config, verbose=verbose,
                    )
                    results[cond] = result
            
            intercept_count += 1
            
            # Check if we've captured all conditions
            if len(results) == len(conditions):
                if verbose:
                    print(f"  [All conditions captured, stopping]")
                break
            
            # Check if we've exceeded max intercepts needed
            if intercept_count > max_intercepts:
                if verbose:
                    print(f"  [Max intercepts reached, stopping]")
                # Generate results for any conditions that haven't been captured yet
                for cond in conditions:
                    if cond not in results:
                        if verbose:
                            print(f"    Generating fallback for {cond} (not enough intercepts)")
                        result = _generate_answer_from_checkpoint(
                            hf_model, tokenizer,
                            generated_ids, clone_kv_cache(past_key_values),
                            think_end_id, think_end_positions,
                            cond, answer_type, gen_config, verbose=verbose,
                        )
                        results[cond] = result
                break
            
            # Inject continuation and continue
            if verbose:
                print(f"    Injecting continuation for further extensions")
            
            continuation_ids = tokenizer.encode(
                CONTINUATION_TEXT,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(hf_model.device)
            
            # Add continuation tokens to our tracking
            generated_ids.extend(continuation_ids[0].tolist())
            
            # Process continuation through model
            with torch.no_grad():
                cont_outputs = hf_model(
                    input_ids=continuation_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = cont_outputs.past_key_values
            input_ids = continuation_ids[:, -1:]
            continue
        
        # Normal token
        generated_ids.append(next_token_id)
        input_ids = torch.tensor([[next_token_id]], device=hf_model.device)
        
        if verbose and len(generated_ids) % 500 == 0:
            print(f"  [Generated {len(generated_ids)} tokens...]")
    
    # Fallback: ensure all conditions have results (handles max_tokens reached case)
    for cond in conditions:
        if cond not in results:
            if verbose:
                print(f"  [Fallback] Generating result for {cond} (loop ended without capturing)")
            result = _generate_answer_from_checkpoint(
                hf_model, tokenizer,
                generated_ids, clone_kv_cache(past_key_values) if past_key_values else None,
                think_end_id, think_end_positions,
                cond, answer_type, gen_config, verbose=verbose,
            )
            results[cond] = result
    
    return results


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    condition: str,
    answer_type: Literal["yesno", "mcq"] = "mcq",
    gen_config: GenerationConfig = None,
    model_name: str = None,
    verbose: bool = False
) -> List[GenerationResult]:
    """
    Generate responses for multiple prompts.
    
    For 'normal' condition: Full batching with HF generate().
    For 'extended_*' conditions: Sequential with O(n) KV caching per sequence.
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
    if model_name is None:
        model_name = MODEL_NAME
    
    think_end_id = get_think_end_id(model_name)
    settings = CONDITION_SETTINGS[condition]
    intercept_count = settings["intercept_count"]
    
    # Get underlying HuggingFace model
    hf_model = getattr(model, '_model', model)
    
    # For extended conditions, run sequentially with KV cache
    if intercept_count > 0:
        results = []
        for p in tqdm(prompts, desc=f"  {condition}", unit="sample", leave=False):
            results.append(generate_response(
                model, tokenizer, p, condition, answer_type, gen_config, model_name, verbose
            ))
        return results
    
    # For normal condition: use HF generate() with batching
    max_tokens = settings["max_tokens"]
    
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(hf_model.device)
    
    with torch.no_grad():
        outputs = hf_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )
    
    results = []
    for i, seq in enumerate(outputs.sequences):
        generated_ids = seq[inputs.input_ids.shape[1]:]
        full_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        think_end_positions = []
        for j, tok_id in enumerate(generated_ids.tolist()):
            if tok_id == think_end_id:
                think_end_positions.append(j)
        
        # Extract answer based on type
        if answer_type == "yesno":
            answer = extract_answer_yesno(full_output)
        else:
            answer = extract_answer_mcq(full_output)
        
        results.append(GenerationResult(
            full_output=full_output,
            answer=answer,
            token_count=len(generated_ids),
            think_end_positions=think_end_positions,
            condition=condition,
        ))
    
    del outputs, inputs
    clear_gpu_memory()
    
    if verbose:
        print(f"  [Batched {len(prompts)} prompts]")
    
    return results


def generate_batch_multi_condition(
    model,
    tokenizer,
    prompts: List[str],
    conditions: List[str],
    answer_type: Literal["yesno", "mcq"] = "mcq",
    gen_config: GenerationConfig = None,
    model_name: str = None,
    verbose: bool = False,
) -> Dict[str, List[GenerationResult]]:
    """
    Generate responses for multiple prompts across multiple conditions.
    
    Uses incremental extension: ext1x extends normal, ext2x extends ext1x, etc.
    This is more efficient than generating each condition from scratch.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompts: List of prompts to generate
        conditions: List of conditions (e.g., ["normal", "extended_1x"])
        answer_type: "yesno" for IPHR, "mcq" for BBQ
        gen_config: Generation parameters
        model_name: Model name for token ID lookup
        verbose: Print debug info
    
    Returns:
        Dict mapping condition -> list of GenerationResult (one per prompt)
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
    if model_name is None:
        model_name = MODEL_NAME
    
    # Initialize results dict
    results_by_condition: Dict[str, List[GenerationResult]] = {c: [] for c in conditions}
    
    # Process each prompt with incremental extensions
    for prompt in tqdm(prompts, desc="  samples", unit="sample", leave=False):
        # Generate all conditions for this prompt in one pass
        prompt_results = generate_with_extensions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            conditions=conditions,
            answer_type=answer_type,
            gen_config=gen_config,
            model_name=model_name,
            verbose=verbose,
        )
        
        # Distribute results to per-condition lists
        for cond in conditions:
            results_by_condition[cond].append(prompt_results[cond])
    
    return results_by_condition


# =============================================================================
# Interactive Playground Generation
# =============================================================================

def generate_with_custom_override(
    model,
    tokenizer,
    prompt: str,
    token_to_match: Union[str, int],
    override_text: Union[str, Callable[[int], str]],
    max_tokens: int = 2000,
    intercept_count: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    streaming: bool = False,
    model_name: str = None,
    enable_thinking: bool = True,
    token_position_overrides: List[tuple] = None,
) -> GenerationResult:
    """
    Generate with custom token override for interactive experimentation.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt: Input prompt
        token_to_match: Token string (e.g., "</think>") or token ID to intercept
        override_text: Text to inject when token is matched. Can be:
                      - A string (same text for all intercepts)
                      - A callable(intercept_num: int) -> str (different text per intercept)
        max_tokens: Maximum tokens to generate
        intercept_count: How many times to intercept (0 = no override)
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        streaming: If True, print tokens as generated
        model_name: Model name for token ID lookup
        enable_thinking: If True, use chat template with thinking mode enabled (for Qwen3)
        token_position_overrides: List of (token_position, text) tuples to inject at specific token counts
    
    Returns:
        GenerationResult with full output, extracted answer, and metadata
    """
    # Resolve token_to_match to ID
    if isinstance(token_to_match, str):
        if model_name and token_to_match == "</think>":
            match_token_id = get_think_end_id(model_name)
        else:
            # Try to encode the string
            encoded = tokenizer.encode(token_to_match, add_special_tokens=False)
            if len(encoded) == 1:
                match_token_id = encoded[0]
            else:
                print(f"Warning: '{token_to_match}' encodes to {len(encoded)} tokens, using first")
                match_token_id = encoded[0] if encoded else None
    else:
        match_token_id = token_to_match
    
    # Check if override_text is callable (for per-intercept customization)
    override_is_callable = callable(override_text)
    
    # Prepare token position overrides
    position_overrides = {}
    if token_position_overrides:
        for pos, text in token_position_overrides:
            position_overrides[pos] = text
    
    print(f"[Config] Token to match: {token_to_match} (ID: {match_token_id})")
    if override_is_callable:
        print(f"[Config] Override text: <dynamic per intercept>")
    else:
        print(f"[Config] Override text: {override_text[:50]}...")
    print(f"[Config] Intercept count: {intercept_count}")
    print(f"[Config] Enable thinking: {enable_thinking}")
    if position_overrides:
        print(f"[Config] Token position overrides at: {sorted(position_overrides.keys())}")
    print()
    
    # Get underlying HuggingFace model
    hf_model = getattr(model, '_model', model)
    device = next(hf_model.parameters()).device
    
    # Format prompt with chat template if thinking mode is enabled
    if enable_thinking:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    encounters = 0
    match_positions = []
    generated_ids = []
    past_key_values = None
    
    if streaming:
        print("=" * 60)
        print("GENERATION OUTPUT (streaming)")
        print("=" * 60)
    
    for step in range(max_tokens):
        with torch.no_grad():
            outputs = hf_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        
        logits = outputs.logits[0, -1, :]
        next_token_id = sample_token(logits, temperature, top_p)
        past_key_values = outputs.past_key_values
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            if streaming:
                print("\n[EOS]")
            break
        
        # Check for token match
        if match_token_id is not None and next_token_id == match_token_id:
            match_positions.append(len(generated_ids))
            encounters += 1
            
            if streaming:
                token_str = tokenizer.decode([next_token_id])
                print(f"\n\n>>> [MATCH #{encounters}: '{token_str}' at position {len(generated_ids)}]")
            
            if encounters <= intercept_count:
                # INTERCEPT: inject override text
                # Get the text for this intercept (call function if callable)
                current_override = override_text(encounters) if override_is_callable else override_text
                
                if streaming:
                    print(f">>> [INJECTING OVERRIDE TEXT (intercept #{encounters})]")
                    print(f">>> {current_override}")
                    print()
                
                override_ids = tokenizer.encode(
                    current_override,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(device)
                
                generated_ids.extend(override_ids[0].tolist())
                
                # Process override through model
                with torch.no_grad():
                    override_outputs = hf_model(
                        input_ids=override_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = override_outputs.past_key_values
                
                # FIX: Sample next token from override output logits
                # (Previously set input_ids = override_ids[:, -1:] which caused
                # the last override token to be processed TWICE, corrupting KV cache)
                next_logits = override_outputs.logits[0, -1, :]
                next_token_id = sample_token(next_logits, temperature, top_p)
                
                # Check EOS
                if next_token_id == tokenizer.eos_token_id:
                    if streaming:
                        print("\n[EOS]")
                    break
                
                generated_ids.append(next_token_id)
                if streaming:
                    print(tokenizer.decode([next_token_id]), end="", flush=True)
                
                input_ids = torch.tensor([[next_token_id]], device=device)
                continue
            else:
                # No more intercepts - add the matched token normally
                generated_ids.append(next_token_id)
                if streaming:
                    print(f">>> [NO MORE INTERCEPTS - letting token through]")
        else:
            # Normal token
            generated_ids.append(next_token_id)
        
        # Streaming output
        if streaming:
            token_str = tokenizer.decode([next_token_id])
            print(token_str, end="", flush=True)
        
        input_ids = torch.tensor([[next_token_id]], device=device)
        
        # Check for token position override
        current_token_count = len(generated_ids)
        if current_token_count in position_overrides:
            pos_override_text = position_overrides[current_token_count]
            
            if streaming:
                print(f"\n\n>>> [TOKEN POSITION OVERRIDE at {current_token_count} tokens]")
                print(f">>> {pos_override_text}")
                print()
            
            pos_override_ids = tokenizer.encode(
                pos_override_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(device)
            
            generated_ids.extend(pos_override_ids[0].tolist())
            
            # Process override through model
            with torch.no_grad():
                pos_override_outputs = hf_model(
                    input_ids=pos_override_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = pos_override_outputs.past_key_values
            
            # FIX: Sample next token from override output logits
            # (Previously set input_ids = pos_override_ids[:, -1:] which caused
            # the last override token to be processed TWICE, corrupting KV cache)
            pos_next_logits = pos_override_outputs.logits[0, -1, :]
            pos_next_token_id = sample_token(pos_next_logits, temperature, top_p)
            
            # Check EOS
            if pos_next_token_id == tokenizer.eos_token_id:
                if streaming:
                    print("\n[EOS]")
                break
            
            generated_ids.append(pos_next_token_id)
            if streaming:
                print(tokenizer.decode([pos_next_token_id]), end="", flush=True)
            
            input_ids = torch.tensor([[pos_next_token_id]], device=device)
    
    if streaming:
        print("\n" + "=" * 60)
    
    # Decode full output
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
    answer = extract_answer_mcq(full_output)
    
    return GenerationResult(
        full_output=full_output,
        answer=answer,
        token_count=len(generated_ids),
        think_end_positions=match_positions,
        condition=f"override_{intercept_count}x",
    )
