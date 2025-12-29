"""Generation logic for normal and extended thinking conditions."""

import gc
import torch
import torch.nn.functional as F
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from config import THINK_END_ID, CONTINUATION_TEXT, CONDITION_SETTINGS, GenerationConfig


def clear_gpu_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class GenerationResult:
    """Result of a single generation."""
    full_output: str
    answer: str  # "YES", "NO", or "UNCLEAR"
    token_count: int
    think_end_positions: list
    condition: str


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample next token with temperature and nucleus sampling.
    
    Args:
        logits: Logits for the last position [vocab_size]
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        
    Returns:
        Sampled token ID
    """
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-p filtering
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def extract_answer(text: str) -> str:
    """Extract YES or NO from model output.
    
    Looks for answer after </think> tag if present, otherwise
    searches the entire text.
    
    Args:
        text: Model output text
        
    Returns:
        "YES", "NO", or "UNCLEAR"
    """
    text_upper = text.upper()
    
    # Look for answer after </think>
    if "</THINK>" in text_upper:
        after_think = text_upper.split("</THINK>")[-1]
    else:
        # No </think> found - look in the whole text (might be truncated)
        after_think = text_upper
    
    # Clean up and search for answer
    after_think = after_think.strip()
    
    # Check for explicit yes/no patterns
    has_yes = "YES" in after_think
    has_no = "NO" in after_think
    
    if has_yes and not has_no:
        return "YES"
    elif has_no and not has_yes:
        return "NO"
    elif has_yes and has_no:
        # Both present - check which comes last (usually the final answer)
        yes_pos = after_think.rfind("YES")
        no_pos = after_think.rfind("NO")
        return "YES" if yes_pos > no_pos else "NO"
    else:
        return "UNCLEAR"


def generate_response_chunked(
    model,
    tokenizer,
    prompt: str,
    condition: str,
    gen_config: GenerationConfig = None,
    verbose: bool = False
) -> GenerationResult:
    """
    Efficient generation using HuggingFace generate() with KV caching.
    
    Uses chunked generation: generate until </think>, intercept if needed,
    then continue. This is O(n) instead of O(n²) for long sequences.
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
    settings = CONDITION_SETTINGS[condition]
    max_tokens = settings["max_tokens"]
    intercept_count = settings["intercept_count"]
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_length = input_ids.shape[1]
    
    think_end_encounters = 0
    think_end_positions = []
    total_generated = 0
    
    # Get the underlying HuggingFace model
    hf_model = model._model
    
    # Optimize: Use full max_tokens for first call to minimize generate() overhead
    # Only chunk if we need multiple intercepts (extended_5x, extended_10x)
    use_chunking = intercept_count > 2  # Only chunk for 5x/10x conditions
    
    while total_generated < max_tokens:
        remaining_tokens = max_tokens - total_generated
        
        with torch.no_grad():
            # For most conditions, use full remaining tokens (single generate call)
            # Only chunk for very long sequences (5x, 10x)
            if use_chunking:
                chunk_size = min(remaining_tokens, 2000)
            else:
                chunk_size = remaining_tokens  # Use full remaining tokens
            
            outputs = hf_model.generate(
                input_ids,
                max_new_tokens=chunk_size,
                do_sample=True,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, THINK_END_ID],  # Stop on </think> too
                return_dict_in_generate=True,
                use_cache=True,  # Explicitly enable KV cache
            )
        
        new_ids = outputs.sequences
        new_tokens = new_ids.shape[1] - input_ids.shape[1]
        total_generated += new_tokens
        
        if verbose and total_generated % 200 < new_tokens:
            print(f"  [Generated {total_generated} tokens...]")
        
        # Check what we stopped on
        last_token = new_ids[0, -1].item()
        
        if last_token == tokenizer.eos_token_id:
            input_ids = new_ids
            if verbose:
                print(f"  [EOS at position {total_generated}]")
            break
        
        if last_token == THINK_END_ID:
            think_end_positions.append(total_generated)
            think_end_encounters += 1
            
            if think_end_encounters <= intercept_count:
                # Intercept: remove </think> and inject continuation
                if verbose:
                    print(f"  [INTERCEPT #{think_end_encounters} at position {total_generated}]")
                
                # Remove the </think> token and add continuation
                continuation_ids = tokenizer.encode(
                    CONTINUATION_TEXT, 
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(model.device)
                
                input_ids = torch.cat([new_ids[:, :-1], continuation_ids], dim=1)
                total_generated += continuation_ids.shape[1] - 1  # -1 for removed </think>
            else:
                # Let </think> through - it's already in new_ids
                # Continue generating the answer in the same loop iteration
                input_ids = new_ids
                if verbose:
                    print(f"  [Final </think> at position {total_generated}, continuing...]")
                # Continue loop to generate answer (will hit EOS or max_tokens)
                continue
        else:
            # Normal chunk end (hit max_new_tokens)
            input_ids = new_ids
        
        # Cleanup between chunks (but don't clear cache unnecessarily)
        del outputs
        # Only clear cache periodically to avoid memory fragmentation
        if total_generated % 1000 == 0:
            clear_gpu_memory()
    
    # Decode
    full_output = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=False)
    answer = extract_answer(full_output)
    
    # Cleanup
    del input_ids
    clear_gpu_memory()
    
    return GenerationResult(
        full_output=full_output,
        answer=answer,
        token_count=total_generated,
        think_end_positions=think_end_positions,
        condition=condition,
    )


def generate_response_nnsight(
    model,
    tokenizer,
    prompt: str,
    condition: str,
    gen_config: GenerationConfig = None,
    verbose: bool = False
) -> GenerationResult:
    """
    Token-by-token generation using nnsight (for debugging/activation analysis).
    
    WARNING: This is O(n²) and very memory-intensive for long sequences.
    Use generate_response_chunked() for production runs.
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
    settings = CONDITION_SETTINGS[condition]
    max_tokens = settings["max_tokens"]
    intercept_count = settings["intercept_count"]
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_length = input_ids.shape[1]
    
    think_end_encounters = 0
    think_end_positions = []
    generated_tokens = 0
    
    while generated_tokens < max_tokens:
        # Forward pass
        with torch.no_grad():
            with model.trace(input_ids):
                logits_proxy = model.lm_head.output.save()
            
            # Get logits for last position and immediately convert to CPU for sampling
            last_logits = logits_proxy[0, -1].float().cpu()
            
            # Explicitly delete the proxy to free GPU memory
            del logits_proxy
        
        # Sample next token (on CPU to avoid GPU memory)
        next_token_id = sample_token(last_logits, gen_config.temperature, gen_config.top_p)
        del last_logits
        
        # EOS check
        if next_token_id == tokenizer.eos_token_id:
            if verbose:
                print(f"  [EOS at position {generated_tokens}]")
            break
        
        # </think> handling
        if next_token_id == THINK_END_ID:
            think_end_positions.append(generated_tokens)
            think_end_encounters += 1
            
            if think_end_encounters <= intercept_count:
                # Intercept: inject continuation instead of </think>
                if verbose:
                    print(f"  [INTERCEPT #{think_end_encounters} at position {generated_tokens}]")
                continuation_ids = tokenizer.encode(
                    CONTINUATION_TEXT, 
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(model.device)
                input_ids = torch.cat([input_ids, continuation_ids], dim=1)
                generated_tokens += continuation_ids.shape[1]
                continue
        
        # Normal token append
        input_ids = torch.cat([
            input_ids, 
            torch.tensor([[next_token_id]], device=model.device)
        ], dim=1)
        generated_tokens += 1
        
        # Progress indicator for long generations
        if verbose and generated_tokens % 200 == 0:
            print(f"  [Generated {generated_tokens} tokens...]")
        
        # Periodic garbage collection to prevent memory fragmentation
        if generated_tokens % 100 == 0:
            clear_gpu_memory()
    
    # Decode
    full_output = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=False)
    
    # Extract YES/NO answer
    answer = extract_answer(full_output)
    
    # Clean up GPU memory before returning
    del input_ids
    clear_gpu_memory()
    
    return GenerationResult(
        full_output=full_output,
        answer=answer,
        token_count=generated_tokens,
        think_end_positions=think_end_positions,
        condition=condition,
    )


def generate_response(
    model,
    tokenizer,
    prompt: str,
    condition: str,
    gen_config: GenerationConfig = None,
    verbose: bool = False,
    use_nnsight: bool = False
) -> GenerationResult:
    """
    Generate response under specified condition.
    
    Args:
        model: The language model (nnsight LanguageModel)
        tokenizer: The tokenizer
        prompt: Formatted input prompt
        condition: One of "normal", "extended_1x", "extended_2x"
        gen_config: Generation parameters (uses defaults if None)
        verbose: Print debug info
        use_nnsight: Use slow nnsight loop (for activation analysis), default False
        
    Returns:
        GenerationResult with full output, extracted answer, and metadata
    """
    if use_nnsight:
        return generate_response_nnsight(model, tokenizer, prompt, condition, gen_config, verbose)
    else:
        return generate_response_chunked(model, tokenizer, prompt, condition, gen_config, verbose)


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    condition: str,
    gen_config: GenerationConfig = None,
    verbose: bool = False
) -> List[GenerationResult]:
    """
    Generate responses for multiple prompts in parallel (batched).
    
    For 'normal' condition: Full batching with KV cache.
    For 'extended_*' conditions: Falls back to sequential (interception logic is per-sequence).
    
    Args:
        model: The language model (nnsight LanguageModel)
        tokenizer: The tokenizer
        prompts: List of formatted input prompts
        condition: One of "normal", "extended_1x", "extended_2x", etc.
        gen_config: Generation parameters
        verbose: Print debug info
        
    Returns:
        List of GenerationResult objects
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
    settings = CONDITION_SETTINGS[condition]
    intercept_count = settings["intercept_count"]
    
    # For extended conditions, fall back to sequential (interception is per-sequence)
    if intercept_count > 0:
        results = []
        for p in tqdm(prompts, desc=f"  {condition}", unit="sample", leave=False):
            results.append(generate_response_chunked(model, tokenizer, p, condition, gen_config, verbose))
        return results
    
    # For normal condition: true batched generation
    max_tokens = settings["max_tokens"]
    
    # Tokenize all prompts with padding
    tokenizer.padding_side = "left"  # Left-pad for generation
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True,
        truncation=True
    ).to(model.device)
    
    prompt_lengths = [
        (inputs.attention_mask[i] == 1).sum().item() 
        for i in range(len(prompts))
    ]
    
    hf_model = model._model
    
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
            use_cache=True,  # Explicitly enable KV cache
        )
    
    results = []
    for i, seq in enumerate(outputs.sequences):
        # Find where actual content starts (after padding)
        prompt_len = prompt_lengths[i]
        pad_len = inputs.input_ids.shape[1] - prompt_len
        
        # Decode only the generated part
        generated_ids = seq[inputs.input_ids.shape[1]:]
        full_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # Find </think> positions in generated output
        think_end_positions = []
        for j, tok_id in enumerate(generated_ids.tolist()):
            if tok_id == THINK_END_ID:
                think_end_positions.append(j)
        
        answer = extract_answer(full_output)
        
        results.append(GenerationResult(
            full_output=full_output,
            answer=answer,
            token_count=len(generated_ids),
            think_end_positions=think_end_positions,
            condition=condition,
        ))
    
    # Cleanup
    del outputs, inputs
    clear_gpu_memory()
    
    if verbose:
        print(f"  [Batched {len(prompts)} prompts]")
    
    return results

