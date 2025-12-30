"""Generation logic for normal and extended thinking conditions."""

import gc
import torch
import torch.nn.functional as F
import re
from typing import List
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


def extract_answer(text: str) -> str:
    """Extract YES or NO from model output.
    
    Uses word boundary matching to avoid false positives like 
    'NO' in 'NORTH' or 'YES' in 'YESTERDAY'.
    """
    
    text_upper = text.upper()
    
    if "</THINK>" in text_upper:
        after_think = text_upper.split("</THINK>")[-1]
    else:
        after_think = text_upper
    
    after_think = after_think.strip()
    
    # Use word boundaries to find standalone YES/NO
    # \b matches word boundaries (start/end of word)
    yes_matches = list(re.finditer(r'\bYES\b', after_think))
    no_matches = list(re.finditer(r'\bNO\b', after_think))
    
    has_yes = len(yes_matches) > 0
    has_no = len(no_matches) > 0
    
    if has_yes and not has_no:
        return "YES"
    elif has_no and not has_yes:
        return "NO"
    elif has_yes and has_no:
        # Find the last occurrence of each
        yes_pos = yes_matches[-1].start()
        no_pos = no_matches[-1].start()
        return "YES" if yes_pos > no_pos else "NO"
    else:
        return "UNCLEAR"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    condition: str,
    gen_config: GenerationConfig = None,
    verbose: bool = False
) -> GenerationResult:
    """
    Generate response with proper O(n) KV caching for extended conditions.
    
    Uses token-by-token generation with persistent KV cache to avoid
    recomputing attention after each intercept.
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
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
        if next_token_id == THINK_END_ID:
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
                # This maintains O(n) because we're processing just continuation tokens
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
        
        if verbose and len(generated_ids) % 200 == 0:
            print(f"  [Generated {len(generated_ids)} tokens...]")
    
    # Decode
    full_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
    answer = extract_answer(full_output)
    
    return GenerationResult(
        full_output=full_output,
        answer=answer,
        token_count=len(generated_ids),
        think_end_positions=think_end_positions,
        condition=condition,
    )


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    condition: str,
    gen_config: GenerationConfig = None,
    verbose: bool = False
) -> List[GenerationResult]:
    """
    Generate responses for multiple prompts.
    
    For 'normal' condition: Full batching with HF generate().
    For 'extended_*' conditions: Sequential with O(n) KV caching per sequence.
    """
    if gen_config is None:
        gen_config = GenerationConfig()
    
    settings = CONDITION_SETTINGS[condition]
    intercept_count = settings["intercept_count"]
    
    # Get underlying HuggingFace model
    hf_model = getattr(model, '_model', model)
    
    # For extended conditions, run sequentially with KV cache
    if intercept_count > 0:
        results = []
        for p in tqdm(prompts, desc=f"  {condition}", unit="sample", leave=False):
            results.append(generate_response(model, tokenizer, p, condition, gen_config, verbose))
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
    
    del outputs, inputs
    clear_gpu_memory()
    
    if verbose:
        print(f"  [Batched {len(prompts)} prompts]")
    
    return results
