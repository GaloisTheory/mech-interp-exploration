#!/usr/bin/env python3
"""
Utility functions for logit lens analysis with TransformerLens models.

This module provides functions to analyze how model predictions evolve
across transformer layers using the logit lens technique.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional


def get_top5_by_layer(
    model,
    cache: Dict[str, torch.Tensor],
    n_layers: int,
    token_position: int = -1
) -> List[Dict]:
    """
    Compute top-5 predicted tokens at each layer using logit lens.
    
    Args:
        model: HookedTransformer model instance
        cache: Cache dictionary from model.run_with_cache()
        n_layers: Number of layers in the model
        token_position: Position in sequence to analyze (default: -1 for final token)
    
    Returns:
        List of dictionaries, one per layer, containing:
            - 'layer': layer index
            - 'top5_tokens': list of top 5 token strings
            - 'top5_probs': list of top 5 probabilities
            - 'top1_prob': probability of top token
    """
    logit_lens_data = []
    
    for layer_idx in range(n_layers):
        # Get residual stream at this layer's output (after attention + MLP)
        resid_key = f"blocks.{layer_idx}.hook_resid_post"
        
        if resid_key not in cache:
            print(f"⚠️  Layer {layer_idx}: Key '{resid_key}' not found, skipping")
            continue
        
        # Get residual at specified token position: shape [hidden_dim]
        resid = cache[resid_key][0, token_position, :]  # batch=0, token_pos, all hidden dims
        
        # Apply layer norm (model.ln_final) then unembedding (model.unembed)
        # This mimics what the model does at the final layer
        resid_normed = model.ln_final(resid)
        logits = model.unembed(resid_normed.unsqueeze(0).unsqueeze(0))  # [1, 1, vocab_size]
        logits = logits.squeeze()  # [vocab_size]
        
        # Get probabilities
        probs = F.softmax(logits.float(), dim=-1)
        
        # Get top-5 tokens
        top5_probs, top5_indices = torch.topk(probs, 5)
        
        # Convert to strings
        top5_tokens = []
        for idx in top5_indices:
            token_str = model.to_string(idx.unsqueeze(0))
            # Clean up the string for display
            token_str = repr(token_str)[1:-1]  # Show escape chars clearly
            top5_tokens.append(token_str)
        
        logit_lens_data.append({
            'layer': layer_idx,
            'top5_tokens': top5_tokens,
            'top5_probs': top5_probs.tolist(),
            'top1_prob': top5_probs[0].item(),
        })
    
    return logit_lens_data


def display_top5_table(logit_lens_data: List[Dict]):
    """
    Display a formatted table of top-5 tokens by layer.
    
    Args:
        logit_lens_data: Output from get_top5_by_layer()
    """
    print("Layer │ Top-1 Prob │ Top-5 Predicted Tokens")
    print("──────┼────────────┼" + "─" * 60)
    
    for data in logit_lens_data:
        layer = data['layer']
        top1_prob = data['top1_prob']
        tokens_with_probs = []
        for tok, prob in zip(data['top5_tokens'], data['top5_probs']):
            # Truncate long tokens for display
            tok_display = tok[:15] + "..." if len(tok) > 18 else tok
            tokens_with_probs.append(f"{tok_display}")
        
        tokens_str = " | ".join(tokens_with_probs)
        print(f"{layer:5d} │ {top1_prob:10.4f} │ {tokens_str}")


def display_probability_by_layer(
    model,
    cache: Dict[str, torch.Tensor],
    n_layers: int,
    prompt: str,
    target_token_id: Optional[int] = None,
    token_position: int = -1
):
    """
    Display a visualization of how a token's probability evolves across layers.
    
    If target_token_id is None, uses the top prediction from the final layer.
    
    Args:
        model: HookedTransformer model instance
        cache: Cache dictionary from model.run_with_cache()
        n_layers: Number of layers in the model
        prompt: The prompt that was used (for display purposes)
        target_token_id: Token ID to track (None = use final layer's top prediction)
        token_position: Position in sequence to analyze (default: -1 for final token)
    """
    print("\n" + "="*80)
    print("LOGIT LENS VISUALIZATION: Token Probability Evolution")
    print("="*80)
    
    # Get the target token to track
    if target_token_id is None:
        # Get the top prediction from the final layer to track
        final_resid = cache[f"blocks.{n_layers-1}.hook_resid_post"][0, token_position, :]
        final_normed = model.ln_final(final_resid)
        final_logits = model.unembed(final_normed.unsqueeze(0).unsqueeze(0)).squeeze()
        target_token_id = torch.argmax(final_logits).item()
    
    target_token_str = model.to_string(torch.tensor([target_token_id]))
    print(f"Prompt: '{prompt}'")
    print(f"Tracking token: '{target_token_str}' (token_id={target_token_id})")
    
    # Get probability of target token at each layer
    target_probs = []
    for layer_idx in range(n_layers):
        resid_key = f"blocks.{layer_idx}.hook_resid_post"
        if resid_key not in cache:
            target_probs.append(0.0)
            continue
        
        resid = cache[resid_key][0, token_position, :]
        resid_normed = model.ln_final(resid)
        logits = model.unembed(resid_normed.unsqueeze(0).unsqueeze(0)).squeeze()
        probs = F.softmax(logits.float(), dim=-1)
        target_probs.append(probs[target_token_id].item())
    
    # ASCII bar chart with dynamic scaling
    print(f"\nProbability of '{target_token_str.strip()}' at each layer:")
    print()
    max_bar_width = 50
    max_prob_in_data = max(target_probs) if target_probs else 1.0
    # Use log scale indicator for better visibility at low probs
    use_linear_scale = max_prob_in_data > 0.1
    
    if use_linear_scale:
        print(f"(Linear scale, max={max_prob_in_data:.3f})")
    else:
        print(f"(Values shown as-is, max={max_prob_in_data:.4f})")
    print()
    
    for layer_idx, prob in enumerate(target_probs):
        # Scale bar width based on max probability in data
        if max_prob_in_data > 0:
            bar_width = int((prob / max_prob_in_data) * max_bar_width)
        else:
            bar_width = 0
        bar = "█" * bar_width + "░" * (max_bar_width - bar_width)
        
        # Highlight key layers
        if layer_idx < 5:
            layer_label = f"L{layer_idx:2d} (early)"
        elif layer_idx >= n_layers - 5:
            layer_label = f"L{layer_idx:2d} (late) "
        else:
            layer_label = f"L{layer_idx:2d}       "
        
        print(f"{layer_label} │{bar}│ {prob:.4f}")
    
    # Summary analysis
    print("\n" + "-"*80)
    early_avg = sum(target_probs[:5]) / 5 if len(target_probs) >= 5 else 0
    late_avg = sum(target_probs[-5:]) / 5 if len(target_probs) >= 5 else 0
    max_prob = max(target_probs)
    max_layer = target_probs.index(max_prob)
    
    print(f"Token tracked: '{target_token_str}' (top prediction at final layer)")
    print(f"Early layers (0-4) average prob: {early_avg:.4f}")
    print(f"Late layers ({n_layers-5}-{n_layers-1}) average prob: {late_avg:.4f}")
    print(f"Max probability: {max_prob:.4f} at layer {max_layer}")
    
    # Analysis of the pattern
    if max_prob > 0.5:
        print("✅ HEALTHY: Strong prediction emerges (>50% at peak)")
        if max_layer > n_layers // 2:
            print("✅ Peak occurs in later half of network (expected)")
        else:
            print("⚠️  Peak occurs early - unusual pattern")
    elif max_prob > 0.1:
        print("⚠️  MODERATE: Prediction reaches moderate confidence")
    else:
        print("⚠️  WEAK: Token never gets high probability")
    
    if late_avg > early_avg * 2:
        print("✅ Late layers have higher prob than early (expected)")
    elif early_avg > late_avg * 2 and early_avg > 0.01:
        print("⚠️  Early layers have higher prob than late (unusual)")
    
    return target_probs, target_token_id, target_token_str


