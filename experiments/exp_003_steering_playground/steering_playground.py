# %% [markdown]
# # Steering Vector Playground
# 
# Based on the official Colab demo from cvenhoff/steering-thinking-llms.
# Uses DeepSeek-R1-Distill-Qwen-1.5B with pre-trained mean-difference steering vectors.
#
# Note: Updated for nnsight 0.5.x API - uses PyTorch hooks for generation interventions.

# %% Imports and Configuration
from experiments.config import *  # Sets HF_HOME before any HF imports

import torch
import os
from contextlib import contextmanager
from typing import Dict, List, Optional
from nnsight import LanguageModel

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
VECTOR_PATH = "/workspace/third_party/steering-thinking-llms/train-steering-vectors/results/vars/mean_vectors_deepseek-r1-distill-qwen-1.5b.pt"

# Steering configuration: each category has optimal layer(s) for intervention
# Layer choices from the Colab demo; additional categories use similar mid-to-late layers
STEERING_CONFIG = {
    "backtracking": {
        "vector_layer": 17,
        "pos_layers": [17],
        "neg_layers": [17],
    },
    "uncertainty-estimation": {
        "vector_layer": 18,
        "pos_layers": [18],
        "neg_layers": [18],
    },
    "example-testing": {
        "vector_layer": 15,
        "pos_layers": [15],
        "neg_layers": [15],
    },
    "adding-knowledge": {
        "vector_layer": 18,
        "pos_layers": [18],
        "neg_layers": [18],
    },
    # Additional categories available in the vectors
    "initializing": {
        "vector_layer": 14,
        "pos_layers": [14],
        "neg_layers": [14],
    },
    "deduction": {
        "vector_layer": 16,
        "pos_layers": [16],
        "neg_layers": [16],
    },
    "summarizing": {
        "vector_layer": 18,
        "pos_layers": [18],
        "neg_layers": [18],
    },
}


# %% Steering Hook Utilities
def make_steering_hook(steering_vector: torch.Tensor, coefficient: float = 1.0):
    """
    Create a forward hook that adds steering vector to layer output.
    
    Args:
        steering_vector: The steering vector to add (shape: [hidden_dim])
        coefficient: Multiplier for the steering vector
    
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # For Qwen decoder layers, output is a tuple: (hidden_states, present_key_value, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Add steering vector to all positions: [batch, seq, hidden] + [1, 1, hidden]
            modified = hidden_states + coefficient * steering_vector.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            return output + coefficient * steering_vector.unsqueeze(0).unsqueeze(0)
    return hook


@contextmanager
def steering_context(hf_model, layer_indices: List[int], steering_vector: torch.Tensor, coefficient: float = 1.0):
    """
    Context manager that applies steering hooks to specified layers.
    
    Args:
        hf_model: The underlying HuggingFace model
        layer_indices: List of layer indices to apply steering to
        steering_vector: The steering vector to add
        coefficient: Multiplier for the steering vector
    """
    hooks = []
    try:
        for layer_idx in layer_indices:
            hook_handle = hf_model.model.layers[layer_idx].register_forward_hook(
                make_steering_hook(steering_vector, coefficient)
            )
            hooks.append(hook_handle)
        yield
    finally:
        for hook in hooks:
            hook.remove()


# %% Load Model and Vectors
def load_model_and_vectors():
    """Load the model, tokenizer, and pre-trained steering vectors."""
    print(f"Loading model {MODEL_NAME}...")

    # Load model with nnsight
    model = LanguageModel(
        MODEL_NAME,
        dispatch=True,
        device_map="cuda",
        dtype=torch.bfloat16
    )

    # Configure generation settings (greedy decoding)
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.do_sample = False

    tokenizer = model.tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load pre-trained vectors
    print(f"Loading steering vectors from {VECTOR_PATH}...")
    if os.path.exists(VECTOR_PATH):
        mean_vectors_dict = torch.load(VECTOR_PATH, weights_only=False)

        # Compute feature vectors by subtracting overall mean
        feature_vectors = {}
        feature_vectors["overall"] = mean_vectors_dict["overall"]['mean']

        for label in ["initializing", "deduction", "adding-knowledge", 
                      "example-testing", "uncertainty-estimation", "backtracking", "summarizing"]:
            if label in mean_vectors_dict:
                feature_vectors[label] = mean_vectors_dict[label]['mean'] - mean_vectors_dict["overall"]['mean']

                # Normalize feature vectors by overall mean norm
                for layer in range(len(feature_vectors["overall"])):
                    overall_norm = feature_vectors["overall"][layer].norm()
                    label_norm = feature_vectors[label][layer].norm()
                    if label_norm > 0:
                        feature_vectors[label][layer] = feature_vectors[label][layer] * (overall_norm / label_norm)

        print("✅ Successfully loaded steering vectors!")
        print(f"   Available categories: {[k for k in feature_vectors.keys() if k != 'overall']}")
        return model, tokenizer, feature_vectors
    else:
        print(f"❌ Vector file not found at {VECTOR_PATH}")
        return model, tokenizer, {}


# %% Generation with Steering
def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 500,
    steering_label: Optional[str] = None,
    feature_vectors: Optional[Dict] = None,
    steer_positive: bool = True,
    coefficient: float = 1.0
):
    """
    Generate text with optional steering vector intervention.

    Args:
        model: The nnsight LanguageModel
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        steering_label: Which reasoning pattern to steer towards/away from
        feature_vectors: Pre-trained feature vectors dict
        steer_positive: If True, steer towards the pattern; if False, steer away
        coefficient: Strength of the steering intervention
    
    Returns:
        Tuple of (full_response, thinking_process, final_answer)
    """
    # Format prompt for chat
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # Get the underlying HuggingFace model for hook-based steering
    hf_model = model._model

    # Determine steering configuration
    if steering_label and feature_vectors and steering_label in feature_vectors:
        config = STEERING_CONFIG[steering_label]
        vector_layer = config["vector_layer"]
        layers = config["pos_layers"] if steer_positive else config["neg_layers"]
        
        # Get the feature vector for the target layer
        feature_vector = feature_vectors[steering_label][vector_layer].to("cuda").to(torch.bfloat16)
        
        # Apply positive or negative steering
        actual_coefficient = coefficient if steer_positive else -coefficient
        
        # Generate with steering hooks
        with steering_context(hf_model, layers, feature_vector, actual_coefficient):
            with torch.no_grad():
                output_ids = hf_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id
                )
    else:
        # Generate without steering
        with torch.no_grad():
            output_ids = hf_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

    # Decode the generated text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract thinking process if present (DeepSeek format)
    if "<think>" in response and "</think>" in response:
        think_start = response.index("<think>") + len("<think>")
        think_end = response.index("</think>")
        thinking_process = response[think_start:think_end].strip()
        final_answer = response[think_end + len("</think>"):].strip()
        return response, thinking_process, final_answer
    else:
        return response, "", response


# %% Compare Generations
def compare_generations(
    prompt: str, 
    model, 
    tokenizer, 
    feature_vectors, 
    coefficient: float = 0.5, 
    steering_label: str = "backtracking",
    max_new_tokens: int = 500
):
    """
    Generate and compare responses with different steering settings.
    
    Args:
        prompt: Input prompt
        model: The language model
        tokenizer: The tokenizer
        feature_vectors: Pre-trained feature vectors
        coefficient: Strength of the steering intervention
        steering_label: Which steering vector to use
        max_new_tokens: Max tokens to generate
    
    Returns:
        Dict with baseline, positive, and negative steering results
    """
    print("=" * 80)
    print(f"🎯 Prompt: {prompt}")
    print(f"🔧 Steering vector: {steering_label} (coefficient: {coefficient})")
    print("=" * 80)

    # Generate baseline (no steering)
    print("\n📝 BASELINE (No steering):")
    print("-" * 40)
    baseline_response, baseline_thinking, baseline_answer = generate_with_steering(
        model, tokenizer, prompt, 
        max_new_tokens=max_new_tokens,
        steering_label=None, 
        coefficient=coefficient
    )
    if baseline_thinking:
        print(f"Thinking: {baseline_thinking[:500]}..." if len(baseline_thinking) > 500 else f"Thinking: {baseline_thinking}")
    print(f"\nAnswer: {baseline_answer}")

    # Generate with positive steering
    print(f"\n🔄 POSITIVE STEERING (Encourage {steering_label}):")
    print("-" * 40)
    pos_response, pos_thinking, pos_answer = generate_with_steering(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        steering_label=steering_label,
        feature_vectors=feature_vectors, 
        steer_positive=True, 
        coefficient=coefficient
    )
    if pos_thinking:
        print(f"Thinking: {pos_thinking[:500]}..." if len(pos_thinking) > 500 else f"Thinking: {pos_thinking}")
    print(f"\nAnswer: {pos_answer}")

    # Generate with negative steering
    print(f"\n🚫 NEGATIVE STEERING (Discourage {steering_label}):")
    print("-" * 40)
    neg_response, neg_thinking, neg_answer = generate_with_steering(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        steering_label=steering_label,
        feature_vectors=feature_vectors, 
        steer_positive=False, 
        coefficient=coefficient
    )
    if neg_thinking:
        print(f"Thinking: {neg_thinking[:500]}..." if len(neg_thinking) > 500 else f"Thinking: {neg_thinking}")
    print(f"\nAnswer: {neg_answer}")
    print("=" * 80)

    return {
        "baseline": {"thinking": baseline_thinking, "answer": baseline_answer},
        f"positive_{steering_label}": {"thinking": pos_thinking, "answer": pos_answer},
        f"negative_{steering_label}": {"thinking": neg_thinking, "answer": neg_answer}
    }


# %% Load Model
print("Loading model and vectors...")
model, tokenizer, feature_vectors = load_model_and_vectors()

# %% Example 1: Backtracking on Math Problem
math_prompt = "hello"
# """A train leaves station A at 2:00 PM traveling at 60 mph. Another train leaves station B at 3:00 PM traveling at 80 mph towards station A. The distance between stations A and B is 300 miles. At what time will the two trains meet?"""

results_math = compare_generations(
    math_prompt, 
    model, tokenizer, feature_vectors, 
    coefficient=1, 
    steering_label="summarizing"
)

# %%
math_prompt = "My name is Christopher"
# """A train leaves station A at 2:00 PM traveling at 60 mph. Another train leaves station B at 3:00 PM traveling at 80 mph towards station A. The distance between stations A and B is 300 miles. At what time will the two trains meet?"""

results_math = compare_generations(
    math_prompt, 
    model, tokenizer, feature_vectors, 
    coefficient=1, 
    steering_label="summarizing"
)
# %%

# %%
# # %% Example 2: Adding Knowledge on Creative Task
creative_prompt = "Write a nice poem about the summer!"

results_creative = compare_generations(
    creative_prompt, 
    model, tokenizer, feature_vectors, 
    coefficient=1, 
    steering_label="adding-knowledge"
)

# %%
for coef in [0.3, 0.5, 0.7, 1.0]:
    print(f"\n=== Coefficient {coef} ===")
    results = compare_generations(
        "What is 15% of 80?",
        model, tokenizer, feature_vectors,
        coefficient=coef,
        steering_label="summarizing",
        max_new_tokens=200
    )
# %%

# %% Interactive Demo
def interactive_demo():
    """Interactive demo where you can try your own prompts."""
    print("\n" + "=" * 80)
    print("🎮 INTERACTIVE STEERING DEMO")
    print("=" * 80)
    print("Try different prompts and see how steering vectors affect the reasoning!")
    print("\nAvailable steering vectors:")
    print("  1. backtracking           - Encourage reconsidering/revising approach")
    print("  2. uncertainty-estimation - Express uncertainty about conclusions")
    print("  3. example-testing        - Test reasoning with concrete examples")
    print("  4. adding-knowledge       - Incorporate additional knowledge")
    print("  5. initializing           - Problem setup and initialization")
    print("  6. deduction              - Logical deduction steps")
    print("  7. summarizing            - Summarize findings")
    print("\nType 'quit' to exit.\n")

    steering_map = {
        "1": "backtracking",
        "2": "uncertainty-estimation",
        "3": "example-testing",
        "4": "adding-knowledge",
        "5": "initializing",
        "6": "deduction",
        "7": "summarizing"
    }

    while True:
        prompt = input("Enter your prompt (or 'quit'): ").strip()
        if prompt.lower() == 'quit':
            print("Goodbye!")
            break
        if not prompt:
            continue

        choice = input("Select steering vector [1-7]: ").strip()
        steering_label = steering_map.get(choice, "backtracking")
        if choice not in steering_map:
            print(f"Invalid choice. Using 'backtracking'.")

        coef_input = input("Steering coefficient (default 0.5): ").strip()
        try:
            coefficient = float(coef_input) if coef_input else 0.5
        except ValueError:
            print("Invalid coefficient. Using 0.5.")
            coefficient = 0.5

        print()
        compare_generations(
            prompt, model, tokenizer, feature_vectors,
            coefficient=coefficient,
            steering_label=steering_label,
            max_new_tokens=300
        )
        print()

# Uncomment to run interactive demo:

# %%
interactive_demo()
# %%
# %%
