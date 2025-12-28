#%% Imports & Setup
"""
Interactive Hybrid Model Runner

This script wraps the thinking-llms-interp hybrid model for interactive use.
The hybrid model combines:
- A THINKING model (e.g., DeepSeek-R1-Distill-Qwen-1.5B) to detect reasoning categories
- A BASE model (e.g., Qwen2.5-Math-1.5B) that gets steered to reason like the thinking model

Key insight from the paper: Base models already have reasoning capabilities,
thinking models just learn WHEN to use them. This hybrid approach recovers
up to 91% of the thinking model's performance gap.

Setup:
    1. Run setup_mech_interp_uv.sh (installs thinking-llms-interp as editable package)
    2. Select kernel: python3-system
    3. Run cells interactively
"""
import os
import sys

# Set HF cache before any imports
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

# Change to hybrid directory so relative paths work (for SAE/vector files)
VENDORED_REPO = "/workspace/third_party/thinking-llms-interp"
os.chdir(f"{VENDORED_REPO}/hybrid")

# Add hybrid/ to path for interactive_hybrid_sentence.py
sys.path.insert(0, f"{VENDORED_REPO}/hybrid")

#%% Configuration - Choose your model pair
"""
Available model pairs (pre-trained steering vectors exist):

| Base Model               | Thinking Model                           | VRAM   | SAE Layer | Steer Layer |
|--------------------------|------------------------------------------|--------|-----------|-------------|
| Qwen/Qwen2.5-Math-1.5B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B| ~6GB   | 4         | 10          | <- DEFAULT
| meta-llama/Llama-3.1-8B  | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | ~32GB  | 6         | 12          |
| Qwen/Qwen2.5-14B         | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | ~56GB  | 38        | 18          |
| Qwen/Qwen2.5-32B         | Qwen/QwQ-32B                             | ~128GB | 27        | 24          |
"""

# Using 1.5B models for faster loading (~6GB VRAM total)
THINKING_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"

# SAE and steering config (from paper's settings for Qwen-1.5B)
SAE_LAYER = 4        # Layer to extract SAE activations from thinking model
STEERING_LAYER = 10  # Layer to apply steering vectors in base model (~37% of 28 layers)
N_CLUSTERS = 15      # Number of reasoning categories (matches available vectors)

# Generation settings
DATASET = "gsm8k"    # "gsm8k", "math500", or "aime"
EXAMPLE_IDX = 0      # Which problem to solve
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.0    # 0 = greedy, >0 = sampling

#%% Load dependencies
print("Loading dependencies...")
import torch
from datasets import load_dataset

# Import from vendored repo
from utils.sae import load_sae
from utils.utils import load_model
from utils.clustering import get_latent_descriptions
from utils.utils import load_steering_vectors as _load_all_steering_vectors

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

#%% Load dataset and select problem
print(f"\nLoading dataset: {DATASET}")
if DATASET == "gsm8k":
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    question = dataset[EXAMPLE_IDX]["question"]
    answer = dataset[EXAMPLE_IDX]["answer"]
elif DATASET == "math500":
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    question = dataset[EXAMPLE_IDX]["problem"]
    answer = dataset[EXAMPLE_IDX]["answer"]
else:  # aime
    dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    question = dataset[EXAMPLE_IDX]["problem"]
    answer = dataset[EXAMPLE_IDX]["answer"]

print("\n" + "=" * 60)
print("PROBLEM")
print("=" * 60)
print(question)
print("\n" + "=" * 60)
print("REFERENCE ANSWER")
print("=" * 60)
print(answer)

#%% Load models
print("\n" + "=" * 60)
print("LOADING MODELS (this takes a few minutes)")
print("=" * 60)

print(f"\nLoading thinking model: {THINKING_MODEL}")
thinking_model, thinking_tok = load_model(model_name=THINKING_MODEL)
thinking_model.tokenizer = thinking_tok

print(f"\nLoading base model: {BASE_MODEL}")
base_model, base_tok = load_model(model_name=BASE_MODEL)

print("\nBoth models loaded!")

#%% Load SAE and steering vectors
print("\n" + "=" * 60)
print("LOADING SAE AND STEERING VECTORS")
print("=" * 60)

# Extract model IDs for file paths
thinker_id = THINKING_MODEL.split("/")[-1].lower()
base_id = BASE_MODEL.split("/")[-1].lower()

# Load SAE
print(f"\nLoading SAE for {thinker_id}, layer {SAE_LAYER}, {N_CLUSTERS} clusters...")
sae, sae_checkpoint = load_sae(thinker_id, SAE_LAYER, N_CLUSTERS)
sae = sae.to(thinking_model.device)

# Load steering vectors
print(f"\nLoading steering vectors for {base_id}...")
hyperparams_dir = "../train-vectors/results/vars/hyperparams"
vectors_dir = "../train-vectors/results/vars/optimized_vectors"

all_vectors = _load_all_steering_vectors(
    hyperparams_dir=hyperparams_dir,
    vectors_dir=vectors_dir,
    verbose=False,
)

# Get latent descriptions (category names)
descriptions = get_latent_descriptions(thinker_id, SAE_LAYER, N_CLUSTERS)

# Map steering vectors to category keys
steering_vectors = {}
for desc in descriptions.values():
    latent_key = desc.get("key", "")
    if latent_key:
        slug = latent_key.lower().replace(" ", "-")
        if slug in all_vectors:
            steering_vectors[latent_key] = all_vectors[slug]

print(f"\nLoaded {len(steering_vectors)} category-specific steering vectors")
print("\nReasoning categories discovered by SAE:")
for idx, desc in descriptions.items():
    has_vector = "✓" if desc.get("key", "") in steering_vectors else "✗"
    print(f"  [{has_vector}] {idx}: {desc['title']}")

#%% Run interactive hybrid generation
"""
This is the main loop from the paper!

At each step:
1. Thinking model generates a sentence
2. SAE detects which reasoning category is active
3. You choose whether to steer the base model with that category's vector
4. Compare steered vs unsteered base model output

Press Ctrl+C to stop at any time.
"""
print("\n" + "=" * 60)
print("INTERACTIVE HYBRID GENERATION")
print("=" * 60)
print("\nStarting generation...")
print("At each sentence, you'll see:")
print("  - [Thinking] What the thinking model generates")
print("  - Top-3 reasoning categories detected")
print("  - [Base] What the base model generates (steered or not)")
print("\nTip: Press Enter to accept default coefficient, 'k' to keep unsteered")

# Import the generation functions
from interactive_hybrid_sentence import (
    thinking_generate_sentence,
    base_generate_sentence,
    compute_sentence_activation,
)

# Prepare initial prompts
thinking_ids = thinking_tok.apply_chat_template(
    [{"role": "user", "content": question}],
    add_generation_prompt=True,
    return_tensors="pt",
).to(thinking_model.device).to(torch.long)

base_prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n"
base_ids = base_tok.encode(base_prompt, return_tensors="pt").to(base_model.device).to(torch.long)

# Generation loop
import gc
generated_tokens = 0
default_coeff = 0.5

while generated_tokens < MAX_NEW_TOKENS:
    try:
        # 1. Thinking model generates one sentence
        sent_think, latent_acts, thinking_ids = thinking_generate_sentence(
            thinking_model,
            thinking_tok,
            thinking_ids,
            SAE_LAYER,
            sae,
            temperature=TEMPERATURE,
        )
        print(f"\n[Thinking] {sent_think}")
        
        # 2. Detect top reasoning categories
        top_vals, top_ids_tensor = torch.topk(latent_acts, k=min(3, latent_acts.shape[0]))
        top_ids = top_ids_tensor.tolist()
        top_vals = top_vals.tolist()
        
        print("\nTop 3 reasoning categories:")
        for rank, (lid, val) in enumerate(zip(top_ids, top_vals), 1):
            title = descriptions[lid]["title"]
            print(f"  {rank}. {title} (activation={val:.3f})")
        
        # Get primary category's steering vector
        latent_id = top_ids[0]
        latent_key = descriptions[latent_id].get("key", "")
        steering_vec = steering_vectors.get(latent_key)
        
        # 3. Generate unsteered base sentence
        base_ids_pre = base_ids.clone()
        sent_unsteered, base_ids_unsteered = base_generate_sentence(
            base_model,
            base_tok,
            base_ids_pre,
            steering_vector=None,
            coefficient=0.0,
            steering_layer=STEERING_LAYER,
            temperature=TEMPERATURE,
        )
        print(f"\n[Base • unsteered] {sent_unsteered}")
        
        # 4. Compare activations
        unsteered_latent_acts = compute_sentence_activation(
            thinking_model,
            thinking_tok,
            SAE_LAYER,
            sae,
            thinking_ids,
            sent_unsteered,
        )
        
        print("\nCategory comparison (thinking vs base unsteered):")
        diff_vals = []
        for rank, lid in enumerate(top_ids):
            think_val = top_vals[rank]
            base_val = unsteered_latent_acts[lid].item()
            diff_val = think_val - base_val
            diff_vals.append(diff_val)
            title = descriptions[lid]["title"]
            arrow = "↑" if diff_val > 0.1 else "↓" if diff_val < -0.1 else "≈"
            print(f"  {title}: think={think_val:.2f}, base={base_val:.2f}, Δ={diff_val:+.2f} {arrow}")
        
        # 5. Find best steering vector to apply
        best_idx = None
        best_abs = 0.0
        best_vec = None
        for idx, (lid, delta_val) in enumerate(zip(top_ids, diff_vals)):
            vec = steering_vectors.get(descriptions[lid].get("key", ""))
            if vec is None:
                continue
            if abs(delta_val) > best_abs:
                best_abs = abs(delta_val)
                best_idx = idx
                best_vec = vec
        
        # Default: keep unsteered
        chosen_coeff = 0.0
        sent_base = sent_unsteered
        base_ids_final = base_ids_unsteered
        
        if best_vec is not None:
            target_title = descriptions[top_ids[best_idx]]["title"]
            suggested_coeff = diff_vals[best_idx]
            
            user_in = input(f"\nSteer '{target_title}'? (Enter={suggested_coeff:+.2f}, k=keep, number=custom): ").strip().lower()
            
            if user_in not in ("k", "keep"):
                if user_in == "":
                    coeff_val = suggested_coeff
                else:
                    try:
                        coeff_val = float(user_in)
                    except ValueError:
                        print("Invalid input, keeping unsteered")
                        coeff_val = None
                
                if coeff_val is not None:
                    # Generate steered sentence
                    sent_steered, base_ids_steered = base_generate_sentence(
                        base_model,
                        base_tok,
                        base_ids.clone(),
                        best_vec.to(base_model.device),
                        coeff_val,
                        STEERING_LAYER,
                        temperature=TEMPERATURE,
                    )
                    print(f"\n[Base • steered coeff={coeff_val:+.2f}] {sent_steered}")
                    
                    accept = input("Accept steered? (y/n): ").strip().lower()
                    if accept in ("y", "yes", ""):
                        chosen_coeff = coeff_val
                        sent_base = sent_steered
                        base_ids_final = base_ids_steered
        
        # Update state
        base_ids = base_ids_final
        generated_tokens += len(base_tok.encode(sent_base))
        
        # Keep thinking model in sync
        think_append = thinking_tok.encode(
            sent_base,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(thinking_model.device).to(torch.long)
        thinking_ids = torch.cat([thinking_ids, think_append], dim=1)
        
        # Check for EOS
        if base_ids[0, -1].item() == base_tok.eos_token_id:
            print("\n<EOS reached>")
            break
        
        # Continue?
        cont = input("\nContinue? (Enter=yes, n=stop): ").strip().lower()
        if cont == "n":
            break
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        break

#%% Show final answer
print("\n" + "=" * 60)
print("FINAL HYBRID ANSWER")
print("=" * 60)
# Decode just the generated portion
base_prompt_len = len(base_tok.encode(base_prompt))
final_answer = base_tok.decode(base_ids[0][base_prompt_len:], skip_special_tokens=True)
print(final_answer)

print("\n" + "=" * 60)
print("REFERENCE ANSWER")
print("=" * 60)
print(answer)

# %%

