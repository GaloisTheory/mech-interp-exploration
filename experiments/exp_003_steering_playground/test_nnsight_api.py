# %% Test nnsight API
from experiments.config import *
import torch
from nnsight import LanguageModel

# Load model
print('Loading model...')
model = LanguageModel(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    dispatch=True,
    device_map='cuda',
    dtype=torch.bfloat16  # Updated from torch_dtype
)

tokenizer = model.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id

# %% Test trace (forward pass only)
print('\n--- Test 1: Simple trace ---')
test_input = 'Hello'
with model.trace(test_input) as tracer:
    out = model.model.layers[0].output[0].save()
print('Trace output shape:', out.shape)
print('SUCCESS with trace!')

# %% Test trace with intervention
print('\n--- Test 2: Trace with intervention ---')
steering_vector = torch.randn(1536, device='cuda', dtype=torch.bfloat16)

with model.trace(test_input) as tracer:
    # Intervention: add vector to layer output
    model.model.layers[10].output[0][:, :] += steering_vector
    out = model.model.layers[10].output[0].save()
print('Intervention output shape:', out.shape)
print('SUCCESS with intervention!')

# %% Test generation with trace
print('\n--- Test 3: Generation with intervention ---')
messages = [{"role": "user", "content": "What is 2+2?"}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Try model.generate() method directly (not the nnsight trace)
print('Trying model.generate()...')
print('Type of model.generate:', type(model.generate))

# Based on nnsight 0.5 - use the Generator class or generate method
# Check if we can use generate with a hook-based approach instead
try:
    # First try: Direct generate without intervention
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=20)
    print('Direct generate output shape:', output_ids.shape)
    print('Generated:', tokenizer.decode(output_ids[0], skip_special_tokens=True))
except Exception as e:
    print(f'Direct generate failed: {e}')

# %% Test 4: Generation with intervention using scan=True
print('\n--- Test 4: Generation with scan=True ---')
try:
    with model.trace(input_ids, scan=True, validate=False) as tracer:
        # Apply intervention during forward
        model.model.layers[10].output[0][:, :] += 0.1 * steering_vector
        logits = model.lm_head.output.save()
    print('Scan trace logits shape:', logits.shape)
except Exception as e:
    print(f'Scan trace failed: {e}')

# %% Test 5: Generation with PyTorch hooks on underlying HF model
print('\n--- Test 5: Generation with PyTorch hooks on HF model ---')

def make_steering_hook(steering_vector, coefficient=1.0):
    """Create a forward hook that adds steering vector to layer output."""
    def hook(module, input, output):
        # For Qwen decoder layers, output is a tuple: (hidden_states, present_key_value, ...)
        # or just hidden_states depending on config
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Add steering vector to all positions
            modified = hidden_states + coefficient * steering_vector.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            return output + coefficient * steering_vector.unsqueeze(0).unsqueeze(0)
    return hook

# Get the underlying HF model directly
hf_model = model._model

# Register hook on the underlying HF model's layer
hook_handle = hf_model.model.layers[10].register_forward_hook(
    make_steering_hook(steering_vector, coefficient=0.5)
)

try:
    # Use the underlying HF model's generate directly, bypassing nnsight
    with torch.no_grad():
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        output_ids = hf_model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )
    print('Hooked generation output shape:', output_ids.shape)
    print('Generated text:', tokenizer.decode(output_ids[0], skip_special_tokens=True)[:300])
finally:
    hook_handle.remove()

print('\n✅ Hook-based generation works!')
print('\nDone with tests!')

