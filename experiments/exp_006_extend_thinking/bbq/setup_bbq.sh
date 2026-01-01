#!/bin/bash
# Minimal setup for playground_bbq.py
# Usage: bash setup_bbq.sh

set -e

echo "🔧 Minimal BBQ Playground Setup"
echo "================================"

# Use persistent cache
export PIP_CACHE_DIR="/workspace/.cache/pip"
export HF_HOME="/workspace/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

# Install core dependencies only
echo "📦 Installing PyTorch 2.8 + transformers..."
pip install -q --upgrade pip

# PyTorch 2.8 with CUDA 12.4 (compatible with Blackwell)
pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cu124

# Transformers + dependencies
pip install -q transformers accelerate datasets

# Jupyter kernel (use the same Python)
pip install -q ipykernel
python -m ipykernel install --user --name bbq --display-name "BBQ Playground"

echo ""
echo "✅ Setup complete!"
echo ""
echo "=== Verification ==="
python -c "
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')
"

echo ""
echo "=== Next steps ==="
echo "1. In Jupyter: Select kernel 'BBQ Playground'"
echo "2. Or run directly: python playground_bbq.py"
echo "3. Or interactive: python -i playground_interactive.py"











