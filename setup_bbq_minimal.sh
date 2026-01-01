#!/bin/bash
set -euo pipefail

# Minimal setup for running playground_bbq.py as an interactive Jupyter notebook
# Uses uv with persistent cache - packages already cached, fast reinstall after GPU switch
# Does NOT reinstall torch (pre-installed on RunPod with CUDA)

echo "🚀 Minimal BBQ Playground Setup"

# ---------------------------------------------------------
# 1. PERSISTENT CACHE (packages already cached here)
# ---------------------------------------------------------
export XDG_CACHE_HOME="/workspace/.cache"
export UV_CACHE_DIR="/workspace/.cache/uv"
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"

# ---------------------------------------------------------
# 2. INSTALL UV (if not present)
# ---------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "⚡ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv: $(uv --version)"
echo "Cache: $UV_CACHE_DIR"

# ---------------------------------------------------------
# 3. INSTALL DEPENDENCIES (from cache, skipping torch)
# ---------------------------------------------------------
echo "📦 Installing Python packages (torch is pre-installed)..."
PYTHON_BIN="$(command -v python3)"

uv pip install --system --break-system-packages --python "$PYTHON_BIN" \
    "transformers>=4.47.0" \
    datasets \
    tqdm \
    "accelerate>=1.10.0" \
    ipykernel \
    transformer_lens

# ---------------------------------------------------------
# 4. REGISTER JUPYTER KERNEL
# ---------------------------------------------------------
echo "🧠 Registering Jupyter kernel..."
KERNEL_NAME="python3-bbq"
DISPLAY_NAME="Python 3 (BBQ)"

"$PYTHON_BIN" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

echo "✅ Kernel registered: $DISPLAY_NAME"

# ---------------------------------------------------------
# 5. SECRETS & AUTHENTICATION
# ---------------------------------------------------------
SECRETS_FILE="/workspace/.secrets"
if [[ -f "$SECRETS_FILE" ]]; then
    echo "🔐 Loading secrets..."
    source "$SECRETS_FILE"
    
    # GitHub auth + git credential setup
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        git config --global user.name "Dohun Lee"
        git config --global user.email "d.lee2176@gmail.com"
        
        # Try gh CLI first (preferred)
        gh_success=false
        if command -v gh &>/dev/null; then
            if echo "$GITHUB_TOKEN" | gh auth login --with-token 2>/dev/null; then
                gh auth setup-git 2>/dev/null
                gh_success=true
                echo "✅ GitHub authenticated via gh CLI"
            fi
        fi
        
        # Fallback: Direct git credential store
        if [[ "$gh_success" == "false" ]]; then
            git config --global credential.helper store
            echo "https://${GITHUB_TOKEN}:x-oauth-basic@github.com" > ~/.git-credentials
            chmod 600 ~/.git-credentials
            echo "✅ GitHub authenticated via git credentials"
        fi
    fi
    
    # HuggingFace auth
    if [[ -n "${HF_TOKEN:-}" ]]; then
        huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
        echo "✅ HuggingFace authenticated"
    fi
else
    echo "⚠️  No secrets file found at $SECRETS_FILE — skipping auth"
fi

# ---------------------------------------------------------
# 6. VERIFY
# ---------------------------------------------------------
echo ""
echo "✅ Setup Complete!"
echo "--------------------------------------------------"
python3 -c "
import sys
print(f'Python: {sys.executable}')
import torch
print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
import transformers
print(f'Transformers: {transformers.__version__}')
import transformer_lens
print(f'TransformerLens: installed')
"
echo "--------------------------------------------------"
echo "Run: cd /workspace/experiments/exp_006_extend_thinking/bbq"
echo "Then open playground_bbq.py or playground_bbq.ipynb"
echo "Select kernel: 'Python 3 (BBQ)'"
