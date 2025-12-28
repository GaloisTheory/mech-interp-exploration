#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------
# 0. SPEED & PERSISTENCE CONFIG
# ---------------------------------------------------------
export XDG_CACHE_HOME="/workspace/.cache"
export UV_CACHE_DIR="/workspace/.cache/uv"
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_CACHE="/workspace/.cache/huggingface/hub"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$HF_HOME"

echo "🚀 Starting Master Setup (uv-based)..."

# Vendored repo path (used later)
VENDORED_REPO="/workspace/third_party/thinking-llms-interp"

# ---------------------------------------------------------
# 1. SYSTEM DEPS (Node.js + basics)
# ---------------------------------------------------------
echo "🔧 Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
  curl ca-certificates git gh \
  python3 python3-venv \
  build-essential

# Node.js 20.x (needed for CircuitsVis/PySvelte workflows)
if ! command -v node &>/dev/null || [[ "$(node -v)" != v20* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y --no-install-recommends nodejs
fi

# ---------------------------------------------------------
# 2. INSTALL uv
# ---------------------------------------------------------
echo "⚡ Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH for this script session (works for root + non-root)
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null

echo "uv: $(uv --version)"

# ---------------------------------------------------------
# 3. PYTHON ENV SETUP (NO VENV; install into system python)
# ---------------------------------------------------------
PYTHON_BIN="$(command -v python3)"
echo "Using python: $PYTHON_BIN"
echo "Using uv cache: $UV_CACHE_DIR"

# Install packages from pyproject.toml
# --break-system-packages: needed for PEP 668 compliant systems (Ubuntu 24.04+)
echo "🐍 Installing Python packages from pyproject.toml..."
cd /workspace
uv pip install --system --break-system-packages --python "$PYTHON_BIN" -r pyproject.toml

# ---------------------------------------------------------
# 4. VENDORED REPO SETUP (thinking-llms-interp)
# ---------------------------------------------------------
if [[ -d "$VENDORED_REPO" ]]; then
    echo "📦 Setting up vendored repo: thinking-llms-interp..."
    
    # Install as editable (no deps - we already installed them above)
    # This adds the repo to sys.path so imports work
    cd "$VENDORED_REPO"
    uv pip install --system --break-system-packages --python "$PYTHON_BIN" --no-deps -e .
    
    echo "✅ thinking-llms-interp installed as editable package"
else
    echo "⚠️  Vendored repo not found at $VENDORED_REPO"
    echo "   Clone it with: git clone <repo-url> $VENDORED_REPO"
fi

cd /workspace

# ---------------------------------------------------------
# 5. REGISTER A JUPYTER KERNEL FOR THIS PYTHON (NO VENV)
# ---------------------------------------------------------
echo "🧠 Registering Jupyter kernel for this Python..."
KERNEL_NAME="python3-system"
DISPLAY_NAME="Python 3 (system: $(basename "$PYTHON_BIN"))"

"$PYTHON_BIN" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

echo "✅ Kernel registered: $DISPLAY_NAME (name=$KERNEL_NAME)"
echo "   In Jupyter: Kernel -> Change Kernel -> \"$DISPLAY_NAME\""
echo "--------------------------------------------------"

# ---------------------------------------------------------
# 6. VERIFICATION
# ---------------------------------------------------------
echo "✅ Setup Complete!"
echo "--------------------------------------------------"

"$PYTHON_BIN" - <<'PY'
import sys
print("Python:", sys.executable)
try:
    import torch
    print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check skipped / failed:", repr(e))
PY

echo "Node.js: $(node -v)"
echo "uv: $(uv --version)"
echo "--------------------------------------------------"

"$PYTHON_BIN" - <<'PY'
import sys
print(f"Python for installs/checks: {sys.executable}")

def check(name, import_name=None):
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        print(f"✅ {name} v{version}")
    except Exception as e:
        print(f"❌ {name} import failed: {repr(e)}")

# Core mech interp
check("transformer_lens", "transformer_lens")
check("nnsight")
check("circuitsvis")
check("sae_lens", "sae_lens")

# ML stack
check("datasets")
check("einops")

# thinking-llms-interp deps
check("anthropic")
check("nltk")
check("dotenv", "dotenv")

# Utilities
check("plotly")
check("pandas")
PY

echo "--------------------------------------------------"
echo "🎉 All done."

# ---------------------------------------------------------
# 7. SECRETS & AUTHENTICATION
# ---------------------------------------------------------
SECRETS_FILE="/workspace/.secrets"
if [[ -f "$SECRETS_FILE" ]]; then
    echo "🔐 Loading secrets..."
    source "$SECRETS_FILE"
    
    # GitHub auth + git credential setup
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        git config --global user.name "Dohun Lee"
        git config --global user.email "d.lee2176@gmail.com"
        
        # Try gh CLI first (preferred - handles token refresh, org access, etc.)
        # Falls back to direct credentials if gh fails (e.g., token missing read:org scope)
        gh_success=false
        if command -v gh &>/dev/null; then
            if echo "$GITHUB_TOKEN" | gh auth login --with-token 2>/dev/null; then
                gh auth setup-git 2>/dev/null
                gh_success=true
                echo "✅ GitHub authenticated via gh CLI"
            fi
        fi
        
        # Fallback: Direct git credential store (works with any token that has repo scope)
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
