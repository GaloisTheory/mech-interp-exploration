#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------
# 0. SPEED & PERSISTENCE CONFIG
# ---------------------------------------------------------
export XDG_CACHE_HOME="/workspace/.cache"
export UV_CACHE_DIR="/workspace/.cache/uv"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR"

echo "🚀 Starting Master Setup (uv-based) for RTX 6000..."

# ---------------------------------------------------------
# 1. SYSTEM DEPS (Node.js + basics)
# ---------------------------------------------------------
echo "🔧 Installing system dependencies..."
apt-get update -qq
apt-get install -y --no-install-recommends \
  curl ca-certificates git \
  python3 python3-venv python3-distutils \
  build-essential

# Node.js 20.x (needed for CircuitsVis/PySvelte workflows)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y --no-install-recommends nodejs

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

# Install core stack via uv into *system* site-packages
# --system: target the interpreter’s system site-packages (no venv)
# --python: explicitly choose the interpreter
echo "🐍 Installing Python packages with uv..."
uv pip install --system --python "$PYTHON_BIN" -U \
  "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
  transformer-lens circuitsvis plotly fancy_einsum pandas tqdm ipykernel jupyterlab

# Pin transformers (force reinstall like your pip version)
uv pip install --system --python "$PYTHON_BIN" -U --force-reinstall "transformers==4.38.0"

# ---------------------------------------------------------
# 4. REGISTER A JUPYTER KERNEL FOR THIS PYTHON (NO VENV)
# ---------------------------------------------------------
echo "🧠 Registering Jupyter kernel for this Python..."
KERNEL_NAME="python3-system"
DISPLAY_NAME="Python 3 (system: $(basename "$PYTHON_BIN"))"

"$PYTHON_BIN" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

echo "✅ Kernel registered: $DISPLAY_NAME (name=$KERNEL_NAME)"
echo "   In Jupyter: Kernel -> Change Kernel -> \"$DISPLAY_NAME\""
echo "--------------------------------------------------"

# ---------------------------------------------------------
# 5. VERIFICATION
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
        __import__(import_name)
        print(f"✅ {name} import OK")
    except Exception as e:
        print(f"❌ {name} import failed: {repr(e)}")

check("transformer_lens", "transformer_lens")
check("circuitsvis")
check("plotly")
check("fancy_einsum")
check("pandas")
check("tqdm")
PY

echo "--------------------------------------------------"
echo "🎉 All done."

