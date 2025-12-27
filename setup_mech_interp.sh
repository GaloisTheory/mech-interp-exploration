#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------
# 0. SPEED & PERSISTENCE CONFIG
# ---------------------------------------------------------
# Force pip to save downloads in persistent storage.
export XDG_CACHE_HOME="/workspace/.cache"
mkdir -p "$XDG_CACHE_HOME"

echo "🚀 Starting Master Setup for RTX 6000..."

# ---------------------------------------------------------
# 1. VS CODE & SYSTEM FIXES (Crucial)
# ---------------------------------------------------------
echo "🔧 Applying VS Code & System Fixes..."
apt-get update -qq

# Install system pip + Node.js (needed for CircuitsVis/PySvelte)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y python3-pip nodejs

# Ensure VS Code and shells can find pip (optional but harmless)
ln -sf "$(command -v pip)"  /usr/bin/pip
ln -sf "$(command -v pip)"  /bin/pip
ln -sf "$(command -v pip)"  /usr/bin/pip3

# ---------------------------------------------------------
# 2. PYTHON ENVIRONMENT SETUP (Install into *python3*)
# ---------------------------------------------------------
echo "🐍 Upgrading pip and installing core libraries..."
PYTHON_BIN="$(command -v python)"
echo "Using python: $PYTHON_BIN"

"$PYTHON_BIN" -m pip install --upgrade pip

# Install Unsloth (Fine-tuning & Optimizations)
"$PYTHON_BIN" -m pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# Install Mech Interp & Visualization Stack
# NOTE: PyPI name is transformer-lens (hyphen), import is transformer_lens (underscore)
"$PYTHON_BIN" -m pip install -U transformer-lens circuitsvis plotly fancy_einsum pandas tqdm ipykernel jupyterlab
"$PYTHON_BIN" -m pip install -U --force-reinstall "transformers==4.38.0"

# ---------------------------------------------------------
# 3. REGISTER A JUPYTER KERNEL FOR THIS PYTHON (NO VENV)
# ---------------------------------------------------------
echo "🧠 Registering Jupyter kernel for this Python..."
"$PYTHON_BIN" -m pip install -U ipykernel

KERNEL_NAME="python3-system"
DISPLAY_NAME="Python 3 (system: $(basename "$PYTHON_BIN"))"

"$PYTHON_BIN" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

echo "✅ Kernel registered: $DISPLAY_NAME (name=$KERNEL_NAME)"
echo "   In Jupyter: Kernel -> Change Kernel -> \"$DISPLAY_NAME\""
echo "--------------------------------------------------"

# ---------------------------------------------------------
# 4. VERIFICATION
# ---------------------------------------------------------
echo "✅ Setup Complete!"
echo "--------------------------------------------------"

# Torch/GPU check (won't fail setup if torch isn't present yet)
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
echo "Pip (python3 -m pip): $($PYTHON_BIN -m pip --version)"
echo "--------------------------------------------------"

# Package import checks
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
