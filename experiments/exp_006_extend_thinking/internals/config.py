"""Configuration for logit lens analysis.

Edit this file to change experiment settings.
The script will hot-reload this on each run.
"""

# =============================================================================
# MODEL SETTINGS
# =============================================================================
MODEL_NAME = "Qwen/Qwen3-8B"

# =============================================================================
# BBQ QUESTION SELECTION
# =============================================================================
# Categories: age, disability, gender, nationality, appearance, race, religion, ses, sexual_orientation
BBQ_CATEGORY = "age"
BBQ_INDEX = 0  # Which question from the category

# =============================================================================
# FORCED COT INJECTION
# =============================================================================
# Text to inject as "unfaithful" reasoning
FORCED_COT_TEXT = "The answer must be A then! "
FORCED_COT_REPEATS = 10  # How many times to repeat

# =============================================================================
# PROMPT SETTINGS
# =============================================================================
USE_FEW_SHOT = True  # Use 10-shot examples to force A/B/C output

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
SAVE_PLOTS = True  # Save plots to outputs/ folder
SHOW_PLOTS = True  # Display plots inline (for Jupyter/interactive)

# Heatmap colormap limits (None = auto-scale)
HEATMAP_VMIN = -1.0  # P(C) - P(A) range
HEATMAP_VMAX = 1.0

