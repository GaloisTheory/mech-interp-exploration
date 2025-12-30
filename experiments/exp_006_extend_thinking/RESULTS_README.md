# IPHR Experiment Results Analyzer

A modern, interactive notebook for exploring IPHR (Implicit Post-Hoc Rationalization) experiment results.

## Quick Start

### 1. Run the analyzer

```bash
python analyze_results.py
```

By default, it will analyze the most recent experiment. The notebook runs all cells sequentially.

### 2. Customize for your experiment

Edit the configuration at the top of `analyze_results.py`:

```python
# Line ~29
EXPERIMENT_NAME = "QWQ_32B_1"  # Change this to your experiment name
USE_MERGED = False  # Set to True for merged results
```

### 3. Dive deeper into specific pairs

Modify these variables to inspect specific pairs:

```python
# Line ~261 - Inspect a specific pair in detail
INSPECT_CONDITION = "normal"  # or "extended_1x", "extended_2x", etc.
INSPECT_PAIR_INDEX = 0  # 0-based index
INSPECT_SAMPLE = 1  # Which sample's reasoning to view

# Line ~315 - Compare all conditions for a pair
COMPARE_PAIR_INDEX = 0  # Which pair to compare
```

## What the Notebook Shows

The analyzer provides several views:

### 1. **Summary Statistics** 
- IPHR rates for each condition
- YES-YES vs NO-NO breakdown
- Changes relative to normal condition

### 2. **Per-Pair Breakdown**
- All pairs across all conditions
- Faithfulness status for each
- Answer distributions

### 3. **Detailed Inspection**
- Full question text
- All answers for both questions
- Faithfulness determination

### 4. **Reasoning Chain Viewer**
- Complete model reasoning for any sample
- Shows how the model arrived at its answer
- Useful for understanding failures

### 5. **Condition Comparison**
- Side-by-side comparison for a specific pair
- Shows how extended thinking affects each pair

### 6. **Summary Export**
- Automatically exports a text summary
- Saved in the same folder as your results

## Example Output

```
================================================================================
SUMMARY STATISTICS
================================================================================

NORMAL:
  IPHR Rate: 40.0% (2/5 pairs unfaithful)
    YES-YES pairs: 0
    NO-NO pairs: 2
  Avg YES rates: Q1=60.0%, Q2=0.0%

EXTENDED_5X:
  IPHR Rate: 0.0% (0/5 pairs unfaithful)
    YES-YES pairs: 0
    NO-NO pairs: 0
  Avg YES rates: Q1=100.0%, Q2=4.0%

CHANGES vs NORMAL:
  extended_5x: -40.0% ⬇️
```

## Running in Interactive Mode

The file is designed to work with VS Code / Cursor's interactive Python mode:

1. Open `analyze_results.py` in VS Code/Cursor
2. Look for the `#%%` cell markers
3. Click "Run Cell" above each marker
4. Modify variables and re-run specific cells

## Available Experiments

The notebook automatically lists all available experiments in the output directory. Look for the "AVAILABLE EXPERIMENTS" section when you run it.

## Tips

- **Quick check**: Just run `python analyze_results.py` to see summary
- **Deep dive**: Open in VS Code and run cells interactively
- **Compare runs**: Change `EXPERIMENT_NAME` and re-run
- **Export**: Summary is auto-exported to `{experiment}_summary.txt`

## Files Generated

- `{experiment}_summary.txt` - Text summary of results
- Merged files are in the same folder as the original results


