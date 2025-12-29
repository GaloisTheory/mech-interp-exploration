#!/usr/bin/env python3
"""Interactive Results Explorer

Use this notebook to:
- Load and explore experiment results
- Merge sharded results
- Analyze IPHR rates across conditions
- Dig into specific question pairs

Run cells interactively with #%% markers in VS Code/Cursor.
"""

#%% Imports
import os
import sys
import json
from glob import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import compare_conditions, format_results_summary, PairResult

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

print(f"Output directory: {OUTPUT_DIR}")
print("Imports complete ✓")

#%% List available result files
print("Available result files:")
print("-" * 60)

files = sorted(glob(os.path.join(OUTPUT_DIR, "*.json")), key=os.path.getmtime, reverse=True)
for i, f in enumerate(files[:20]):  # Show last 20
    basename = os.path.basename(f)
    mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
    size_kb = os.path.getsize(f) / 1024
    print(f"  [{i}] {basename} ({size_kb:.1f} KB, {mtime})")

if len(files) > 20:
    print(f"  ... and {len(files) - 20} more files")

#%% Load a specific result file
# Change the index or filename as needed
FILE_INDEX = 0  # Use index from list above
# FILE_NAME = "my_exp_merged.json"  # Or specify filename directly

if 'FILE_NAME' in dir() and FILE_NAME:
    result_file = os.path.join(OUTPUT_DIR, FILE_NAME)
else:
    result_file = files[FILE_INDEX] if files else None

if result_file and os.path.exists(result_file):
    with open(result_file) as f:
        data = json.load(f)
    print(f"Loaded: {os.path.basename(result_file)}")
    print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"  Config: {json.dumps(data.get('config', {}), indent=4)}")
else:
    print("No result file found!")
    data = None

#%% View summary statistics
if data:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for condition, stats in data.get("summary", {}).items():
        iphr = stats.get("iphr_rate", 0)
        n_unf = stats.get("n_unfaithful", 0)
        n_tot = stats.get("n_total", 0)
        acc = stats.get("avg_accuracy", 0)
        print(f"\n{condition}:")
        print(f"  IPHR rate: {iphr:.1%} ({n_unf}/{n_tot} unfaithful)")
        print(f"  Avg accuracy: {acc:.1%}")
        print(f"  By type: {stats.get('by_type', {})}")

#%% View deltas between conditions
if data and data.get("deltas"):
    print("\n" + "=" * 60)
    print("DELTAS (change from normal)")
    print("=" * 60)
    
    for key, value in data.get("deltas", {}).items():
        print(f"  {key}: {value:+.1%}")

#%% Detailed per-pair results
if data:
    print("\n" + "=" * 60)
    print("PER-PAIR RESULTS")
    print("=" * 60)
    
    for condition in data.get("config", {}).get("conditions", []):
        pairs = data.get("by_pair", {}).get(condition, [])
        n_unfaithful = sum(1 for p in pairs if p.get("is_unfaithful"))
        
        print(f"\n{condition.upper()}: {n_unfaithful}/{len(pairs)} unfaithful")
        print("-" * 40)
        
        for pair in pairs:
            status = "❌" if pair.get("is_unfaithful") else "✓"
            cat = pair.get("category", "?")
            q1_yes = pair.get("q1_yes_rate", 0)
            q2_yes = pair.get("q2_yes_rate", 0)
            print(f"  {status} [{cat}] Q1={q1_yes:.0%} Q2={q2_yes:.0%}")

#%% Inspect a specific pair
CONDITION = "extended_1x"  # Change as needed
PAIR_INDEX = 0  # Change as needed

if data:
    pairs = data.get("by_pair", {}).get(CONDITION, [])
    if PAIR_INDEX < len(pairs):
        pair = pairs[PAIR_INDEX]
        print(f"\nInspecting {CONDITION} pair {PAIR_INDEX}:")
        print("=" * 60)
        print(f"Category: {pair.get('category')}")
        print(f"\nQ1: {pair.get('q1_text')}")
        print(f"  Expected: {pair.get('q1_expected')}")
        print(f"  Answers: {pair.get('q1_answers')}")
        print(f"  YES rate: {pair.get('q1_yes_rate', 0):.0%}")
        print(f"\nQ2: {pair.get('q2_text')}")
        print(f"  Expected: {pair.get('q2_expected')}")
        print(f"  Answers: {pair.get('q2_answers')}")
        print(f"  YES rate: {pair.get('q2_yes_rate', 0):.0%}")
        print(f"\nUnfaithful: {pair.get('is_unfaithful')} ({pair.get('unfaithfulness_type')})")
    else:
        print(f"Pair index {PAIR_INDEX} out of range (max: {len(pairs)-1})")

# ============================================================================
# MERGE SHARDS
# ============================================================================

#%% Find and list shard files for an experiment
EXPERIMENT_NAME = "baseline"  # Change to your experiment name

shard_pattern = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_shard*_*.json")
shard_files = sorted(glob(shard_pattern))

print(f"Shard files for '{EXPERIMENT_NAME}':")
for f in shard_files:
    print(f"  {os.path.basename(f)}")

if not shard_files:
    print(f"  (none found matching '{EXPERIMENT_NAME}_shard*')")

#%% Merge shard files
if shard_files:
    print(f"\nMerging {len(shard_files)} shard files...")
    
    # Load all shards
    shards = []
    for f in shard_files:
        with open(f) as fp:
            shards.append(json.load(fp))
        print(f"  Loaded: {os.path.basename(f)}")
    
    # Merge
    merged = shards[0].copy()
    
    # Update config
    total_pairs = sum(s["config"]["n_pairs"] for s in shards)
    merged["config"]["n_pairs"] = total_pairs
    merged["config"]["shard"] = None
    merged["config"]["merged_from"] = [os.path.basename(f) for f in shard_files]
    merged["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Merge by_pair data
    conditions = merged["config"]["conditions"]
    merged["by_pair"] = {c: [] for c in conditions}
    
    for shard in shards:
        for condition in conditions:
            if condition in shard.get("by_pair", {}):
                merged["by_pair"][condition].extend(shard["by_pair"][condition])
    
    # Recompute summary
    print("\n  Recomputing statistics...")
    
    results_by_condition = {}
    for condition in conditions:
        results_by_condition[condition] = []
        for pair_data in merged["by_pair"][condition]:
            result = PairResult(
                q1_text=pair_data["q1_text"],
                q2_text=pair_data["q2_text"],
                q1_answers=pair_data["q1_answers"],
                q2_answers=pair_data["q2_answers"],
                q1_expected=pair_data["q1_expected"],
                q2_expected=pair_data["q2_expected"],
                category=pair_data["category"],
                condition=condition,
                q1_outputs=[],
                q2_outputs=[],
            )
            results_by_condition[condition].append(result)
    
    comparison = compare_conditions(results_by_condition)
    
    # Update summary
    merged["summary"] = {}
    for condition in conditions:
        metrics = comparison.get(condition, {})
        merged["summary"][condition] = {
            "iphr_rate": metrics.get("iphr_rate", 0),
            "n_unfaithful": metrics.get("n_unfaithful", 0),
            "n_total": metrics.get("n_total", 0),
            "by_type": metrics.get("by_type", {}),
            "avg_accuracy": metrics.get("avg_accuracy", 0),
        }
    
    # Update deltas
    merged["deltas"] = {}
    for key, value in comparison.items():
        if key.startswith("delta_"):
            merged["deltas"][key] = value
    
    print(f"\n  Total pairs after merge: {total_pairs}")
    
    # Store for next cell
    merged_data = merged
else:
    merged_data = None
    print("No shards to merge")

#%% Save merged results
if merged_data:
    output_file = os.path.join(OUTPUT_DIR, f"{EXPERIMENT_NAME}_merged.json")
    
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"✓ Saved merged results to: {output_file}")
    
    # Print summary
    print("\nMerged Summary:")
    print("-" * 40)
    for condition, stats in merged_data["summary"].items():
        print(f"  {condition}: IPHR = {stats['iphr_rate']:.1%} ({stats['n_unfaithful']}/{stats['n_total']})")

#%% Compare two experiments
# Load two result files and compare

EXP1_FILE = None  # e.g., "baseline_merged.json"
EXP2_FILE = None  # e.g., "with_steering_merged.json"

if EXP1_FILE and EXP2_FILE:
    with open(os.path.join(OUTPUT_DIR, EXP1_FILE)) as f:
        exp1 = json.load(f)
    with open(os.path.join(OUTPUT_DIR, EXP2_FILE)) as f:
        exp2 = json.load(f)
    
    print(f"Comparing: {EXP1_FILE} vs {EXP2_FILE}")
    print("=" * 60)
    
    conditions = exp1.get("config", {}).get("conditions", [])
    for condition in conditions:
        s1 = exp1.get("summary", {}).get(condition, {})
        s2 = exp2.get("summary", {}).get(condition, {})
        
        iphr1 = s1.get("iphr_rate", 0)
        iphr2 = s2.get("iphr_rate", 0)
        diff = iphr2 - iphr1
        
        print(f"\n{condition}:")
        print(f"  {EXP1_FILE}: {iphr1:.1%}")
        print(f"  {EXP2_FILE}: {iphr2:.1%}")
        print(f"  Difference: {diff:+.1%}")
else:
    print("Set EXP1_FILE and EXP2_FILE to compare experiments")

