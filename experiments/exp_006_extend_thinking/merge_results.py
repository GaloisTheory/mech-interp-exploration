#!/usr/bin/env python3
"""Merge sharded experiment results into a single file.

Usage:
    python merge_results.py my_exp
    
This will find all files matching 'my_exp_shard*' in outputs/ and merge them
into 'my_exp_merged.json'.
"""

import argparse
import json
import os
import sys
from glob import glob
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation import compare_conditions, PairResult


def find_shard_files(experiment_name: str, output_dir: str) -> list:
    """Find all shard files for an experiment."""
    pattern = os.path.join(output_dir, f"{experiment_name}_shard*_*.json")
    files = glob(pattern)
    
    # Sort by shard number
    def get_shard_num(f):
        basename = os.path.basename(f)
        # Extract shard number from 'name_shard1of3_timestamp.json'
        try:
            shard_part = basename.split("_shard")[1].split("of")[0]
            return int(shard_part)
        except:
            return 0
    
    return sorted(files, key=get_shard_num)


def merge_results(shard_files: list) -> dict:
    """Merge multiple shard result files."""
    if not shard_files:
        raise ValueError("No shard files to merge")
    
    # Load all shards
    shards = []
    for f in shard_files:
        with open(f) as fp:
            shards.append(json.load(fp))
        print(f"  Loaded: {os.path.basename(f)}")
    
    # Use first shard as base
    merged = shards[0].copy()
    
    # Update config
    total_pairs = sum(s["config"]["n_pairs"] for s in shards)
    merged["config"]["n_pairs"] = total_pairs
    merged["config"]["shard"] = None  # No longer sharded
    merged["config"]["merged_from"] = [os.path.basename(f) for f in shard_files]
    merged["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Merge by_pair data
    conditions = merged["config"]["conditions"]
    merged["by_pair"] = {c: [] for c in conditions}
    
    for shard in shards:
        for condition in conditions:
            if condition in shard.get("by_pair", {}):
                merged["by_pair"][condition].extend(shard["by_pair"][condition])
    
    # Recompute summary statistics
    print("\n  Recomputing statistics...")
    
    # Convert back to PairResult objects for comparison
    results_by_condition = {}
    for condition in conditions:
        results_by_condition[condition] = []
        for pair_data in merged["by_pair"][condition]:
            # Create a minimal PairResult-like object
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
    
    # Compute comparison
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
    
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python merge_results.py my_exp
  
  This finds outputs/my_exp_shard1of3_*.json, my_exp_shard2of3_*.json, etc.
  and merges them into outputs/my_exp_merged.json
        """
    )
    
    parser.add_argument(
        "experiment_name",
        help="Name of the experiment to merge"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory containing shard files (default: ./outputs)"
    )
    
    args = parser.parse_args()
    
    print(f"Merging results for: {args.experiment_name}")
    print(f"Looking in: {args.output_dir}")
    print()
    
    # Find shard files
    shard_files = find_shard_files(args.experiment_name, args.output_dir)
    
    if not shard_files:
        print(f"ERROR: No shard files found matching '{args.experiment_name}_shard*'")
        print(f"  Expected pattern: {args.experiment_name}_shard1of3_*.json")
        sys.exit(1)
    
    print(f"Found {len(shard_files)} shard files:")
    
    # Merge
    merged = merge_results(shard_files)
    
    # Save
    output_file = os.path.join(args.output_dir, f"{args.experiment_name}_merged.json")
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n✓ Merged results saved to: {output_file}")
    print(f"  Total pairs: {merged['config']['n_pairs']}")
    print(f"  Conditions: {merged['config']['conditions']}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 40)
    for condition, stats in merged["summary"].items():
        print(f"  {condition}: IPHR rate = {stats['iphr_rate']:.1%} ({stats['n_unfaithful']}/{stats['n_total']})")


if __name__ == "__main__":
    main()

