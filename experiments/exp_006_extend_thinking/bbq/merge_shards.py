#!/usr/bin/env python3
"""Merge sharded BBQ experiment results into a single file."""

import json
import os
from glob import glob
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def merge_shards(experiment_name: str):
    """Merge all shards for an experiment into a single file.
    
    Args:
        experiment_name: Name prefix of the experiment (e.g., "full_11cat")
    """
    # Find all shard files
    pattern = os.path.join(OUTPUT_DIR, f"{experiment_name}_*", f"{experiment_name}_shard*.json")
    shard_files = sorted(glob(pattern))
    
    if not shard_files:
        print(f"❌ No shard files found matching: {pattern}")
        return None
    
    print(f"Found {len(shard_files)} shard files:")
    for f in shard_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load and merge
    merged_results = []
    merged_config = None
    
    for shard_file in shard_files:
        with open(shard_file) as f:
            data = json.load(f)
        
        # Get config from first shard
        if merged_config is None:
            merged_config = data.get('config', {})
        
        # Append results
        results = data.get('results', [])
        merged_results.extend(results)
        print(f"  Loaded {len(results)} results from {os.path.basename(shard_file)}")
    
    # Compute merged metrics
    by_condition = {}
    for r in merged_results:
        cond = r['condition']
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r)
    
    metrics = {}
    for cond, cond_results in by_condition.items():
        n_total = len(cond_results)
        n_correct = sum(1 for r in cond_results if r['is_correct'])
        n_invalid = sum(1 for r in cond_results if r['model_answer'] == 'INVALID')
        
        metrics[cond] = {
            'n_total': n_total,
            'n_correct': n_correct,
            'n_invalid': n_invalid,
            'accuracy': n_correct / n_total if n_total > 0 else 0,
            'invalid_rate': n_invalid / n_total if n_total > 0 else 0,
        }
    
    # Create merged output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_config['shard'] = None  # Clear shard info
    merged_config['n_items'] = len(merged_results) // len(by_condition) if by_condition else 0
    
    output_data = {
        'timestamp': timestamp,
        'config': merged_config,
        'metrics': metrics,
        'results': merged_results,
    }
    
    # Save merged file
    output_folder = os.path.join(OUTPUT_DIR, f"{experiment_name}_merged_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{experiment_name}_merged_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Merged {len(merged_results)} total results")
    print(f"   Output: {output_file}")
    
    # Print summary
    print("\n📊 MERGED METRICS:")
    for cond, m in metrics.items():
        print(f"   {cond}: {m['accuracy']:.1%} accuracy ({m['n_correct']}/{m['n_total']})")
    
    return output_file


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = "full_11cat"
    
    merge_shards(experiment_name)

