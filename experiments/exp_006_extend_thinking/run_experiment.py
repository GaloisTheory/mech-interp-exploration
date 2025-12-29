#!/usr/bin/env python3
"""IPHR Experiment Runner - Clean CLI interface.

Usage:
    # Run all pairs:
    python run_experiment.py --name my_exp
    
    # Run with sharding (3 terminals):
    python run_experiment.py --name my_exp --shard 1/3
    python run_experiment.py --name my_exp --shard 2/3
    python run_experiment.py --name my_exp --shard 3/3
    
    # Quick test:
    python run_experiment.py --name test_run --test
    
    # Custom settings:
    python run_experiment.py --name my_exp --samples 5 --conditions normal extended_1x
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_utils import (
    load_model,
    load_pairs,
    run_experiment,
    save_results,
    print_results_summary,
    parse_shard,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run IPHR faithfulness experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --name baseline
  python run_experiment.py --name baseline --shard 1/3
  python run_experiment.py --name quick_test --test --verbose
        """
    )
    
    parser.add_argument(
        "--name", 
        required=True,
        help="Experiment name (used for output filenames)"
    )
    parser.add_argument(
        "--shard",
        help="Shard spec for parallel runs, e.g. '1/3', '2/3', '3/3'"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5, 
        help="Samples per question (default: 3)"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["normal", "extended_1x", "extended_2x", "extended_5x"],
        help="Conditions to test (default: normal extended_1x extended_2x extended_5x)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode (3 pairs only)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-sample output"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save full model outputs (large files)"
    )
    
    args = parser.parse_args()
    
    # Parse shard
    shard = parse_shard(args.shard)
    
    # Print config
    print("=" * 70)
    print("IPHR EXPERIMENT")
    print("=" * 70)
    print(f"  Experiment:  {args.name}")
    print(f"  Conditions:  {args.conditions}")
    print(f"  Samples/Q:   {args.samples}")
    print(f"  Shard:       {args.shard if args.shard else 'None (all pairs)'}")
    print(f"  Test mode:   {args.test}")
    print(f"  Verbose:     {args.verbose}")
    print()
    
    # Load model
    model, tokenizer = load_model()
    
    # Load pairs
    pairs, shard_info = load_pairs(shard=shard, test_mode=args.test)
    
    # Run experiment
    results = run_experiment(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs,
        conditions=args.conditions,
        samples_per_question=args.samples,
        verbose=args.verbose,
        save_raw=args.save_raw,
    )
    
    # Print summary
    print_results_summary(results, args.conditions)
    
    # Save results
    save_results(
        results_by_condition=results,
        experiment_name=args.name,
        conditions=args.conditions,
        samples_per_question=args.samples,
        shard=shard_info,
        test_mode=args.test,
        save_raw=args.save_raw,
    )


if __name__ == "__main__":
    main()

