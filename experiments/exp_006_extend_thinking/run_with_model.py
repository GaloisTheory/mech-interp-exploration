#!/usr/bin/env python3
"""Wrapper to run experiment with a specific model by temporarily modifying config."""
import sys
import argparse

# Parse model argument first
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model name to use")
args, remaining_args = parser.parse_known_args()

# Modify config before imports
import config
config.MODEL_NAME = args.model
config.THINK_END_ID = config.get_think_end_id(args.model)

print(f"Using model: {config.MODEL_NAME}")
print(f"Think end ID: {config.THINK_END_ID}")

# Now run the main experiment with remaining args
sys.argv = [sys.argv[0]] + remaining_args
import run_experiment
run_experiment.main()




