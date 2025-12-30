"""Shared utilities for IPHR experiment - used by both CLI and interactive notebook."""

import os
import sys
import json
import warnings
from datetime import datetime
from glob import glob
from typing import Dict, List, Optional, Tuple

# Set cache before imports
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

# Suppress HuggingFace compile warnings
warnings.filterwarnings("ignore", message=".*compile_config.*")
warnings.filterwarnings("ignore", message=".*Compilation will be skipped.*")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_NAME, GenerationConfig
from data.question_pairs import get_question_pairs, format_prompt
from generation import generate_batch
from evaluation import PairResult, compare_conditions


def load_model():
    """Load HuggingFace model and tokenizer.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {MODEL_NAME}")
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {gpu_mem:.1f} GB")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        print("No GPU available, using CPU (will be slow)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Model loaded!")
    return model, tokenizer


def parse_shard(shard_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse shard string like '1/3' into tuple (1, 3).
    
    Args:
        shard_str: String like "1/3" or None
        
    Returns:
        Tuple (shard_id, total_shards) or None
    """
    if shard_str is None:
        return None
    
    parts = shard_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid shard format: {shard_str}. Expected 'X/N' like '1/3'")
    
    return (int(parts[0]), int(parts[1]))


def load_pairs(
    shard: Optional[Tuple[int, int]] = None,
    test_mode: bool = False,
    test_n: int = 3,
    max_pairs: Optional[int] = None
) -> Tuple[List, Optional[Tuple[int, int]]]:
    """Load question pairs, optionally sharded.
    
    Args:
        shard: Tuple (shard_id, total_shards) or None for all pairs
        test_mode: If True, only load test_n pairs
        test_n: Number of pairs in test mode
        max_pairs: Maximum number of pairs to load (overrides test_mode/test_n if set)
        
    Returns:
        Tuple of (pairs list, shard_info)
    """
    if max_pairs is not None:
        n_pairs = max_pairs
    else:
        n_pairs = test_n if test_mode else None
    all_pairs = get_question_pairs(n_pairs)
    
    if shard is not None:
        shard_id, total_shards = shard
        shard_size = len(all_pairs) // total_shards
        remainder = len(all_pairs) % total_shards
        
        # Distribute remainder across first shards
        start = (shard_id - 1) * shard_size + min(shard_id - 1, remainder)
        end = start + shard_size + (1 if shard_id <= remainder else 0)
        
        pairs = all_pairs[start:end]
        print(f"SHARD {shard_id}/{total_shards}: Processing pairs {start}-{end-1} ({len(pairs)} pairs)")
    else:
        pairs = all_pairs
    
    print(f"Loaded {len(pairs)} question pairs")
    return pairs, shard


def run_experiment(
    model,
    tokenizer,
    pairs: List,
    conditions: List[str],
    samples_per_question: int,
    verbose: bool = False,
    save_raw: bool = False,
    experiment_name: Optional[str] = None,
    shard: Optional[Tuple[int, int]] = None,
    test_mode: bool = False
) -> Dict[str, List[PairResult]]:
    """Run the IPHR experiment loop.
    
    Args:
        model: The HuggingFace model
        tokenizer: The tokenizer
        pairs: List of question pairs
        conditions: List of condition names to test
        samples_per_question: Number of samples per question
        verbose: Print detailed output
        save_raw: Save full model outputs
        experiment_name: Name for incremental saves (if None, no incremental saves)
        shard: Shard info for incremental saves
        test_mode: Test mode flag for incremental saves
        
    Returns:
        Dict mapping condition -> list of PairResult
    """
    print("=" * 70)
    print("RUNNING IPHR EXPERIMENT")
    print("=" * 70)
    print(f"Total: {len(pairs)} pairs × {len(conditions)} conditions = {len(pairs) * len(conditions)} batches")
    if experiment_name:
        print(f"Incremental saves: ENABLED (after each pair)")
    print()
    
    gen_config = GenerationConfig()
    results_by_condition: Dict[str, List[PairResult]] = {c: [] for c in conditions}
    
    # Progress bar for pairs
    pair_pbar = tqdm(pairs, desc="Question pairs", unit="pair")
    
    for pair_idx, (q1, q2, q1_exp, q2_exp, category) in enumerate(pair_pbar):
        pair_pbar.set_postfix({"category": category[:15]})
        
        for condition in conditions:
            # Batch ALL samples for Q1 and Q2 together
            prompt1 = format_prompt(q1)
            prompt2 = format_prompt(q2)
            
            all_prompts = [prompt1] * samples_per_question + [prompt2] * samples_per_question
            all_results = generate_batch(
                model, tokenizer, all_prompts, condition,
                gen_config=gen_config, verbose=verbose
            )
            
            # Split results back into Q1 and Q2
            q1_results = all_results[:samples_per_question]
            q2_results = all_results[samples_per_question:]
            
            q1_answers = [r.answer for r in q1_results]
            q1_outputs = [r.full_output for r in q1_results]
            q2_answers = [r.answer for r in q2_results]
            q2_outputs = [r.full_output for r in q2_results]
            
            if verbose:
                for i in range(samples_per_question):
                    tqdm.write(f"    Sample {i + 1}: Q1={q1_answers[i]} ({q1_results[i].token_count}tok), Q2={q2_answers[i]} ({q2_results[i].token_count}tok)")
            
            # Create result
            # Always store full outputs (reasoning chains) for readability
            result = PairResult(
                q1_text=q1,
                q2_text=q2,
                q1_answers=q1_answers,
                q2_answers=q2_answers,
                q1_expected=q1_exp,
                q2_expected=q2_exp,
                category=category,
                condition=condition,
                q1_outputs=q1_outputs,  # Always include reasoning chains
                q2_outputs=q2_outputs,  # Always include reasoning chains
            )
            results_by_condition[condition].append(result)
        
        # Save incremental checkpoint after each pair
        if experiment_name is not None:
            save_results(
                results_by_condition=results_by_condition,
                experiment_name=experiment_name,
                conditions=conditions,
                samples_per_question=samples_per_question,
                shard=shard,
                test_mode=test_mode,
                save_raw=save_raw,
                incremental=True
            )
    
    pair_pbar.close()
    print("\n" + "=" * 70)
    print("Experiment complete!")
    
    return results_by_condition


def save_results(
    results_by_condition: Dict[str, List[PairResult]],
    experiment_name: str,
    conditions: List[str],
    samples_per_question: int,
    shard: Optional[Tuple[int, int]] = None,
    test_mode: bool = False,
    save_raw: bool = False,
    incremental: bool = False
) -> str:
    """Save experiment results and config to JSON files.
    
    Args:
        results_by_condition: Dict mapping condition -> list of PairResult
        experiment_name: Name for output files
        conditions: List of conditions that were run
        samples_per_question: Number of samples per question
        shard: Shard info tuple or None
        test_mode: Whether this was a test run
        save_raw: Whether raw outputs are included
        incremental: If True, this is an incremental save (quieter output)
        
    Returns:
        Path to saved results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Determine experiment subfolder
    # For sharded runs or incremental saves, check if a folder for this experiment already exists (created recently)
    # This allows multiple shards to use the same folder and incremental saves to reuse the same folder
    experiment_folder = None
    if shard is not None or incremental:
        # Look for existing folders with this experiment name created in the last 24 hours
        pattern = os.path.join(base_output_dir, f"{experiment_name}_*")
        existing_folders = glob(pattern)
        
        # Filter to only directories created in the last 24 hours
        current_time = datetime.now()
        for folder_path in existing_folders:
            if os.path.isdir(folder_path):
                try:
                    # Try to parse timestamp from folder name
                    folder_name = os.path.basename(folder_path)
                    if folder_name.startswith(f"{experiment_name}_"):
                        folder_timestamp_str = folder_name[len(f"{experiment_name}_"):]
                        folder_time = datetime.strptime(folder_timestamp_str, "%Y%m%d_%H%M%S")
                        time_diff = (current_time - folder_time).total_seconds()
                        # If created within last 24 hours, reuse it
                        if time_diff < 86400:
                            experiment_folder = folder_path
                            break
                except (ValueError, IndexError):
                    continue
    
    # If no existing folder found, create a new one
    if experiment_folder is None:
        experiment_folder = os.path.join(base_output_dir, f"{experiment_name}_{timestamp}")
    
    # Create the experiment folder
    os.makedirs(experiment_folder, exist_ok=True)
    output_dir = experiment_folder
    
    # Compute comparison metrics
    comparison = compare_conditions(results_by_condition)
    
    # Count pairs
    n_pairs = len(next(iter(results_by_condition.values()))) if results_by_condition else 0
    
    # Prepare config
    config_data = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "model": MODEL_NAME,
        "conditions": conditions,
        "samples_per_question": samples_per_question,
        "test_mode": test_mode,
        "n_pairs": n_pairs,
        "shard": shard,
    }
    
    # Prepare serializable output
    output_data = {
        "timestamp": timestamp,
        "config": config_data,
        "summary": {},
        "deltas": {},
        "by_pair": {},
    }
    
    # Extract summary metrics
    for condition in conditions:
        metrics = comparison.get(condition, {})
        output_data["summary"][condition] = {
            "iphr_rate": metrics.get("iphr_rate", 0),
            "n_unfaithful": metrics.get("n_unfaithful", 0),
            "n_total": metrics.get("n_total", 0),
            "by_type": metrics.get("by_type", {}),
            "avg_accuracy": metrics.get("avg_accuracy", 0),
        }
    
    # Extract deltas
    for key, value in comparison.items():
        if key.startswith("delta_"):
            output_data["deltas"][key] = value
    
    # Add per-pair details
    for condition, results in results_by_condition.items():
        output_data["by_pair"][condition] = []
        for pair_idx, r in enumerate(results):
            # Structure reasoning chains by sample number for easy access
            # Format: reasoning_chains[sample_index][q1|q2] = full_output
            reasoning_chains = {}
            for sample_idx in range(len(r.q1_answers)):
                reasoning_chains[f"sample_{sample_idx + 1}"] = {
                    "q1_reasoning": r.q1_outputs[sample_idx] if sample_idx < len(r.q1_outputs) else "",
                    "q2_reasoning": r.q2_outputs[sample_idx] if sample_idx < len(r.q2_outputs) else "",
                    "q1_answer": r.q1_answers[sample_idx],
                    "q2_answer": r.q2_answers[sample_idx],
                }
            
            pair_data = {
                "pair_index": pair_idx,
                "category": r.category,
                "q1_text": r.q1_text,
                "q2_text": r.q2_text,
                "q1_answers": r.q1_answers,
                "q2_answers": r.q2_answers,
                "q1_expected": r.q1_expected,
                "q2_expected": r.q2_expected,
                "q1_yes_rate": r.q1_yes_rate,
                "q2_yes_rate": r.q2_yes_rate,
                "is_unfaithful": r.is_unfaithful,
                "unfaithfulness_type": r.unfaithfulness_type,
                "reasoning_chains": reasoning_chains,  # Always include reasoning chains
            }
            output_data["by_pair"][condition].append(pair_data)
    
    # Build filename
    shard_str = f"_shard{shard[0]}of{shard[1]}" if shard else ""
    output_file = os.path.join(output_dir, f"{experiment_name}{shard_str}_{timestamp}.json")
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Save config separately
    config_file = os.path.join(output_dir, f"{experiment_name}_config.json")
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    
    if not incremental:
        print(f"\nResults saved to: {output_file}")
        print(f"Config saved to: {config_file}")
        print(f"Experiment folder: {experiment_folder}")
    
    return output_file


def print_results_summary(results_by_condition: Dict[str, List[PairResult]], conditions: List[str]):
    """Print a summary of results to console."""
    from evaluation import compare_conditions, format_results_summary
    
    comparison = compare_conditions(results_by_condition)
    print(format_results_summary(comparison))
    
    print("\n" + "=" * 70)
    print("DETAILED PER-PAIR RESULTS")
    print("=" * 70)
    
    for condition in conditions:
        print(f"\n{condition.upper()}:")
        print("-" * 50)
        
        for result in results_by_condition.get(condition, []):
            status = "❌ UNFAITHFUL" if result.is_unfaithful else "✓ Faithful"
            print(f"  [{result.category}] {status}")
            print(f"    Q1 answers: {result.q1_answers} → YES rate: {result.q1_yes_rate:.0%}")
            print(f"    Q2 answers: {result.q2_answers} → YES rate: {result.q2_yes_rate:.0%}")

