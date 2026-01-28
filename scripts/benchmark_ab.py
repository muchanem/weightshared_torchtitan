#!/usr/bin/env python3
"""
A/B benchmark: batched LoRA vs sequential LoRA.
Patches the code at import time to test both configurations.
"""

import subprocess
import re
import os


def run_with_config(use_batched: bool, steps: int = 25):
    """Run training with specified batched_lora setting."""

    # Create a wrapper script that patches the module
    if use_batched:
        patch_code = ""  # Default is True
    else:
        patch_code = """
# Patch SharedFeedForward to use sequential LoRA
import torchtitan.models.qwen3.model.weight_sharing as ws
_orig_init = ws.SharedFeedForward.__init__
def _patched_init(self, shared_weights, dim, hidden_dim, lora_rank, use_batched_lora=False):
    _orig_init(self, shared_weights, dim, hidden_dim, lora_rank, use_batched_lora=False)
ws.SharedFeedForward.__init__ = _patched_init
"""

    wrapper_script = f'''
{patch_code}
import sys
sys.argv = sys.argv[1:]  # Remove script name
from torchtitan.train import Trainer, main
main(Trainer)
'''

    # Write to temp file
    script_path = '/tmp/train_wrapper.py'
    with open(script_path, 'w') as f:
        f.write(wrapper_script)

    cmd = f'''WANDB_MODE=disabled .venv/bin/torchrun --nproc_per_node=1 {script_path} \
        --job.config_file torchtitan/models/qwen3/train_configs/250m_combined.toml \
        --training.steps {steps} \
        --checkpoint.no-enable \
        --parallelism.data_parallel_replicate_degree 1'''

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse TPS from output
    tps_matches = re.findall(r'tps: ([\d,]+)', output)
    if tps_matches:
        # Get the last few TPS values (after warmup, skip step 1)
        tps_values = [int(t.replace(',', '')) for t in tps_matches[1:]]  # Skip first (warmup)
        if tps_values:
            return sum(tps_values) / len(tps_values)

    print(f"  [DEBUG] Failed to parse. Last 1000 chars of output:")
    print(output[-1000:])
    return None


def main():
    print("=" * 60)
    print("A/B Benchmark: Batched LoRA vs Sequential LoRA")
    print("=" * 60)

    # Run with batched LoRA (pre-stacked weights)
    print("\nRunning with BATCHED LoRA (pre-stacked weights)...")
    tps_batched = run_with_config(use_batched=True, steps=25)
    if tps_batched:
        print(f"  TPS (batched): {tps_batched:,.0f}")

    # Run with sequential LoRA
    print("\nRunning with SEQUENTIAL LoRA...")
    tps_sequential = run_with_config(use_batched=False, steps=25)
    if tps_sequential:
        print(f"  TPS (sequential): {tps_sequential:,.0f}")

    # Compare
    if tps_batched and tps_sequential:
        speedup = tps_batched / tps_sequential
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Batched LoRA:    {tps_batched:,.0f} TPS")
        print(f"  Sequential LoRA: {tps_sequential:,.0f} TPS")
        print(f"  Speedup:         {speedup:.3f}x")
        print(f"  Improvement:     {(speedup-1)*100:+.2f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
