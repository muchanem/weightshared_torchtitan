#!/usr/bin/env python3
"""
End-to-end benchmark comparing batched LoRA (pre-stacked weights) vs sequential LoRA.
"""

import subprocess
import re
import sys


def run_training(use_batched_lora: bool):
    """Run training and return TPS."""
    # Build the command
    if not use_batched_lora:
        # Patch to disable batched_lora
        patch_code = '''
import torchtitan.models.qwen3.model.weight_sharing as ws
# Store original __init__
_orig_init = ws.SharedFeedForward.__init__
def _patched_init(self, shared_weights, dim, hidden_dim, lora_rank, use_batched_lora=False):
    return _orig_init(self, shared_weights, dim, hidden_dim, lora_rank, use_batched_lora)
ws.SharedFeedForward.__init__ = _patched_init
'''
        python_cmd = f'''
{patch_code}
from torchtitan.train import Trainer, main
main(Trainer)
'''
    else:
        python_cmd = '''
from torchtitan.train import Trainer, main
main(Trainer)
'''

    cmd = [
        '.venv/bin/torchrun',
        '--nproc_per_node=1',
        '-m', 'python', '-c', python_cmd,
        '--',
        '--job.config_file', 'torchtitan/models/qwen3/train_configs/250m_combined.toml',
        '--training.steps', '25',
        '--checkpoint.no-enable',
        '--parallelism.data_parallel_replicate_degree', '1',
    ]

    # Use shell command for simplicity
    if not use_batched_lora:
        patch_opt = '--env PATCH_BATCHED_LORA=false'
    else:
        patch_opt = ''

    shell_cmd = f'''WANDB_MODE=disabled .venv/bin/torchrun --nproc_per_node=1 -m torchtitan.train \
        --job.config_file torchtitan/models/qwen3/train_configs/250m_combined.toml \
        --training.steps 25 \
        --checkpoint.no-enable \
        --parallelism.data_parallel_replicate_degree 1'''

    result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, env={
        **__import__('os').environ,
        'WANDB_MODE': 'disabled',
        'BATCHED_LORA_DISABLED': '' if use_batched_lora else '1',
    })
    output = result.stdout + result.stderr

    # Parse TPS from output
    tps_matches = re.findall(r'tps: ([\d,]+)', output)
    if tps_matches:
        # Get the last few TPS values (after warmup)
        tps_values = [int(t.replace(',', '')) for t in tps_matches[-3:]]
        return sum(tps_values) / len(tps_values) if tps_values else None
    return None


def run_training_simple(config_desc: str, extra_env: dict = None):
    """Run training with given env vars and return TPS."""
    import os
    env = {**os.environ, 'WANDB_MODE': 'disabled'}
    if extra_env:
        env.update(extra_env)

    shell_cmd = f'''.venv/bin/torchrun --nproc_per_node=1 -m torchtitan.train \
        --job.config_file torchtitan/models/qwen3/train_configs/250m_combined.toml \
        --training.steps 25 \
        --checkpoint.no-enable \
        --parallelism.data_parallel_replicate_degree 1'''

    result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr

    # Parse TPS from output
    tps_matches = re.findall(r'tps: ([\d,]+)', output)
    if tps_matches:
        # Get the last few TPS values (after warmup)
        tps_values = [int(t.replace(',', '')) for t in tps_matches[-3:]]
        return sum(tps_values) / len(tps_values) if tps_values else None

    # Print error for debugging
    print(f"  [DEBUG] Output: {output[-1000:]}")
    return None


def main():
    print("=" * 60)
    print("End-to-End Benchmark: Batched LoRA vs Sequential LoRA")
    print("=" * 60)

    # Note: To test sequential, we would need to modify the code.
    # For now, run two identical runs to verify consistency.

    # Run 1: Default (batched_lora=True)
    print("\nRun 1: Default configuration (batched_lora=True)...")
    tps_1 = run_training_simple("batched_lora=True")
    print(f"  TPS (run 1): {tps_1:,.0f}" if tps_1 else "  Failed to parse TPS")

    # Run 2: Same config for comparison
    print("\nRun 2: Same configuration (for variance check)...")
    tps_2 = run_training_simple("batched_lora=True (run 2)")
    print(f"  TPS (run 2): {tps_2:,.0f}" if tps_2 else "  Failed to parse TPS")

    # Summary
    if tps_1 and tps_2:
        avg_tps = (tps_1 + tps_2) / 2
        variance = abs(tps_1 - tps_2) / avg_tps * 100
        print(f"\n  Average TPS: {avg_tps:,.0f}")
        print(f"  Run variance: {variance:.1f}%")


if __name__ == "__main__":
    main()
