#!/usr/bin/env python3
"""
End-to-end benchmark comparing grouped_mm vs sequential LoRA.
"""

import subprocess
import re
import sys


def run_training(use_grouped_mm: bool):
    """Run training and return TPS."""
    # Set environment
    env_cmd = "WANDB_MODE=disabled WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500"

    # Toggle grouped_mm by patching the module
    if not use_grouped_mm:
        # Disable grouped_mm
        patch_cmd = """
import torchtitan.models.qwen3.model.weight_sharing as ws
ws.SharedFeedForward.__init__.__defaults__ = (False,)
ws.SharedAttention.__init__.__defaults__ = (None, False, 1, 8, False, False)
ws.CombinedSharingAttention.__init__.__defaults__ = (False,)
"""
    else:
        patch_cmd = ""

    cmd = f'''
{env_cmd} .venv/bin/python -c "
{patch_cmd}
import os
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

import sys
sys.argv = ['train.py',
    '--job.config_file', 'torchtitan/models/qwen3/train_configs/250m_combined.toml',
    '--training.steps', '30',
    '--checkpoint.no-enable',
    '--parallelism.data_parallel_replicate_degree', '1'
]

from torchtitan.train import main
main()
"
'''
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse TPS from output
    tps_matches = re.findall(r'tps: ([\d,]+)', output)
    if tps_matches:
        # Get the last stable TPS (after warmup)
        tps_values = [int(t.replace(',', '')) for t in tps_matches[-2:]]
        return sum(tps_values) / len(tps_values) if tps_values else None
    return None


def main():
    print("=" * 60)
    print("End-to-End Benchmark: Grouped_mm vs Sequential LoRA")
    print("=" * 60)

    # Run with grouped_mm
    print("\nRunning with grouped_mm enabled...")
    tps_grouped = run_training(use_grouped_mm=True)
    print(f"  TPS (grouped): {tps_grouped:,.0f}" if tps_grouped else "  Failed to parse TPS")

    # Run without grouped_mm
    print("\nRunning with grouped_mm disabled...")
    tps_sequential = run_training(use_grouped_mm=False)
    print(f"  TPS (sequential): {tps_sequential:,.0f}" if tps_sequential else "  Failed to parse TPS")

    # Compare
    if tps_grouped and tps_sequential:
        speedup = tps_grouped / tps_sequential
        print(f"\n  Speedup: {speedup:.2f}x")
        print(f"  {'IMPROVEMENT' if speedup > 1 else 'REGRESSION'}: {(speedup-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
