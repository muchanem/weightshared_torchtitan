# Weight-Shared TorchTitan

Qwen3 250M weight sharing experiments built on [torchtitan](https://github.com/pytorch/torchtitan).

Two configs are provided for 8xH100:
- **250m_combined** — attention sharing + layer sharing + factorized embeddings
- **250m_unshared** — baseline without weight sharing

Both train on the `hq_data_20bt` dataset (expected at `/fsx-checkpoints/sanaelotfi/data/hq_data_20bt`).

## Installation

Requires Python >= 3.10 and CUDA-capable GPUs. We install from source with PyTorch nightly.

```bash
conda create -n torchtitan python=3.10 -y
conda activate torchtitan

# Install PyTorch nightly (replace cu126 with your CUDA version if needed)
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall

# Install torchtitan from source
pip install -r requirements.txt
pip install -e .
```

### Download tokenizer assets

The Qwen3 configs expect tokenizer files at `./assets/hf/Qwen3-0.6B-Base`.

```bash
# Get your HF token from https://huggingface.co/settings/tokens
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B-Base --assets tokenizer --hf_token=...
```

## Running 250M experiments (8xH100)

**Combined weight sharing:**

```bash
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/250m_combined.toml" ./run_train.sh
```

**Unshared baseline:**

```bash
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/250m_unshared.toml" ./run_train.sh
```

Both configs default to `NGPU=8`. To override, set `NGPU` before the command. Metrics are logged to Weights & Biases (set `WANDB_API_KEY` or run `wandb login` first).

---

## About torchtitan

`torchtitan` is a PyTorch native platform designed for rapid experimentation and large-scale training of generative AI models. As a minimal clean-room implementation of PyTorch native scaling techniques, it provides a flexible foundation for developers to build upon. With `torchtitan` [extension points](docs/extension.md), one can easily create custom extensions tailored to specific needs.

### Key features

1. Multi-dimensional composable parallelisms
   - [FSDP2](docs/fsdp.md) with per-parameter sharding
   - [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) (including [async TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487))
   - [Pipeline Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
   - [Context Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082)
2. [Meta device](https://pytorch.org/docs/stable/meta.html) initialization
3. Selective (layer or operator) and full activation checkpointing
4. [Distributed checkpointing](https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250) (including async checkpointing)
   - [Interoperable checkpoints](docs/checkpoint.md) which can be loaded directly into [`torchtune`](https://github.com/pytorch/torchtune) for fine-tuning
5. `torch.compile` support
6. [Float8](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323) support ([how-to](docs/float8.md))
7. [MXFP8 training for dense and MoE models](docs/mxfp8.md) on Blackwell GPUs
8. DDP and HSDP
9. [TorchFT](https://github.com/pytorch/torchft) integration
10. Checkpointable data-loading, with the C4 dataset pre-configured (144M entries) and support for [custom datasets](docs/datasets.md)
11. Gradient accumulation, enabled by giving an additional `--training.global_batch_size` argument in configuration
12. Flexible learning rate scheduler (warmup-stable-decay)
13. Loss, GPU memory, throughput (tokens/sec), TFLOPs, and MFU displayed and logged via [Tensorboard or Weights & Biases](/docs/metrics.md)
14. [Debugging tools](docs/debugging.md) including CPU/GPU profiling, memory profiling, Flight Recorder, etc.
15. All options easily configured via [toml files](torchtitan/models/llama3/train_configs/)
16. [Helper scripts](scripts/) to
    - download tokenizers from Hugging Face
    - convert original Llama 3 checkpoints into the expected DCP format
    - estimate FSDP/HSDP memory usage without materializing the model
    - run distributed inference with Tensor Parallel

### Alternative installation methods

**Nightly builds** (requires nightly PyTorch):

```sh
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu126
```

**Stable releases** via pip or conda:

```sh
pip install torchtitan
```
```sh
conda install conda-forge::torchtitan
```

Note that each stable release pins the nightly versions of `torch` and `torchao`. Please see [release.md](docs/release.md) for more details.

### Multi-Node Training

For training on ParallelCluster/Slurm type configurations, you can use the `multinode_trainer.slurm` file to submit your sbatch job.

To get started adjust the number of nodes and GPUs:
```
#SBATCH --ntasks=2
#SBATCH --nodes=2
```

Then start a run where `nnodes` is your total node count, matching the sbatch node count above.

```
srun torchrun --nnodes 2
```

If your gpu count per node is not 8, adjust `--nproc_per_node` in the torchrun command and `#SBATCH --gpus-per-task` in the SBATCH command section.

## Citation

[TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://openreview.net/forum?id=SFN6Wm7YBI)
```
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```

## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.
