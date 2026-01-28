# Mamba-2 (Minimal)

This repository has been pruned to keep only the Python code required to run the Mamba-2 block (and a simplified variant).

Paper: https://arxiv.org/abs/2405.21060

## Installation

From this repository:

```bash
pip install .
```

Optional speedups:

```bash
pip install "mamba-ssm[causal-conv1d]"
```

Upstream requirements (unchanged by this pruning):

- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

## Usage

Mamba-2 block:

```python
import torch
from mamba_ssm import Mamba2

batch, length, dim = 2, 64, 256
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(
    d_model=dim,
    d_state=64,
    d_conv=4,
    expand=2,
).to("cuda")
y = model(x)
assert y.shape == x.shape
```

## Offensive Training

This repo includes a minimal offensive-text classifier training script that freezes a Mamba2 backbone and trains a small MLP head.

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m pip install -e ".[train]"

.venv/bin/python train/train_offensive.py \
  --pretrained_dir predict \
  --tokenizer_name_or_path gpt2 \
  --train_csv dataset/COLDataset/train.csv \
  --dev_csv dataset/COLDataset/dev.csv \
  --fp16 \
  --batch_size 8 \
  --grad_accum 4 \
  --max_length 256 \
  --epochs 3 \
  --save_dir runs/offensive_head
```

### NVIDIA Mamba2-8B (Megatron checkpoint) Version

If you downloaded `nvidia/mamba2-8b-3t-4k` in Megatron-LM checkpoint format (e.g. `release/mp_rank_00/model_optim_rng.pt`), use the 8B training entrypoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m pip install -e ".[train]" sentencepiece

.venv/bin/python train/train_offensive_nvidia8b.py \
  --megatron_ckpt predict/mamba2-8b-3t-4k/release/mp_rank_00/model_optim_rng.pt \
  --converted_dir predict/mamba2-8b-3t-4k_converted \
  --tokenizer_model_path predict/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
  --train_csv dataset/COLDataset/train.csv \
  --dev_csv dataset/COLDataset/dev.csv \
  --fp16 \
  --batch_size 1 \
  --grad_accum 8 \
  --max_length 256 \
  --epochs 1 \
  --save_dir runs/offensive_head_nvidia8b
```

Notes:

- The script automatically converts the Megatron checkpoint to a local `config.json` + `model.safetensors` folder (default: `predict/mamba2-8b-3t-4k_converted`) before loading it.
- The backbone forward is forced under `no_grad` and pooled features are detached, so only the head is trained (lower VRAM than full finetuning).

Simplified block:

```python
import torch
from mamba_ssm import Mamba2Simple

batch, length, dim = 2, 64, 256
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2Simple(
    d_model=dim,
    d_state=64,
    d_conv=4,
    expand=2,
).to("cuda")
y = model(x)
assert y.shape == x.shape
```

## Citation

```bibtex
@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```
