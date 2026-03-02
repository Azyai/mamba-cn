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

This repo includes two training entrypoints for offensive-text classification: a 2.8B base model (frozen backbone + MLP head) and an 8B LoRA finetuning script.

### 2.8B Base (train_offensive.py)

```bash
cd /hy-tmp/mamba
export HF_ENDPOINT=https://hf-mirror.com
python -m pip install -e ".[train]"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/train_offensive.py \
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

2.8B 双数据集训练（cold + toxicn，支持 dataset_weights / balance_datasets）：

```bash
cd /hy-tmp/mamba
export HF_ENDPOINT=https://hf-mirror.com
python -m pip install -e ".[train]"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/train_offensive.py \
  --pretrained_dir predict \
  --tokenizer_name_or_path gpt2 \
  --datasets cold,toxicn \
  --toxicn_csv dataset/ToxiCN/ToxiCN_1.0.csv \
  --toxicn_dev_ratio 0.1 \
  --fp16 \
  --batch_size 8 \
  --grad_accum 4 \
  --max_length 256 \
  --epochs 3 \
  --save_dir runs/offensive_head_multi
```

### NVIDIA Mamba2-8B LoRA (train_offensive_nvidia8b.py)

```bash
cd /hy-tmp/mamba
export HF_ENDPOINT=https://hf-mirror.com
python -m pip install -e ".[train]" sentencepiece

CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/train_offensive_nvidia8b.py \
  --datasets cold,toxicn \
  --toxicn_csv dataset/ToxiCN/ToxiCN_1.0.csv \
  --toxicn_dev_ratio 0.1 \
  --fp16 \
  --batch_size 8 --grad_accum 8 --lr 2e-4 --epochs 8 --max_length 256 \
  --loss focal --focal_gamma 2.0 --focal_alpha_non_toxic 1.3 --focal_alpha_toxic 1.0 \
  --save_dir runs/lora3_b
```

## Prediction and Evaluation

### Offline eval on dev/test splits

```bash
cd /hy-tmp/mamba
python -m pip install -e ".[train]"

# eval run_dir on multiple datasets
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test/val/eval_run.py \
  --run_dir runs/offensive_head_multi \
  --datasets cold,toxicn \
  --split dev \
  --max_items 0 \
  --batch_size 8 \
  --device cuda \
  --dtype fp16
```

### Inspect a run directory (checkpoint + metrics summary)

```bash
cd /hy-tmp/mamba
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test/val/inspect_run.py \
  --run_dir runs/offensive_head_multi \
  --dataset_for_threshold toxicn
```

### Web demo server

```bash
cd /hy-tmp/mamba
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test/web_toxicity_demo/server.py \
  --run_dir runs/offensive_head_multi \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda \
  --dtype fp16
```

Then open http://127.0.0.1:8000/ in your browser.

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
