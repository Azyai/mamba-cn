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
