# Mamba-2 (Minimal)

## 安装与环境

```bash
pip install .
```

可选加速：

```bash
pip install "mamba-ssm[causal-conv1d]"
```

依赖环境：Linux / NVIDIA GPU / PyTorch 1.12+ / CUDA 11.6+

## 2.8B 训练脚本（LoRA）

```bash
cd /hy-tmp/mamba
export HF_ENDPOINT=https://hf-mirror.com
python -m pip install -e ".[train]"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/train_offensive.py \
  --pretrained_dir predict/mamba2-2.8b \
  --tokenizer_name_or_path gpt2 \
  --datasets cold,toxicn \
  --toxicn_csv dataset/ToxiCN/ToxiCN_1.0.csv \
  --toxicn_dev_ratio 0.1 \
  --fp16 \
  --batch_size 8 --grad_accum 8 --lr 2e-4 --epochs 8 --max_length 256 \
  --loss focal --focal_gamma 2.0 --focal_alpha_non_toxic 1.3 --focal_alpha_toxic 1.0 \
  --save_dir runs/lora_2_8b_1
```

## 8B 训练脚本（LoRA）

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

## test 目录下的测试命令

离线评估：

```bash
cd /hy-tmp/mamba
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test/val/eval_run.py \
  --run_dir runs/lora3_b \
  --datasets cold,toxicn \
  --split dev \
  --max_items 0 \
  --batch_size 8 \
  --device cuda \
  --dtype fp16
```

查看 run 产物概览：

```bash
cd /hy-tmp/mamba
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test/val/inspect_run.py \
  --run_dir runs/lora3_b \
  --dataset_for_threshold toxicn
```

Web demo：

```bash
cd /hy-tmp/mamba
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test/web_toxicity_demo/server.py \
  --run_dir runs/lora3_b \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda \
  --dtype fp16
```

浏览器打开：http://127.0.0.1:8000/
