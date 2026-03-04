# Mamba-2 (Minimal)

## 安装与环境

建议先创建并激活 conda 环境，并安装 GPU 版 PyTorch（含 CUDA）。

```bash
conda create -n mamba python=3.10 -y
conda activate mamba

# 根据你的 CUDA 版本选择合适的 pytorch/torchvision/torchaudio 组合
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

```bash
pip install .
```

可选加速：

```bash
pip install "mamba-ssm[causal-conv1d]"
```
依赖环境：Linux / NVIDIA GPU / PyTorch 1.12+ / CUDA 11.6+

## 项目目录结构

```text
.
├── dataset/
│   ├── COLDataset/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   ├── test.csv
│   │   └── smoke.csv
│   └── ToxiCN/
│       ├── ToxiCN_1.0.csv
│       ├── train.json
│       └── test.json
├── mamba_ssm/
│   ├── distributed/
│   ├── models/
│   ├── modules/
│   ├── ops/
│   └── utils/
├── predict/
│   ├── gpt2/
│   ├── mamba2-2.8b/
│   └── mamba2-8b-3t-4k_converted/
├── runs/
│   ├── joint_officialsplit_acc/
│   ├── lora_2_8b_1/
│   └── lora_8_b_1/
├── test/
│   ├── val/
│   └── web_toxicity_demo/
├── train/
│   ├── train_offensive.py
│   ├── train_offensive_nvidia8b.py
│   ├── offensive_infer.py
│   └── sentencepiece_tokenizer.py
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
├── README.md
└── setup.py
```

项目所需数据集：
- https://huggingface.co/datasets/ay011123/mamba-fk-dataset/tree/main

项目所需预训练模型：
- https://huggingface.co/datasets/ay011123/mamba-fk/upload/main


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

参数说明（2.8B，train_offensive.py）：

| 参数 | 作用 | 取值范围/说明 |
| --- | --- | --- |
| `--pretrained_dir` | 预训练权重目录 | 路径，需包含 config.json 与 model.safetensors |
| `--dataset_dir` | 默认数据集目录 | 路径 |
| `--train_csv` | 训练集 CSV 路径 | 路径；为空则用默认 |
| `--dev_csv` | 验证集 CSV 路径 | 路径；为空则用默认 |
| `--datasets` | 使用的数据集 | 逗号分隔，例如 cold,toxicn |
| `--toxicn_csv` | ToxiCN CSV 路径 | 路径 |
| `--toxicn_train_json` | ToxiCN 训练 JSON 路径 | 路径 |
| `--toxicn_test_json` | ToxiCN 测试 JSON 路径 | 路径 |
| `--toxicn_dev_ratio` | ToxiCN dev 划分比例 | 0-1 浮点数 |
| `--toxicn_add_metadata` | 是否拼接元数据 | true/false |
| `--balance_datasets` | 数据集均衡采样 | true/false |
| `--dataset_weights` | 数据集权重 | 形如 cold=1,toxicn=2 |
| `--max_train_items` | 训练样本上限 | >= 0；0 表示不限制 |
| `--max_dev_items` | 验证样本上限 | >= 0；0 表示不限制 |
| `--tokenizer_name_or_path` | tokenizer 名称或路径 | 必填；HF 名称或本地路径 |
| `--tokenizer_cache_dir` | tokenizer 缓存目录 | 路径 |
| `--log_every` | 日志频率 | >= 0 step；0 关闭 |
| `--csv_every` | CSV 记录频率 | >= 0 step；0 关闭 |
| `--max_length` | 最大序列长度 | 正整数 |
| `--batch_size` | batch 大小 | 正整数 |
| `--epochs` | 训练轮数 | 正整数 |
| `--lr` | 基础学习率 | > 0 |
| `--lora_lr` | LoRA 学习率 | > 0 |
| `--head_lr` | 分类头学习率 | > 0 |
| `--weight_decay` | 权重衰减 | >= 0 |
| `--grad_accum` | 梯度累积步数 | 正整数 |
| `--dropout` | dropout | 0-1 |
| `--head_hidden_dim` | 分类头隐藏维度 | 正整数 |
| `--fp16` | 开启 FP16 | true/false |
| `--bf16` | 开启 BF16 | true/false |
| `--seed` | 随机种子 | 整数 |
| `--lora_enable` | 启用 LoRA | true/false |
| `--lora_target` | LoRA 目标模块 | 逗号分隔，如 in_proj |
| `--lora_r` | LoRA rank | 正整数 |
| `--lora_alpha` | LoRA alpha | 正整数 |
| `--lora_dropout` | LoRA dropout | 0-1 |
| `--lora_train_head` | 是否训练分类头 | <= 0 则冻结；> 0 则训练 |
| `--gradient_checkpointing` | 梯度检查点 | true/false |
| `--disable_mem_eff_path` | 关闭高效路径 | true/false |
| `--class_weight_non_toxic` | 非毒类权重 | > 0 |
| `--class_weight_toxic` | 毒类权重 | > 0 |
| `--loss` | 损失函数 | ce 或 focal |
| `--focal_gamma` | focal gamma | >= 0 |
| `--focal_alpha_non_toxic` | focal alpha(非毒) | > 0 |
| `--focal_alpha_toxic` | focal alpha(毒) | > 0 |
| `--eval_optimize_threshold` | 阈值优化 | true/false |
| `--eval_threshold_min` | 阈值搜索下限 | 0-1 |
| `--eval_threshold_max` | 阈值搜索上限 | 0-1 |
| `--eval_threshold_step` | 阈值步长 | > 0 |
| `--eval_threshold_fpr_max` | 阈值搜索 FPR 上限 | 0-1 |
| `--eval_threshold_objective` | 阈值优化目标 | macro_f1/acc/toxic_recall/toxic_f1 |
| `--train_norm` | 训练归一化层 | true/false |
| `--save_full_model` | 保存全模型 | true/false |
| `--save_dir` | 输出目录 | 路径 |

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
  --save_dir runs/lora_8_b_1
```

参数说明（8B，train_offensive_nvidia8b.py）：

| 参数 | 作用 | 取值范围/说明 |
| --- | --- | --- |
| `--converted_dir` | 8B 转换后权重目录 | 路径，需包含 config.json 与 model.safetensors |
| `--dataset_dir` | 默认数据集目录 | 路径 |
| `--train_csv` | 训练集 CSV 路径 | 路径；为空则用默认 |
| `--dev_csv` | 验证集 CSV 路径 | 路径；为空则用默认 |
| `--datasets` | 使用的数据集 | 逗号分隔，例如 cold,toxicn |
| `--toxicn_csv` | ToxiCN CSV 路径 | 路径 |
| `--toxicn_train_json` | ToxiCN 训练 JSON 路径 | 路径 |
| `--toxicn_test_json` | ToxiCN 测试 JSON 路径 | 路径 |
| `--toxicn_dev_ratio` | ToxiCN dev 划分比例 | 0-1 浮点数 |
| `--toxicn_add_metadata` | 是否拼接元数据 | true/false |
| `--balance_datasets` | 数据集均衡采样 | true/false |
| `--dataset_weights` | 数据集权重 | 形如 cold=1,toxicn=2 |
| `--max_train_items` | 训练样本上限 | >= 0；0 表示不限制 |
| `--max_dev_items` | 验证样本上限 | >= 0；0 表示不限制 |
| `--tokenizer_model_path` | SentencePiece 模型路径 | 路径 |
| `--tokenizer_add_bos` | tokenizer 加 BOS | true/false |
| `--tokenizer_no_eos` | tokenizer 不加 EOS | true/false |
| `--max_length` | 最大序列长度 | 正整数 |
| `--batch_size` | batch 大小 | 正整数 |
| `--epochs` | 训练轮数 | 正整数 |
| `--lr` | 基础学习率 | > 0 |
| `--lora_lr` | LoRA 学习率 | > 0 |
| `--head_lr` | 分类头学习率 | > 0 |
| `--weight_decay` | 权重衰减 | >= 0 |
| `--grad_accum` | 梯度累积步数 | 正整数 |
| `--dropout` | dropout | 0-1 |
| `--head_hidden_dim` | 分类头隐藏维度 | 正整数 |
| `--fp16` | 开启 FP16 | true/false |
| `--bf16` | 开启 BF16 | true/false |
| `--seed` | 随机种子 | 整数 |
| `--log_every` | 日志频率 | >= 0 step；0 关闭 |
| `--csv_every` | CSV 记录频率 | >= 0 step；0 关闭 |
| `--save_full_model` | 保存全模型 | true/false |
| `--lora_enable` | 启用 LoRA | true/false |
| `--lora_target` | LoRA 目标模块 | 逗号分隔，如 in_proj |
| `--lora_r` | LoRA rank | 正整数 |
| `--lora_alpha` | LoRA alpha | 正整数 |
| `--lora_dropout` | LoRA dropout | 0-1 |
| `--lora_train_head` | 是否训练分类头 | <= 0 则冻结；> 0 则训练 |
| `--gradient_checkpointing` | 梯度检查点 | true/false |
| `--disable_mem_eff_path` | 关闭高效路径 | true/false |
| `--class_weight_non_toxic` | 非毒类权重 | > 0 |
| `--class_weight_toxic` | 毒类权重 | > 0 |
| `--loss` | 损失函数 | ce 或 focal |
| `--focal_gamma` | focal gamma | >= 0 |
| `--focal_alpha_non_toxic` | focal alpha(非毒) | > 0 |
| `--focal_alpha_toxic` | focal alpha(毒) | > 0 |
| `--eval_optimize_threshold` | 阈值优化 | true/false |
| `--eval_threshold_min` | 阈值搜索下限 | 0-1 |
| `--eval_threshold_max` | 阈值搜索上限 | 0-1 |
| `--eval_threshold_step` | 阈值步长 | > 0 |
| `--eval_threshold_fpr_max` | 阈值搜索 FPR 上限 | 0-1 |
| `--eval_threshold_objective` | 阈值优化目标 | macro_f1/acc/toxic_recall/toxic_f1 |
| `--train_norm` | 训练归一化层 | true/false |
| `--save_dir` | 输出目录 | 路径 |

## test 目录下的测试命令

离线评估：

```bash
cd /hy-tmp/mamba
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test/val/eval_run.py \
  --run_dir runs/lora_8_b_1 \
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
  --run_dir runs/lora_8_b_1 \
  --dataset_for_threshold toxicn
```

Web demo：

```bash
cd /hy-tmp/mamba
CUDA_VISIBLE_DEVICES=0 .venv/bin/python test/web_toxicity_demo/server.py \
  --run_dir runs/lora_8_b_1 \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda \
  --dtype fp16
```

浏览器打开：http://127.0.0.1:8000/
