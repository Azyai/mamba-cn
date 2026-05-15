# Mamba-FK-Lora (Multimodal Edition)

## 安装与环境

建议先创建并激活 conda 环境，并安装 GPU 版 PyTorch（含 CUDA）。

```bash
conda create -n mamba python=3.12 -y
conda activate mamba

# 根据你的 CUDA 版本选择合适的 pytorch/torchvision/torchaudio 组合
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 mkl=2024.0.0 -c pytorch -c nvidia -y

export HF_ENDPOINT=https://hf-mirror.com
```

```bash
python -m pip install ".[train]"


# 图像处理所需依赖
pip install pillow

python -m pip install easyocr

# (可选) Web端动态OCR和ASR所需依赖(仅在运行 web_toxicity_demo 时需要)
pip install paddleocr paddlepaddle-gpu openai-whisper -i https://mirrors.aliyun.com/pypi/simple/
# Linux 系统可能还需要安装 ffmpeg 以支持 whisper 音频解析
# sudo apt-get install ffmpeg
```

RAG/Agent 可选依赖：

```bash
python -m pip install ".[rag]" ".[agent]"
# 或者手动安装
# pip install jieba rank-bm25 sentence-transformers faiss-cpu pyahocorasick langchain-openai
```

如果需要 GPU 版 FAISS，请根据 CUDA 环境单独安装 `faiss-gpu`。

依赖环境：Linux / NVIDIA GPU / PyTorch / CUDA 12.4+ / transformers / pillow

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
│   ├── multimodal/
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
- Mamba文本底座: https://huggingface.co/datasets/ay011123/mamba-fk/upload/main
- 视觉特征底座: `google/vit-base-patch16-224` 或 `OFA-Sys/chinese-clip-vit-base-patch16` (将自动下至 `predict/multimodal`)
- 听觉特征底座: `facebook/wav2vec2-base-960h` (将自动下载至 `predict/multimodal`)

## 多模态 2.8B 训练脚本（LoRA）

支持传入图片和音频数据。训练数据 `csv` 文件需包含 `image_path` 和 `audio_path` 字段。

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python train/train_offensive.py \
  --pretrained_dir predict/mamba2-2.8b \
  --tokenizer_name_or_path gpt2 \
  --vit_name_or_path OFA-Sys/chinese-clip-vit-base-patch16 \
  --wav2vec2_name_or_path facebook/wav2vec2-base-960h \
  --datasets cold,toxicn \
  --toxicn_csv dataset/ToxiCN/ToxiCN_1.0.csv \
  --toxicn_dev_ratio 0.1 \
  --fp16 \
  --batch_size 8 --grad_accum 8 --lr 2e-4 --epochs 8 --max_length 256 \
  --loss focal --focal_gamma 2.0 --focal_alpha_non_toxic 1.3 --focal_alpha_toxic 1.0 \
  --save_dir runs/lora_2_8b_multimodal
```

参数说明（新增多模态参数）：

| 参数 | 作用 | 取值范围/说明 |
| --- | --- | --- |
| `--vit_name_or_path` | 视觉骨干模型路径或HF ID | `google/vit-base-patch16-224` 或 `OFA-Sys/chinese-clip-vit-base-patch16` |
| `--wav2vec2_name_or_path` | 听觉骨干模型路径或HF ID | `facebook/wav2vec2-base-960h` |
| `--multimodal_cache_dir` | 多模态模型下载缓存目录 | 默认 `predict/multimodal` |
| `--image_dim` | 视觉特征维度 | 默认 768 |
| `--audio_dim` | 听觉特征维度 | 默认 768 |
| `--image_drop_prob` | 图像模态随机丢弃概率 | 默认 0.0 表示不丢弃；数值越高表示训练时随机置空图像模态的比例越大，用于缺失模态鲁棒训练 |
| `--audio_drop_prob` | 音频模态随机丢弃概率 | 默认 0.0 表示不丢弃；数值越高表示训练时随机置空音频模态的比例越大，用于缺失模态鲁棒训练 |

其他参数继承原文本分类任务配置。

## 8B 训练脚本（LoRA）

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python train/train_offensive_nvidia8b.py \
  --datasets cold,toxicn \
  --toxicn_csv dataset/ToxiCN/ToxiCN_1.0.csv \
  --toxicn_dev_ratio 0.1 \
  --fp16 \
  --batch_size 8 --grad_accum 8 --lr 2e-4 --epochs 8 --max_length 256 \
  --loss focal --focal_gamma 2.0 --focal_alpha_non_toxic 1.3 --focal_alpha_toxic 1.0 \
  --vit_name_or_path OFA-Sys/chinese-clip-vit-base-patch16 \
  --wav2vec2_name_or_path facebook/wav2vec2-base-960h \
  --save_dir runs/lora_8_b_multimodal
```

### 8B 双向 Mamba-2 上下文增强（可选）

默认情况下，双向 Mamba-2 是关闭的：

- `--bidirectional_layers 0`，即不改动原始 Mamba-2 顺序扫描。
- 多模态融合方式仍然是原来的 MRGF：文本特征先由 `Mamba2Backbone` 提取，再与 ViT 图像特征、Wav2Vec2 音频特征通过缺失模态感知可靠门控融合。
- LoRA 默认仍注入文本主干 Mamba2 block 的 `mixer.in_proj`。

如果要启用论文中的 **Bi-Mamba2 Context Enhancement Block**，推荐先只在顶部若干层开启双向扫描：

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_offensive_nvidia8b.py \
  --datasets cold,toxicn \
  --toxicn_csv dataset/ToxiCN/ToxiCN_1.0.csv \
  --toxicn_dev_ratio 0.1 \
  --fp16 \
  --batch_size 8 --grad_accum 8 --lr 2e-4 --epochs 8 --max_length 256 \
  --loss focal --focal_gamma 2.0 --focal_alpha_non_toxic 1.3 --focal_alpha_toxic 1.0 \
  --vit_name_or_path OFA-Sys/chinese-clip-vit-base-patch16 \
  --wav2vec2_name_or_path facebook/wav2vec2-base-960h \
  --bidirectional_layers 4 \
  --bidirectional_fusion gate \
  --bidirectional_share_mixer \
  --save_dir runs/lora_8_b_bimamba2_multimodal
```

双向扫描参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--bidirectional_layers` | `0` | 启用双向扫描的 Mamba2 block 数量；从主干顶部最后 N 层开始替换，`0` 表示关闭 |
| `--bidirectional_fusion` | `gate` | 前向扫描与反向扫描的融合方式，可选 `add`、`gate`、`concat` |
| `--bidirectional_share_mixer` / `--no-bidirectional_share_mixer` | `True` | 是否让 forward/backward 共享同一个 Mamba-2 mixer 权重 |
| `--bidirectional_train_backward` / `--no-bidirectional_train_backward` | `False` | 在不共享 mixer 时，是否训练独立 backward mixer；默认只训练融合层和 LoRA |

三种常用运行方式：

1. **关闭双向，保持原始 MRGF 多模态方案**
   `--bidirectional_layers 0`
   这是当前默认设置，适合复现实验基线。

2. **共享权重双向扫描，推荐默认实验设置**
   `--bidirectional_layers 4 --bidirectional_fusion gate --bidirectional_share_mixer`
   forward 和 backward 使用同一套 Mamba-2 mixer 参数，对原序列和反转序列各扫描一次，再通过门控融合；参数量基本不增加，但计算量约增加一次 Mamba scan。

3. **独立 backward mixer，容量更大但显存更高**
   `--bidirectional_layers 4 --bidirectional_fusion gate --no-bidirectional_share_mixer`
   会为反向分支创建独立 Mamba-2 mixer，并从 forward mixer 初始化。默认 backward mixer 冻结；若需要训练它，再加 `--bidirectional_train_backward`。8B 场景下该模式显存和参数开销明显更高，建议作为消融实验使用。

融合方式含义：

- `add`：前向/反向输出直接平均，额外参数最少。
- `gate`：使用可学习门控按 token 动态融合前后向上下文，推荐用于攻击性、反讽、否定等需要完整上下文的文本检测。
- `concat`：拼接前后向输出后线性投影回原维度，表达能力较强，但比 `add` 多一个投影层。

实现细节（Bi-Mamba）：

- 双向扫描的实现位于文本主干 `Mamba2Backbone` 内部，具体在 `mamba_ssm/models/mamba2_backbone.py`。当使用 `--bidirectional_layers N` 时，backbone 会对最后 N 层的 `Block` 逐层调用 `enable_bidirectional_`，这些 `Block` 内部负责反向序列的构建、调用 `backward_mixer`（或共享 forward mixer）计算反向特征，并按 `add`/`gate`/`concat` 策略融合前向/反向输出。[实现参考：Block.enable_bidirectional_](mamba_ssm/models/mamba2_backbone.py#L86) / [Mamba2Backbone.enable_bidirectional_](mamba_ssm/models/mamba2_backbone.py#L261)。

- 训练时控制项：通过 `--bidirectional_share_mixer` 决定是否为 backward 创建独立 mixer（默认共享以节省显存），通过 `--bidirectional_train_backward` 在非共享时决定是否解冻 backward mixer 参与训练。默认做法是冻结主干，仅训练 LoRA、分类头与融合（gate/proj）层以降低计算与显存开销。

### PAER 并联式反规避证据保持模块（可选）

PAER（Parallel Anti-Evasion Evidence Retention）是新增的第二个功能模块，默认关闭，不影响原有 `Bi-Mamba2 + MRGF` 实验。启用后，模型会从文本主干输出的 token hidden states 拉出一条并联旁路，进行 toxic span evidence mining、evasion intent detection，并在 `MRGF + MLP` 得到 `base_logits` 后对 toxic logit 做风险校准。

推荐把它作为第二阶段增量消融实验使用：

```text
Baseline + Bi-Mamba2 + MRGF
Baseline + Bi-Mamba2 + MRGF + PAER
```

PAER 同时兼容 2.8B 与 8B 训练脚本。启用示例：

```bash
--paer_enable \
--paer_span_kernel_size 5 \
--paer_topk 3 \
--paer_beta 1.0 \
--paer_lambda_logit 1.0 \
--paer_span_pooling topk
```

常用参数说明：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--paer_enable` | `False` | 启用 PAER 风险校准模块 |
| `--paer_span_kernel_size` | `5` | span-level toxic evidence 的局部卷积窗口，建议使用奇数 |
| `--paer_topk` | `3` | toxic span / evasion risk 的 Top-K pooling 数量 |
| `--paer_beta` | `1.0` | evasion risk 对 toxic logit 增强项的调节系数 |
| `--paer_lambda_logit` | `1.0` | PAER 对 toxic logit 的整体校准强度 |
| `--paer_span_pooling` | `topk` | span 风险聚合方式，可选 `topk` 或 `noisy_or` |
| `--paer_balance_logits` | `False` | 可选项；启用后在增强 toxic logit 的同时轻微降低 non-toxic logit |

完整 8B 组合示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_offensive_nvidia8b.py \
  --datasets cold,toxicn \
  --toxicn_csv dataset/ToxiCN/ToxiCN_1.0.csv \
  --toxicn_dev_ratio 0.1 \
  --fp16 \
  --batch_size 8 --grad_accum 8 --lr 2e-4 --epochs 8 --max_length 256 \
  --loss focal --focal_gamma 2.0 --focal_alpha_non_toxic 1.3 --focal_alpha_toxic 1.0 \
  --vit_name_or_path OFA-Sys/chinese-clip-vit-base-patch16 \
  --wav2vec2_name_or_path facebook/wav2vec2-base-960h \
  --bidirectional_layers 4 \
  --bidirectional_fusion gate \
  --bidirectional_share_mixer \
  --paer_enable \
  --paer_span_kernel_size 5 \
  --paer_topk 3 \
  --paer_beta 1.0 \
  --paer_lambda_logit 1.0 \
  --paer_span_pooling topk \
  --save_dir runs/lora_8_b_bimamba2_mrgf_paer
```

实现位置：

- `mamba_ssm/models/paer_module.py`：PAERModule 与可选 PAERLoss
- `mamba_ssm/models/offensive_classifier.py`：在 `MultimodalClassifier` 中接入 PAER，并返回 `base_logits` / `paer_aux`
- `train/train_offensive.py`、`train/train_offensive_nvidia8b.py`：训练参数与 checkpoint 保存
- `train/offensive_infer.py`：推理时根据 checkpoint 自动重建 PAER

## test 目录下的测试命令

多模态 Web demo 演示（支持纯文本、纯图片、纯音频或任意多模态组合输入）：

```bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python test/web_toxicity_demo/server.py \
  --run_dir runs/lora_2_8b_multimodal \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda \
  --dtype fp16
```

浏览器打开：http://IP:8000/
在界面上不仅可以输入文本，还可以填入图片和音频在服务器上的本地绝对路径进行测试。

### RAG 索引构建（Sensitive-lexicon + 规则文档）

```bash
git clone https://github.com/konsheng/Sensitive-lexicon rag_data/lexicon

python -m mamba_ssm.rag.build_index \
  --lexicon_dir rag_data/lexicon \
  --rules_dir rag_data/rules \
  --output_dir rag_data/index \
  --embedding_model BAAI/bge-base-zh-v1.5 \
  --device cpu
```

如果你想用 GPU 构建，可以把 `--device cpu` 改成 `--device cuda`（脚本也兼容 `GPU` 这种写法，但推荐直接写 `cuda`）。

### 启用 RAG/Agent 的 Web Demo

```bash

export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY=your_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-v4-pro


CUDA_VISIBLE_DEVICES=0 python test/web_toxicity_demo/server.py \
  --run_dir runs/lora_2_8b_multimodal \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda \
  --dtype fp16 \
  --rag_index_dir rag_data/index \
  --rag_device cpu \
  --rag_top_k 5 \
  --fusion_threshold 0.5
```

Agent 对话页入口：http://IP:8000/agent

## 8B 模型测试 (Web Demo)

与多模态 2.8B 模型类似，可以使用以下命令启动运行 8B 模型的 Web Demo：

```bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python test/web_toxicity_demo/server.py \
  --run_dir runs/lora_8_b_multimodal \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda \
  --dtype fp16
```
