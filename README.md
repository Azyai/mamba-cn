# Mamba-FK-Lora (Multimodal Edition)

## 安装与环境

建议先创建并激活 conda 环境，并安装 GPU 版 PyTorch（含 CUDA）。

```bash
conda create -n mamba -y
conda activate mamba

# 根据你的 CUDA 版本选择合适的 pytorch/torchvision/torchaudio 组合
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
python -m pip install ".[train]"

pip install "mamba-ssm[causal-conv1d]"

# 图像处理所需依赖
pip install pillow

# (可选) Web端动态OCR和ASR所需依赖(仅在运行 web_toxicity_demo 时需要)
pip install paddleocr paddlepaddle-gpu openai-whisper -i https://mirrors.aliyun.com/pypi/simple/
# Linux 系统可能还需要安装 ffmpeg 以支持 whisper 音频解析
# sudo apt-get install ffmpeg

# 修复 Conda 环境下 PyTorch 可能出现的 libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent 问题
pip install mkl==2024.0.0
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

其他参数继承原文本分类任务配置。

## 8B 训练脚本（LoRA）

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
  --save_dir runs/lora_8_b_multimodal
```

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

### 启用 RAG/Agent 的 Web Demo

```bash
export QWEN_API_KEY=your_key
export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export QWEN_MODEL=qwen-plus

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
