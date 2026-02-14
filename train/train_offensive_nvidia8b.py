from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mamba_ssm.models.mamba2_backbone import Mamba2Backbone
from mamba_ssm.models.offensive_classifier import MLPHead, masked_mean_pool
from mamba_ssm.models.lora import LoRAConfig, inject_lora, lora_state_dict
from sentencepiece_tokenizer import SentencePieceTokenizer, SentencePieceTokenizerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_cold_csv(path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("TEXT") or row.get("text") or row.get("content")
            label = row.get("label") or row.get("LABEL")
            if text is None or label is None:
                continue
            items.append((text, int(label)))
    return items


def read_toxicn_csv(path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("content") or row.get("TEXT") or row.get("text")
            label = row.get("toxic") or row.get("label") or row.get("LABEL")
            if text is None or label is None:
                continue
            items.append((text, int(label)))
    return items


def split_train_dev(items: List[Tuple[str, int]], dev_ratio: float, seed: int) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    if dev_ratio <= 0:
        return items, []
    if dev_ratio >= 1:
        return [], items
    rng = random.Random(seed)
    pos_idx = [i for i, (_, y) in enumerate(items) if int(y) == 1]
    neg_idx = [i for i, (_, y) in enumerate(items) if int(y) == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    total_dev_n = int(len(items) * dev_ratio)
    dev_pos_n = int(len(pos_idx) * dev_ratio)
    dev_neg_n = int(len(neg_idx) * dev_ratio)
    dev_n = dev_pos_n + dev_neg_n
    if dev_n < total_dev_n:
        remaining = total_dev_n - dev_n
        tail = pos_idx[dev_pos_n:] + neg_idx[dev_neg_n:]
        rng.shuffle(tail)
        extra = tail[:remaining]
        dev_idx = set(pos_idx[:dev_pos_n] + neg_idx[:dev_neg_n] + extra)
    else:
        dev_idx = set(pos_idx[:dev_pos_n] + neg_idx[:dev_neg_n])

    train_items = [x for i, x in enumerate(items) if i not in dev_idx]
    dev_items = [x for i, x in enumerate(items) if i in dev_idx]
    return train_items, dev_items


class TextLabelDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        text, label = self.items[idx]
        return {"text": text, "label": label}


def compute_binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_binary_metrics(pred: torch.Tensor, gold: torch.Tensor) -> Dict[str, float]:
    pred = pred.to(torch.int64)
    gold = gold.to(torch.int64)
    tp = int(((pred == 1) & (gold == 1)).sum().item())
    tn = int(((pred == 0) & (gold == 0)).sum().item())
    fp = int(((pred == 1) & (gold == 0)).sum().item())
    fn = int(((pred == 0) & (gold == 1)).sum().item())
    return compute_binary_metrics_from_counts(tp, tn, fp, fn)


def compute_binary_counts(pred: torch.Tensor, gold: torch.Tensor) -> Tuple[int, int, int, int]:
    pred = pred.to(torch.int64)
    gold = gold.to(torch.int64)
    tp = int(((pred == 1) & (gold == 1)).sum().item())
    tn = int(((pred == 0) & (gold == 0)).sum().item())
    fp = int(((pred == 1) & (gold == 0)).sum().item())
    fn = int(((pred == 0) & (gold == 1)).sum().item())
    return tp, tn, fp, fn


def compute_ccdc_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, object]:
    toxic_prec = tp / max(tp + fp, 1)
    toxic_rec = tp / max(tp + fn, 1)
    toxic_f1 = 0.0 if (toxic_prec + toxic_rec) == 0 else 2 * toxic_prec * toxic_rec / (toxic_prec + toxic_rec)

    non_toxic_prec = tn / max(tn + fn, 1)
    non_toxic_rec = tn / max(tn + fp, 1)
    non_toxic_f1 = (
        0.0
        if (non_toxic_prec + non_toxic_rec) == 0
        else 2 * non_toxic_prec * non_toxic_rec / (non_toxic_prec + non_toxic_rec)
    )

    macro_prec = 0.5 * (toxic_prec + non_toxic_prec)
    macro_rec = 0.5 * (toxic_rec + non_toxic_rec)
    macro_f1 = 0.5 * (toxic_f1 + non_toxic_f1)
    fpr = fp / max(fp + tn, 1)

    return {
        "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
        "non_toxic": {"precision": non_toxic_prec, "recall": non_toxic_rec, "f1": non_toxic_f1},
        "toxic": {"precision": toxic_prec, "recall": toxic_rec, "f1": toxic_f1},
        "fpr": fpr,
    }


def flatten_ccdc_metrics(ccdc: Dict[str, object]) -> Dict[str, float]:
    macro = ccdc.get("macro", {}) if isinstance(ccdc.get("macro", {}), dict) else {}
    non_toxic = ccdc.get("non_toxic", {}) if isinstance(ccdc.get("non_toxic", {}), dict) else {}
    toxic = ccdc.get("toxic", {}) if isinstance(ccdc.get("toxic", {}), dict) else {}
    fpr = float(ccdc.get("fpr", 0.0))
    return {
        "macro_precision": float(macro.get("precision", 0.0)),
        "macro_recall": float(macro.get("recall", 0.0)),
        "macro_f1": float(macro.get("f1", 0.0)),
        "non_toxic_precision": float(non_toxic.get("precision", 0.0)),
        "non_toxic_recall": float(non_toxic.get("recall", 0.0)),
        "non_toxic_f1": float(non_toxic.get("f1", 0.0)),
        "toxic_precision": float(toxic.get("precision", 0.0)),
        "toxic_recall": float(toxic.get("recall", 0.0)),
        "toxic_f1": float(toxic.get("f1", 0.0)),
        "fpr": fpr,
    }


def normalize_path_arg(value: str) -> str:
    return value.replace("\\", "/")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted_dir", type=str, default="predict/mamba2-8b-3t-4k_converted")
    parser.add_argument("--dataset_dir", type=str, default="dataset/COLDataset")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--dev_csv", type=str, default="")
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--toxicn_csv", type=str, default="dataset/ToxiCN/ToxiCN_1.0.csv")
    parser.add_argument("--toxicn_dev_ratio", type=float, default=0.1)
    parser.add_argument("--balance_datasets", action="store_true")
    parser.add_argument("--max_train_items", type=int, default=0)
    parser.add_argument("--max_dev_items", type=int, default=0)
    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        default="predict/mamba2-8b-3t-4k_converted/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model",
    )
    parser.add_argument("--tokenizer_add_bos", action="store_true")
    parser.add_argument("--tokenizer_no_eos", action="store_true")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden_dim", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--csv_every", type=int, default=100)
    parser.add_argument("--save_full_model", action="store_true")
    parser.add_argument("--lora_enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_target", type=str, default="in_proj")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_train_head", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--save_dir", type=str, default="runs/offensive_head_nvidia8b")
    args = parser.parse_args()

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA Mamba2-8B 训练需要 CUDA GPU（本机未检测到 CUDA）。")
    device = torch.device("cuda")
    amp_dtype = None
    if args.bf16:
        amp_dtype = torch.bfloat16
    elif args.fp16:
        amp_dtype = torch.float16

    root = Path(__file__).resolve().parents[1]
    args.converted_dir = normalize_path_arg(args.converted_dir)
    args.tokenizer_model_path = normalize_path_arg(args.tokenizer_model_path)
    args.dataset_dir = normalize_path_arg(args.dataset_dir)
    args.train_csv = normalize_path_arg(args.train_csv)
    args.dev_csv = normalize_path_arg(args.dev_csv)
    args.toxicn_csv = normalize_path_arg(args.toxicn_csv)

    tok_cfg = SentencePieceTokenizerConfig(
        model_file=str((root / args.tokenizer_model_path).resolve()) if not Path(args.tokenizer_model_path).is_absolute() else args.tokenizer_model_path,
        add_bos=bool(args.tokenizer_add_bos),
        add_eos=not bool(args.tokenizer_no_eos),
    )
    tokenizer = SentencePieceTokenizer(tok_cfg)

    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        texts = [x["text"] for x in batch]
        labels = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    converted_dir = Path(args.converted_dir)
    if not converted_dir.is_absolute():
        converted_dir = (root / converted_dir).resolve()
    config_path = converted_dir / "config.json"
    weights_path = converted_dir / "model.safetensors"
    if not (config_path.exists() and weights_path.exists()):
        raise FileNotFoundError(
            f"未在 {converted_dir} 找到已转换好的权重文件：config.json / model.safetensors。"
            "请先将预训练 Megatron checkpoint 转换为 safetensors，再运行本训练脚本。"
        )

    backbone_dtype = None
    if args.bf16:
        backbone_dtype = torch.bfloat16
    elif args.fp16:
        backbone_dtype = torch.float16
    backbone, load_info = Mamba2Backbone.load_pretrained(converted_dir, device=device, dtype=backbone_dtype, strict=False)
    backbone = backbone.to(device)
    backbone.freeze_()
    backbone.eval()

    lora_cfg = None
    lora_replaced: List[str] = []
    lora_targets = tuple(x.strip() for x in str(args.lora_target).split(",") if x.strip())
    if args.lora_enable:
        lora_cfg = LoRAConfig(r=int(args.lora_r), alpha=int(args.lora_alpha), dropout=float(args.lora_dropout), target=lora_targets or ("in_proj",))
        lora_replaced = inject_lora(backbone, lora_cfg)
        if args.gradient_checkpointing:
            backbone.enable_gradient_checkpointing_()
        need_disable_mem_eff = bool(args.disable_mem_eff_path) or ("out_proj" in lora_cfg.target)
        if need_disable_mem_eff:
            for layer in backbone.layers:
                mixer = getattr(layer, "mixer", None)
                if mixer is not None and hasattr(mixer, "use_mem_eff_path"):
                    mixer.use_mem_eff_path = False

    def load_dataset(ds_name: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        if ds_name in {"cold", "coldataset", "col"}:
            dataset_dir = (root / "dataset/COLDataset").resolve()
            train_items = read_cold_csv(dataset_dir / "train.csv")
            dev_items = read_cold_csv(dataset_dir / "dev.csv")
            return train_items, dev_items
        if ds_name in {"toxicn", "toxi_cn"}:
            toxicn_path = Path(args.toxicn_csv)
            if not toxicn_path.is_absolute():
                toxicn_path = (root / toxicn_path).resolve()
            all_items = read_toxicn_csv(toxicn_path)
            return split_train_dev(all_items, dev_ratio=args.toxicn_dev_ratio, seed=args.seed)

        dataset_dir = (root / args.dataset_dir).resolve()
        train_path = Path(args.train_csv) if args.train_csv else dataset_dir / "train.csv"
        dev_path = Path(args.dev_csv) if args.dev_csv else dataset_dir / "dev.csv"
        if not train_path.is_absolute():
            train_path = (root / train_path).resolve()
        if not dev_path.is_absolute():
            dev_path = (root / dev_path).resolve()
        return read_cold_csv(train_path), read_cold_csv(dev_path)

    datasets_arg = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not datasets_arg:
        datasets_arg = ["custom"]

    save_dir = (root / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "load_info.json").write_text(json.dumps(load_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_dir / "backbone_converted_dir.txt").write_text(str(converted_dir), encoding="utf-8")

    train_items_by_dataset: Dict[str, List[Tuple[str, int]]] = {}
    dev_items_by_dataset: Dict[str, List[Tuple[str, int]]] = {}
    for ds_name in datasets_arg:
        ds_train, ds_dev = load_dataset(ds_name)
        train_items_by_dataset[ds_name] = ds_train
        dev_items_by_dataset[ds_name] = ds_dev

    rng = random.Random(args.seed)
    if args.balance_datasets and train_items_by_dataset:
        max_len = max((len(v) for v in train_items_by_dataset.values()), default=0)
        balanced: List[Tuple[str, int]] = []
        for ds_name in datasets_arg:
            items = list(train_items_by_dataset.get(ds_name, []))
            if not items:
                continue
            if len(items) < max_len:
                need = max_len - len(items)
                items.extend(rng.choices(items, k=need))
            balanced.extend(items)
        train_items_all = balanced
    else:
        train_items_all: List[Tuple[str, int]] = []
        for ds_name in datasets_arg:
            train_items_all.extend(train_items_by_dataset.get(ds_name, []))

    rng.shuffle(train_items_all)
    if args.max_train_items and args.max_train_items > 0:
        train_items_all = train_items_all[: args.max_train_items]
    for ds_name in list(dev_items_by_dataset.keys()):
        if args.max_dev_items and args.max_dev_items > 0:
            dev_items_by_dataset[ds_name] = dev_items_by_dataset[ds_name][: args.max_dev_items]

    train_ds = TextLabelDataset(train_items_all)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    dev_loaders: Dict[str, DataLoader] = {}
    for ds_name, ds_dev in dev_items_by_dataset.items():
        dev_ds = TextLabelDataset(ds_dev)
        dev_loaders[ds_name] = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    head = MLPHead(d_model=backbone.config.d_model, hidden_dim=args.head_hidden_dim, dropout=args.dropout).to(device)
    if args.lora_enable and int(args.lora_train_head) <= 0:
        for p in head.parameters():
            p.requires_grad = False

    trainable_params = [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    desired_csv_fields = [
        "time",
        "epoch",
        "step",
        "global_step",
        "optimizer_step",
        "window_steps",
        "loss",
        "acc",
        "precision",
        "recall",
        "f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "non_toxic_precision",
        "non_toxic_recall",
        "non_toxic_f1",
        "toxic_precision",
        "toxic_recall",
        "toxic_f1",
        "fpr",
        "lr",
    ]

    csv_path = save_dir / "train_steps.csv"
    csv_file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    if csv_file_exists:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                first_line = f.readline().strip()
            existing_fields = [x.strip() for x in first_line.split(",") if x.strip()]
            if existing_fields != desired_csv_fields:
                csv_path = save_dir / "train_steps_v3.csv"
                csv_file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        except Exception:
            csv_path = save_dir / "train_steps_v3.csv"
            csv_file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    csv_f = csv_path.open("a", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=desired_csv_fields)
    if not csv_file_exists:
        csv_writer.writeheader()
        csv_f.flush()

    best_avg_f1 = -1.0
    best_head_state = None
    best_metrics: Dict[str, object] = {}
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    global_step = 0

    def forward_logits(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if args.lora_enable:
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs["last_hidden_state"]
            pooled = masked_mean_pool(last_hidden_state, outputs.get("attention_mask", attention_mask))
            return head(pooled)
        with torch.no_grad():
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs["last_hidden_state"]
            pooled = masked_mean_pool(last_hidden_state, outputs.get("attention_mask", attention_mask))
        pooled = pooled.detach()
        return head(pooled)

    try:
        for epoch in range(1, args.epochs + 1):
            if args.lora_enable:
                backbone.train()
            else:
                backbone.eval()
            head.train()
            optimizer.zero_grad(set_to_none=True)
            step = 0
            total_loss = 0.0

            log_loss_sum = 0.0
            log_tp = 0
            log_tn = 0
            log_fp = 0
            log_fn = 0
            log_window_steps = 0

            csv_loss_sum = 0.0
            csv_tp = 0
            csv_tn = 0
            csv_fp = 0
            csv_fn = 0
            csv_window_steps = 0

            for batch in train_loader:
                step += 1
                global_step += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                    logits = forward_logits(input_ids=input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(logits, labels)
                    loss = loss / max(args.grad_accum, 1)

                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    pred_pos = pred == 1
                    gold_pos = labels == 1
                    tp = int((pred_pos & gold_pos).sum().item())
                    fp = int((pred_pos & (~gold_pos)).sum().item())
                    fn = int(((~pred_pos) & gold_pos).sum().item())
                    tn = int(((~pred_pos) & (~gold_pos)).sum().item())

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                batch_loss = float(loss.item()) * max(args.grad_accum, 1)
                total_loss += batch_loss

                log_loss_sum += batch_loss
                log_tp += tp
                log_tn += tn
                log_fp += fp
                log_fn += fn
                log_window_steps += 1

                csv_loss_sum += batch_loss
                csv_tp += tp
                csv_tn += tn
                csv_fp += fp
                csv_fn += fn
                csv_window_steps += 1

                if step % args.grad_accum == 0:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.log_every > 0 and log_window_steps >= args.log_every:
                    avg_loss = log_loss_sum / max(log_window_steps, 1)
                    m = compute_binary_metrics_from_counts(log_tp, log_tn, log_fp, log_fn)
                    opt_step = step // max(args.grad_accum, 1)
                    lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.lr
                    print(
                        f"[train] "
                        f"epoch {epoch}/{args.epochs} "
                        f"step {step}/{len(train_loader)} "
                        f"global_step {global_step} "
                        f"opt_step {opt_step} "
                        f"loss {avg_loss:.4f} "
                        f"acc {m['acc']:.4f} "
                        f"prec {m['precision']:.4f} "
                        f"rec {m['recall']:.4f} "
                        f"f1 {m['f1']:.4f} "
                        f"lr {lr:g}"
                    )
                    log_loss_sum = 0.0
                    log_tp = 0
                    log_tn = 0
                    log_fp = 0
                    log_fn = 0
                    log_window_steps = 0

                if args.csv_every > 0 and csv_window_steps >= args.csv_every:
                    avg_loss = csv_loss_sum / max(csv_window_steps, 1)
                    m = compute_binary_metrics_from_counts(csv_tp, csv_tn, csv_fp, csv_fn)
                    ccdc = compute_ccdc_metrics_from_counts(csv_tp, csv_tn, csv_fp, csv_fn)
                    ccdc_flat = flatten_ccdc_metrics(ccdc)
                    opt_step = step // max(args.grad_accum, 1)
                    lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.lr
                    csv_writer.writerow(
                        {
                            "time": f"{time.time():.3f}",
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "optimizer_step": opt_step,
                            "window_steps": csv_window_steps,
                            "loss": f"{avg_loss:.6f}",
                            "acc": f"{m['acc']:.6f}",
                            "precision": f"{m['precision']:.6f}",
                            "recall": f"{m['recall']:.6f}",
                            "f1": f"{m['f1']:.6f}",
                            "macro_precision": f"{ccdc_flat['macro_precision']:.6f}",
                            "macro_recall": f"{ccdc_flat['macro_recall']:.6f}",
                            "macro_f1": f"{ccdc_flat['macro_f1']:.6f}",
                            "non_toxic_precision": f"{ccdc_flat['non_toxic_precision']:.6f}",
                            "non_toxic_recall": f"{ccdc_flat['non_toxic_recall']:.6f}",
                            "non_toxic_f1": f"{ccdc_flat['non_toxic_f1']:.6f}",
                            "toxic_precision": f"{ccdc_flat['toxic_precision']:.6f}",
                            "toxic_recall": f"{ccdc_flat['toxic_recall']:.6f}",
                            "toxic_f1": f"{ccdc_flat['toxic_f1']:.6f}",
                            "fpr": f"{ccdc_flat['fpr']:.6f}",
                            "lr": f"{lr:.12g}",
                        }
                    )
                    csv_f.flush()
                    csv_loss_sum = 0.0
                    csv_tp = 0
                    csv_tn = 0
                    csv_fp = 0
                    csv_fn = 0
                    csv_window_steps = 0

            head.eval()
            backbone.eval()
            eval_metrics: Dict[str, Dict[str, float]] = {}
            eval_f1s: List[float] = []
            with torch.no_grad():
                for ds_name, dev_loader in dev_loaders.items():
                    all_pred: List[torch.Tensor] = []
                    all_gold: List[torch.Tensor] = []
                    for batch in dev_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch.get("attention_mask", None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)
                        labels = batch["labels"].to(device)
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                            logits = forward_logits(input_ids=input_ids, attention_mask=attention_mask)
                            pred = logits.argmax(dim=-1)
                        all_pred.append(pred.detach().cpu())
                        all_gold.append(labels.detach().cpu())
                    pred_cat = torch.cat(all_pred, dim=0) if all_pred else torch.zeros((0,), dtype=torch.int64)
                    gold_cat = torch.cat(all_gold, dim=0) if all_gold else torch.zeros((0,), dtype=torch.int64)
                    tp, tn, fp, fn = compute_binary_counts(pred_cat, gold_cat)
                    m = compute_binary_metrics_from_counts(tp, tn, fp, fn)
                    ccdc = compute_ccdc_metrics_from_counts(tp, tn, fp, fn)
                    ccdc_flat = flatten_ccdc_metrics(ccdc)
                    m_out: Dict[str, object] = {k: float(v) for k, v in m.items()}
                    m_out["size"] = float(len(dev_items_by_dataset.get(ds_name, [])))
                    m_out["ccdc"] = ccdc
                    for k, v in ccdc_flat.items():
                        m_out[k] = float(v)
                    eval_metrics[ds_name] = m_out
                    eval_f1s.append(float(m["f1"]))

            avg_f1 = sum(eval_f1s) / max(len(eval_f1s), 1)
            epoch_metrics: Dict[str, object] = {
                "epoch": float(epoch),
                "train_loss": float(total_loss / max(len(train_loader), 1)),
                "train_size": float(len(train_items_all)),
                "eval": eval_metrics,
                "avg_f1": float(avg_f1),
            }
            (save_dir / f"metrics_epoch_{epoch}.json").write_text(
                json.dumps(epoch_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            if float(avg_f1) > best_avg_f1:
                best_avg_f1 = float(avg_f1)
                best_metrics = dict(epoch_metrics)
                best_head_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
                ckpt = {
                    "head": best_head_state,
                    "config": asdict(backbone.config),
                    "tokenizer_model_path": str(Path(tok_cfg.model_file)),
                    "max_length": args.max_length,
                }
                if lora_cfg is not None:
                    ckpt["lora"] = {k: v.detach().cpu() for k, v in lora_state_dict(backbone).items()}
                    ckpt["lora_cfg"] = json.loads(lora_cfg.to_json())
                    ckpt["lora_replaced"] = list(lora_replaced)
                torch.save(ckpt, save_dir / "best_head.pt")
                (save_dir / "best_metrics.json").write_text(
                    json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if lora_cfg is not None:
                    (save_dir / "lora_config.json").write_text(lora_cfg.to_json(), encoding="utf-8")
                    torch.save({k: v.detach().cpu() for k, v in lora_state_dict(backbone).items()}, save_dir / "lora_adapter.pt")

            parts = [f"{k}:{float(eval_metrics[k]['f1']):.4f}" for k in eval_metrics]
            print(f"[eval] epoch {epoch}/{args.epochs} avg_f1 {avg_f1:.4f} " + " ".join(parts))
    finally:
        csv_f.close()

    if args.save_full_model and best_head_state is not None:
        head.load_state_dict(best_head_state, strict=True)
        full_ckpt = {
            "backbone": {k: v.detach().cpu() for k, v in backbone.state_dict().items()},
            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "config": asdict(backbone.config),
            "tokenizer_model_path": str(Path(tok_cfg.model_file)),
            "max_length": args.max_length,
            "head_hidden_dim": args.head_hidden_dim,
            "dropout": args.dropout,
            "num_labels": 2,
        }
        torch.save(full_ckpt, save_dir / "full_model.pt")

    (save_dir / "benchmark_summary.json").write_text(
        json.dumps({"datasets": datasets_arg, "best": best_metrics}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"done. best_avg_f1={best_avg_f1:.4f}. saved at: {save_dir}")


if __name__ == "__main__":
    main()
