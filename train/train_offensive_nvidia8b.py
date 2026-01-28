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


def normalize_path_arg(value: str) -> str:
    return value.replace("\\", "/")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted_dir", type=str, default="predict/mamba2-8b-3t-4k_converted")
    parser.add_argument("--dataset_dir", type=str, default="dataset/COLDataset")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--dev_csv", type=str, default="")
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
    dataset_dir = (root / args.dataset_dir).resolve()
    train_path = Path(args.train_csv) if args.train_csv else dataset_dir / "train.csv"
    dev_path = Path(args.dev_csv) if args.dev_csv else dataset_dir / "dev.csv"

    train_items = read_cold_csv(train_path)
    dev_items = read_cold_csv(dev_path)

    args.converted_dir = normalize_path_arg(args.converted_dir)
    args.tokenizer_model_path = normalize_path_arg(args.tokenizer_model_path)

    tok_cfg = SentencePieceTokenizerConfig(
        model_file=str((root / args.tokenizer_model_path).resolve()) if not Path(args.tokenizer_model_path).is_absolute() else args.tokenizer_model_path,
        add_bos=bool(args.tokenizer_add_bos),
        add_eos=not bool(args.tokenizer_no_eos),
    )
    tokenizer = SentencePieceTokenizer(tok_cfg)

    train_ds = TextLabelDataset(train_items)
    dev_ds = TextLabelDataset(dev_items)

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    save_dir = (root / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

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

    (save_dir / "load_info.json").write_text(json.dumps(load_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_dir / "backbone_converted_dir.txt").write_text(str(converted_dir), encoding="utf-8")

    head = MLPHead(d_model=backbone.config.d_model, hidden_dim=args.head_hidden_dim, dropout=args.dropout).to(device)
    trainable_params = [p for p in head.parameters() if p.requires_grad]
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
                csv_path = save_dir / "train_steps_v2.csv"
                csv_file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        except Exception:
            csv_path = save_dir / "train_steps_v2.csv"
            csv_file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    csv_f = csv_path.open("a", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=desired_csv_fields)
    if not csv_file_exists:
        csv_writer.writeheader()
        csv_f.flush()

    best_f1 = -1.0
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    global_step = 0
    best_head_state = None

    def forward_logits(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        with torch.no_grad():
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs["last_hidden_state"]
            pooled = masked_mean_pool(last_hidden_state, outputs.get("attention_mask", attention_mask))
        pooled = pooled.detach()
        return head(pooled)

    try:
        for epoch in range(1, args.epochs + 1):
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
            all_pred: List[torch.Tensor] = []
            all_gold: List[torch.Tensor] = []
            with torch.no_grad():
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

            pred_cat = torch.cat(all_pred, dim=0)
            gold_cat = torch.cat(all_gold, dim=0)
            metrics = compute_binary_metrics(pred_cat, gold_cat)
            metrics["epoch"] = epoch
            metrics["train_loss"] = total_loss / max(len(train_loader), 1)

            (save_dir / f"metrics_epoch_{epoch}.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            if metrics["f1"] > best_f1:
                best_f1 = float(metrics["f1"])
                best_head_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
                ckpt = {
                    "head": best_head_state,
                    "config": asdict(backbone.config),
                    "tokenizer_model_path": str(Path(tok_cfg.model_file)),
                    "max_length": args.max_length,
                }
                torch.save(ckpt, save_dir / "best_head.pt")
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

    print(f"done. best_f1={best_f1:.4f}. saved at: {save_dir}")


if __name__ == "__main__":
    main()
