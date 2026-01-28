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


def compute_binary_metrics(pred: torch.Tensor, gold: torch.Tensor) -> Dict[str, float]:
    pred = pred.to(torch.int64)
    gold = gold.to(torch.int64)
    tp = int(((pred == 1) & (gold == 1)).sum().item())
    tn = int(((pred == 0) & (gold == 0)).sum().item())
    fp = int(((pred == 1) & (gold == 0)).sum().item())
    fn = int(((pred == 0) & (gold == 1)).sum().item())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


def compute_binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    eps = 1e-10
    f1 = 0.0 if (prec + rec) < eps else 2 * prec * rec / (prec + rec)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}


def _load_pretrained_vocab_size(pretrained_dir: str) -> int:
    try:
        cfg_path = Path(pretrained_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        vocab_size = int(cfg.get("vocab_size", 0))
        return vocab_size if vocab_size > 0 else 0
    except Exception:
        return 0


def _build_fallback_tokenizer(texts: List[str], vocab_size: int):
    try:
        from tokenizers import Tokenizer
        from tokenizers import decoders, models, pre_tokenizers, processors, trainers
        from transformers import PreTrainedTokenizerFast
    except Exception as e:
        raise RuntimeError(
            "无法创建本地 tokenizer（需要 tokenizers + transformers）。"
            "请安装: pip install transformers"
        ) from e

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tok.train_from_iterator(texts, trainer=trainer)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    return hf_tok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_dir", type=str, default="predict/mamba2-2.8b")
    parser.add_argument("--dataset_dir", type=str, default="dataset/COLDataset")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--dev_csv", type=str, default="")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_cache_dir", type=str, default="predict/gpt2")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--csv_every", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden_dim", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="runs/offensive_head")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None
    if device.type == "cuda" and args.bf16:
        amp_dtype = torch.bfloat16
    elif device.type == "cuda" and args.fp16:
        amp_dtype = torch.float16

    root = Path(__file__).resolve().parents[1]
    dataset_dir = (root / args.dataset_dir).resolve()
    train_path = Path(args.train_csv) if args.train_csv else dataset_dir / "train.csv"
    dev_path = Path(args.dev_csv) if args.dev_csv else dataset_dir / "dev.csv"

    train_items = read_cold_csv(train_path)
    dev_items = read_cold_csv(dev_path)

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("缺少 transformers（用于 tokenizer）。请安装: pip install transformers") from e

    tokenizer_cache_dir = Path(args.tokenizer_cache_dir)
    if not tokenizer_cache_dir.is_absolute():
        tokenizer_cache_dir = (root / tokenizer_cache_dir).resolve()
    tokenizer_cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path, use_fast=True, local_files_only=True, cache_dir=str(tokenizer_cache_dir)
        )
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name_or_path, use_fast=True, cache_dir=str(tokenizer_cache_dir)
            )
        except Exception:
            texts = [t for (t, _) in train_items]
            inferred_vocab_size = _load_pretrained_vocab_size(args.pretrained_dir)
            target_vocab_size = 8192
            if inferred_vocab_size > 0:
                target_vocab_size = min(target_vocab_size, inferred_vocab_size)
            tokenizer = _build_fallback_tokenizer(texts, vocab_size=target_vocab_size)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

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

    try:
        from mamba_ssm.models.mamba2_backbone import Mamba2Backbone
        from mamba_ssm.models.offensive_classifier import FrozenBackboneClassifier, MLPHead
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "未能导入 mamba_ssm（通常是缺少 triton 或 GPU 环境不满足）。"
            "该训练脚本需要 Linux + CUDA + triton 才能运行 Mamba2 内核。"
        ) from e

    backbone, load_info = Mamba2Backbone.load_pretrained(args.pretrained_dir, device=device, dtype=None, strict=False)
    backbone = backbone.to(device)
    backbone.freeze_()

    head = MLPHead(d_model=backbone.config.d_model, hidden_dim=args.head_hidden_dim, dropout=args.dropout).to(device)
    model = FrozenBackboneClassifier(backbone=backbone, head=head).to(device)
    model.freeze_backbone_()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    save_dir = (root / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "load_info.json").write_text(json.dumps(load_info, ensure_ascii=False, indent=2), encoding="utf-8")

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

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
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
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(out.logits, labels)
                    loss = loss / max(args.grad_accum, 1)

                with torch.no_grad():
                    pred = out.logits.argmax(dim=-1)
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

            model.eval()
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
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                        pred = out.logits.argmax(dim=-1)
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
                best_head_state = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
                ckpt = {
                    "head": best_head_state,
                    "config": asdict(backbone.config),
                    "tokenizer_name_or_path": args.tokenizer_name_or_path,
                    "max_length": args.max_length,
                }
                torch.save(ckpt, save_dir / "best_head.pt")
    finally:
        csv_f.close()

    if best_head_state is not None:
        model.head.load_state_dict(best_head_state, strict=True)
        full_ckpt = {
            "backbone": {k: v.detach().cpu() for k, v in model.backbone.state_dict().items()},
            "head": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},
            "config": asdict(backbone.config),
            "tokenizer_name_or_path": args.tokenizer_name_or_path,
            "max_length": args.max_length,
            "head_hidden_dim": args.head_hidden_dim,
            "dropout": args.dropout,
            "num_labels": 2,
        }
        torch.save(full_ckpt, save_dir / "full_model.pt")

    print(f"done. best_f1={best_f1:.4f}. saved at: {save_dir}")


if __name__ == "__main__":
    main()
