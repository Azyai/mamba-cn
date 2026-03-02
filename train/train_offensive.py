from __future__ import annotations

import argparse
import copy
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

from mamba_ssm.models.lora import LoRAConfig, inject_lora, lora_state_dict


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


_CCDC_FLAT_KEYS = {
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
}

_CALIBRATED_FLAT_KEYS = {
    "calibrated_macro_precision",
    "calibrated_macro_recall",
    "calibrated_macro_f1",
    "calibrated_non_toxic_precision",
    "calibrated_non_toxic_recall",
    "calibrated_non_toxic_f1",
    "calibrated_toxic_precision",
    "calibrated_toxic_recall",
    "calibrated_toxic_f1",
    "calibrated_fpr",
}


def compact_metrics_for_save(metrics: Dict[str, object]) -> Dict[str, object]:
    out = copy.deepcopy(metrics)
    if "ccdc" in out:
        for k in _CCDC_FLAT_KEYS:
            out.pop(k, None)
    if "calibrated" in out:
        for k in _CALIBRATED_FLAT_KEYS:
            out.pop(k, None)
    return out


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weight: torch.Tensor | None,
    alpha_non_toxic: float,
    alpha_toxic: float,
    gamma: float,
) -> torch.Tensor:
    labels = labels.to(torch.int64)
    logp = F.log_softmax(logits, dim=-1)
    logp_y = logp.gather(1, labels.view(-1, 1)).squeeze(1)
    ce = -logp_y
    if class_weight is not None:
        w = class_weight.gather(0, labels)
        ce = ce * w
    alpha = torch.where(labels == 0, torch.tensor(alpha_non_toxic, device=logits.device), torch.tensor(alpha_toxic, device=logits.device))
    pt = torch.exp(logp_y)
    loss = alpha * ((1.0 - pt) ** float(gamma)) * ce
    return loss.mean()


def search_best_threshold(
    *,
    probs: torch.Tensor,
    gold: torch.Tensor,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    fpr_max: float,
) -> Dict[str, object]:
    probs = probs.to(torch.float32).view(-1)
    gold = gold.to(torch.int64).view(-1)
    best = {"score": -1e9, "threshold": 0.5, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    t = float(thr_min)
    while t <= float(thr_max) + 1e-12:
        pred = (probs >= t).to(torch.int64)
        tp = int(((pred == 1) & (gold == 1)).sum().item())
        tn = int(((pred == 0) & (gold == 0)).sum().item())
        fp = int(((pred == 1) & (gold == 0)).sum().item())
        fn = int(((pred == 0) & (gold == 1)).sum().item())
        ccdc = compute_ccdc_metrics_from_counts(tp, tn, fp, fn)
        flat = flatten_ccdc_metrics(ccdc)
        if float(flat["fpr"]) <= float(fpr_max) + 1e-12:
            score = float(flat["macro_f1"])
            if score > float(best["score"]):
                best = {"score": score, "threshold": float(t), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
        t += float(thr_step)

    tp, tn, fp, fn = int(best["tp"]), int(best["tn"]), int(best["fp"]), int(best["fn"])
    m = compute_binary_metrics_from_counts(tp, tn, fp, fn)
    ccdc = compute_ccdc_metrics_from_counts(tp, tn, fp, fn)
    out: Dict[str, object] = {"threshold": float(best["threshold"]), "score": float(best["score"]), "metrics": m, "ccdc": ccdc}
    out.update({k: float(v) for k, v in flatten_ccdc_metrics(ccdc).items()})
    return out


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
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden_dim", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_target", type=str, default="in_proj")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_train_head", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_mem_eff_path", action="store_true")
    parser.add_argument("--class_weight_non_toxic", type=float, default=1.0)
    parser.add_argument("--class_weight_toxic", type=float, default=1.0)
    parser.add_argument("--loss", type=str, default="ce")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha_non_toxic", type=float, default=1.0)
    parser.add_argument("--focal_alpha_toxic", type=float, default=1.0)
    parser.add_argument("--best_metric", type=str, default="avg_sum")
    parser.add_argument("--best_fpr_max", type=float, default=1.0)
    parser.add_argument("--eval_optimize_threshold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_threshold_min", type=float, default=0.05)
    parser.add_argument("--eval_threshold_max", type=float, default=0.95)
    parser.add_argument("--eval_threshold_step", type=float, default=0.01)
    parser.add_argument("--eval_threshold_fpr_max", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="runs/offensive_head")
    args = parser.parse_args()

    best_metric_lower = str(args.best_metric).strip().lower()
    if float(args.eval_threshold_fpr_max) == 1.0 and "under_fpr" in best_metric_lower:
        args.eval_threshold_fpr_max = float(args.best_fpr_max)

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
    else:
        model.freeze_backbone_()

    if args.lora_enable and int(args.lora_train_head) <= 0:
        for p in model.head.parameters():
            p.requires_grad = False

    lora_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []
    for name, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            lora_params.append(p)
        else:
            other_params.append(p)
    for p in model.head.parameters():
        if p.requires_grad:
            head_params.append(p)

    param_groups: List[Dict[str, object]] = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": float(args.lora_lr), "weight_decay": 0.0})
    if head_params:
        param_groups.append({"params": head_params, "lr": float(args.head_lr), "weight_decay": float(args.weight_decay)})
    if other_params:
        param_groups.append({"params": other_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})

    optimizer = torch.optim.AdamW(param_groups)

    ce_weight = None
    if float(args.class_weight_non_toxic) != 1.0 or float(args.class_weight_toxic) != 1.0:
        ce_weight = torch.tensor([float(args.class_weight_non_toxic), float(args.class_weight_toxic)], device=device, dtype=torch.float32)
    loss_name = str(args.loss).strip().lower()
    focal_gamma = float(args.focal_gamma)
    focal_alpha_non_toxic = float(args.focal_alpha_non_toxic)
    focal_alpha_toxic = float(args.focal_alpha_toxic)

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

    best_avg_sum = -1.0
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
                    if loss_name == "focal":
                        loss = focal_loss(
                            out.logits,
                            labels,
                            class_weight=ce_weight,
                            alpha_non_toxic=focal_alpha_non_toxic,
                            alpha_toxic=focal_alpha_toxic,
                            gamma=focal_gamma,
                        )
                    else:
                        loss = F.cross_entropy(out.logits, labels, weight=ce_weight)
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

            model.eval()
            all_pred: List[torch.Tensor] = []
            all_gold: List[torch.Tensor] = []
            all_logits: List[torch.Tensor] = []
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
                    all_logits.append(out.logits.detach().cpu())

            pred_cat = torch.cat(all_pred, dim=0)
            gold_cat = torch.cat(all_gold, dim=0)
            tp, tn, fp, fn = compute_binary_counts(pred_cat, gold_cat)
            metrics = compute_binary_metrics_from_counts(tp, tn, fp, fn)
            ccdc = compute_ccdc_metrics_from_counts(tp, tn, fp, fn)
            metrics["ccdc"] = ccdc
            metrics.update(flatten_ccdc_metrics(ccdc))
            if bool(args.eval_optimize_threshold) and all_logits:
                logits_cat = torch.cat(all_logits, dim=0).to(torch.float32)
                probs = torch.softmax(logits_cat, dim=-1)[:, 1]
                cal = search_best_threshold(
                    probs=probs,
                    gold=gold_cat,
                    thr_min=float(args.eval_threshold_min),
                    thr_max=float(args.eval_threshold_max),
                    thr_step=float(args.eval_threshold_step),
                    fpr_max=float(args.eval_threshold_fpr_max),
                )
                metrics["calibrated"] = {"threshold": cal["threshold"], "score": cal["score"], "metrics": cal["metrics"], "ccdc": cal["ccdc"]}
                metrics["calibrated_threshold"] = float(cal["threshold"])
                metrics["calibrated_score"] = float(cal["score"])
                for k in (
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
                ):
                    metrics[f"calibrated_{k}"] = float(cal.get(k, 0.0))
            avg_acc = float(metrics.get("acc", 0.0))
            macro_avg_precision = float(metrics.get("macro_precision", 0.0))
            macro_avg_recall = float(metrics.get("macro_recall", 0.0))
            macro_avg_f1 = float(metrics.get("macro_f1", 0.0))
            non_toxic_avg_precision = float(metrics.get("non_toxic_precision", 0.0))
            non_toxic_avg_recall = float(metrics.get("non_toxic_recall", 0.0))
            non_toxic_avg_f1 = float(metrics.get("non_toxic_f1", 0.0))
            toxic_avg_precision = float(metrics.get("toxic_precision", 0.0))
            toxic_avg_recall = float(metrics.get("toxic_recall", 0.0))
            toxic_avg_f1 = float(metrics.get("toxic_f1", 0.0))
            fpr_score_avg = 1.0 - float(metrics.get("fpr", 0.0))
            avg_sum = (
                avg_acc
                + macro_avg_precision
                + macro_avg_recall
                + macro_avg_f1
                + non_toxic_avg_precision
                + non_toxic_avg_recall
                + non_toxic_avg_f1
                + toxic_avg_precision
                + toxic_avg_recall
                + toxic_avg_f1
                + fpr_score_avg
            )

            metrics["avg_acc"] = float(avg_acc)
            metrics["macro_avg_precision"] = float(macro_avg_precision)
            metrics["macro_avg_recall"] = float(macro_avg_recall)
            metrics["macro_avg_f1"] = float(macro_avg_f1)
            metrics["non_toxic_avg_precision"] = float(non_toxic_avg_precision)
            metrics["non_toxic_avg_recall"] = float(non_toxic_avg_recall)
            metrics["non_toxic_avg_f1"] = float(non_toxic_avg_f1)
            metrics["toxic_avg_precision"] = float(toxic_avg_precision)
            metrics["toxic_avg_recall"] = float(toxic_avg_recall)
            metrics["toxic_avg_f1"] = float(toxic_avg_f1)
            metrics["fpr_score_avg"] = float(fpr_score_avg)
            metrics["avg_sum"] = float(avg_sum)
            metrics["epoch"] = epoch
            metrics["train_loss"] = total_loss / max(len(train_loader), 1)

            (save_dir / f"metrics_epoch_{epoch}.json").write_text(
                json.dumps(compact_metrics_for_save(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
            )

            score = float(avg_sum)

            if float(score) > best_avg_sum:
                best_avg_sum = float(score)
                best_head_state = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}
                ckpt = {
                    "head": best_head_state,
                    "config": asdict(backbone.config),
                    "tokenizer_name_or_path": args.tokenizer_name_or_path,
                    "max_length": args.max_length,
                }
                if lora_cfg is not None:
                    ckpt["lora"] = {k: v.detach().cpu() for k, v in lora_state_dict(backbone).items()}
                    ckpt["lora_cfg"] = json.loads(lora_cfg.to_json())
                    ckpt["lora_replaced"] = list(lora_replaced)
                torch.save(ckpt, save_dir / "best_head.pt")
                if lora_cfg is not None:
                    (save_dir / "lora_config.json").write_text(lora_cfg.to_json(), encoding="utf-8")
                    torch.save({k: v.detach().cpu() for k, v in lora_state_dict(backbone).items()}, save_dir / "lora_adapter.pt")
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

    print(f"done. best_avg_sum={best_avg_sum:.4f}. saved at: {save_dir}")


if __name__ == "__main__":
    main()
