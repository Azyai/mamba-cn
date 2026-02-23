from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from train.offensive_infer import load_offensive_predictor


def read_cold_csv(path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("TEXT") or row.get("text") or row.get("content")
            label = row.get("label") or row.get("LABEL")
            if text is None or label is None:
                continue
            items.append((str(text), int(label)))
    return items


def read_toxicn_csv(path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("content") or row.get("text") or row.get("TEXT")
            label = row.get("toxic") or row.get("label") or row.get("LABEL")
            if text is None or label is None:
                continue
            items.append((str(text), int(label)))
    return items


def read_toxicn_json(path: Path) -> List[Tuple[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items: List[Tuple[str, int]] = []
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            text = row.get("content") or row.get("text")
            label = row.get("toxic") if "toxic" in row else row.get("label")
            if text is None or label is None:
                continue
            items.append((str(text), int(label)))
    return items


def compute_ccdc_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Any]:
    toxic_prec = tp / max(tp + fp, 1)
    toxic_rec = tp / max(tp + fn, 1)
    toxic_f1 = 0.0 if (toxic_prec + toxic_rec) == 0 else 2 * toxic_prec * toxic_rec / (toxic_prec + toxic_rec)

    non_toxic_prec = tn / max(tn + fn, 1)
    non_toxic_rec = tn / max(tn + fp, 1)
    non_toxic_f1 = 0.0 if (non_toxic_prec + non_toxic_rec) == 0 else 2 * non_toxic_prec * non_toxic_rec / (non_toxic_prec + non_toxic_rec)

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


def flatten_ccdc_metrics(ccdc: Dict[str, Any]) -> Dict[str, float]:
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


def compute_counts(pred: List[int], gold: List[int]) -> Tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for p, g in zip(pred, gold):
        if p == 1 and g == 1:
            tp += 1
        elif p == 0 and g == 0:
            tn += 1
        elif p == 1 and g == 0:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def _objective_score(flat: Dict[str, float], objective: str) -> float:
    o = str(objective).strip().lower()
    if o == "acc":
        tp = flat.get("_tp", 0.0)
        tn = flat.get("_tn", 0.0)
        fp = flat.get("_fp", 0.0)
        fn = flat.get("_fn", 0.0)
        return float(tp + tn) / max(float(tp + tn + fp + fn), 1.0)
    if o == "toxic_recall":
        tp = flat.get("_tp", 0.0)
        fn = flat.get("_fn", 0.0)
        return float(tp) / max(float(tp + fn), 1.0)
    if o == "toxic_f1":
        return float(flat.get("toxic_f1", 0.0))
    return float(flat.get("macro_f1", 0.0))


def search_best_threshold(
    probs: List[float],
    gold: List[int],
    *,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    fpr_max: float,
    objective: str,
) -> Dict[str, Any]:
    best = {"score": -1e9, "threshold": 0.5, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    t = float(thr_min)
    while t <= float(thr_max) + 1e-12:
        pred = [1 if float(p) >= float(t) else 0 for p in probs]
        tp, tn, fp, fn = compute_counts(pred, gold)
        ccdc = compute_ccdc_metrics_from_counts(tp, tn, fp, fn)
        flat = flatten_ccdc_metrics(ccdc)
        flat["_tp"] = float(tp)
        flat["_tn"] = float(tn)
        flat["_fp"] = float(fp)
        flat["_fn"] = float(fn)
        if float(flat["fpr"]) <= float(fpr_max) + 1e-12:
            score = _objective_score(flat, objective)
            if float(score) > float(best["score"]):
                best = {"score": float(score), "threshold": float(t), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
        t += float(thr_step)
    tp, tn, fp, fn = int(best["tp"]), int(best["tn"]), int(best["fp"]), int(best["fn"])
    ccdc = compute_ccdc_metrics_from_counts(tp, tn, fp, fn)
    out: Dict[str, Any] = {"threshold": float(best["threshold"]), "score": float(best["score"]), "ccdc": ccdc}
    out.update(flatten_ccdc_metrics(ccdc))
    out["objective"] = str(objective)
    return out


def load_items_for_dataset(root: Path, name: str, split: str, toxicn_train_json: str, toxicn_test_json: str, toxicn_csv: str) -> List[Tuple[str, int]]:
    n = str(name).strip().lower()
    s = str(split).strip().lower()
    if n in {"cold", "coldataset", "col"}:
        dataset_dir = (root / "dataset" / "COLDataset").resolve()
        if s == "test":
            return read_cold_csv(dataset_dir / "test.csv")
        return read_cold_csv(dataset_dir / "dev.csv")
    if n in {"toxicn", "toxi_cn"}:
        train_json = Path(toxicn_train_json)
        test_json = Path(toxicn_test_json)
        if not train_json.is_absolute():
            train_json = (root / train_json).resolve()
        if not test_json.is_absolute():
            test_json = (root / test_json).resolve()
        if test_json.exists():
            return read_toxicn_json(test_json)
        csv_path = Path(toxicn_csv)
        if not csv_path.is_absolute():
            csv_path = (root / csv_path).resolve()
        return read_toxicn_csv(csv_path)
    raise ValueError(f"不支持的数据集: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--datasets", type=str, default="toxicn")
    ap.add_argument("--split", type=str, default="dev")
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    ap.add_argument("--threshold_mode", type=str, default="calibrated")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--thr_fpr_max", type=float, default=1.0)
    ap.add_argument("--thr_objective", type=str, default="macro_f1")
    ap.add_argument("--dataset_for_threshold", type=str, default="toxicn")
    ap.add_argument("--toxicn_train_json", type=str, default="dataset/ToxiCN/train.json")
    ap.add_argument("--toxicn_test_json", type=str, default="dataset/ToxiCN/test.json")
    ap.add_argument("--toxicn_csv", type=str, default="dataset/ToxiCN/ToxiCN_1.0.csv")
    ap.add_argument("--pretrained_dir", type=str, default="")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    predictor = load_offensive_predictor(
        run_dir=args.run_dir,
        device=str(args.device),
        dtype=str(args.dtype),
        dataset_for_threshold=str(args.dataset_for_threshold),
        pretrained_dir=str(args.pretrained_dir) if str(args.pretrained_dir).strip() else None,
    )

    report: Dict[str, Any] = {"run_dir": str(Path(args.run_dir).expanduser().resolve()), "split": str(args.split), "datasets": {}}
    for ds in [x.strip() for x in str(args.datasets).split(",") if x.strip()]:
        items = load_items_for_dataset(root, ds, args.split, args.toxicn_train_json, args.toxicn_test_json, args.toxicn_csv)
        if int(args.max_items) > 0:
            items = items[: int(args.max_items)]
        texts = [t for t, _ in items]
        gold = [int(y) for _, y in items]

        p = predictor.predict_proba(texts, batch_size=int(args.batch_size))
        mode = str(args.threshold_mode).strip().lower()
        thr = float(args.threshold)
        cal = None
        if mode in {"search", "grid"}:
            cal = search_best_threshold(
                p,
                gold,
                thr_min=float(args.thr_min),
                thr_max=float(args.thr_max),
                thr_step=float(args.thr_step),
                fpr_max=float(args.thr_fpr_max),
                objective=str(args.thr_objective),
            )
            thr = float(cal["threshold"])
        elif mode in {"calibrated", "cal"} and predictor.calibrated_threshold is not None:
            thr = float(predictor.calibrated_threshold)
        elif mode in {"argmax", "0.5"}:
            thr = 0.5

        pred = [1 if float(x) >= float(thr) else 0 for x in p]
        tp, tn, fp, fn = compute_counts(pred, gold)
        ccdc = compute_ccdc_metrics_from_counts(tp, tn, fp, fn)
        flat = flatten_ccdc_metrics(ccdc)
        out: Dict[str, Any] = {
            "n": len(items),
            "threshold_mode": mode,
            "threshold": float(thr),
            "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "ccdc": ccdc,
            "flat": flat,
        }
        if cal is not None:
            out["threshold_search"] = cal
        report["datasets"][ds] = out

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

