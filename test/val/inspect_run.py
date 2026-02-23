from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_calibrated_threshold(metrics: Dict[str, Any], *, dataset: str) -> Optional[float]:
    if "calibrated_threshold" in metrics:
        try:
            return float(metrics["calibrated_threshold"])
        except Exception:
            return None
    evals = metrics.get("eval", None)
    if isinstance(evals, dict):
        ds = evals.get(dataset, None)
        if isinstance(ds, dict) and "calibrated_threshold" in ds:
            try:
                return float(ds["calibrated_threshold"])
            except Exception:
                return None
    return None


def _pick(metrics: Dict[str, Any], keys) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in keys:
        if k in metrics:
            out[k] = metrics[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--dataset_for_threshold", type=str, default="toxicn")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    best_head = run_dir / "best_head.pt"
    if not best_head.exists():
        raise FileNotFoundError(f"缺少 best_head.pt: {best_head}")
    ckpt = torch.load(str(best_head), map_location="cpu")

    metrics = {}
    bm = run_dir / "best_metrics.json"
    if bm.exists():
        metrics = _read_json(bm)

    lora_cfg = ckpt.get("lora_cfg", None)
    lora_replaced = ckpt.get("lora_replaced", None)
    head_state = ckpt.get("head", {})
    hidden_dim = None
    if isinstance(head_state, dict) and "fc1.weight" in head_state:
        hidden_dim = int(head_state["fc1.weight"].shape[0])

    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "files": {
            "best_head_pt": str(best_head),
            "best_metrics_json": str(bm) if bm.exists() else None,
            "lora_adapter_pt": str(run_dir / "lora_adapter.pt") if (run_dir / "lora_adapter.pt").exists() else None,
            "lora_config_json": str(run_dir / "lora_config.json") if (run_dir / "lora_config.json").exists() else None,
            "full_model_pt": str(run_dir / "full_model.pt") if (run_dir / "full_model.pt").exists() else None,
            "backbone_converted_dir_txt": str(run_dir / "backbone_converted_dir.txt") if (run_dir / "backbone_converted_dir.txt").exists() else None,
        },
        "backbone_config": ckpt.get("config", None),
        "tokenizer": {
            "tokenizer_model_path": ckpt.get("tokenizer_model_path", None),
            "tokenizer_name_or_path": ckpt.get("tokenizer_name_or_path", None),
        },
        "max_length": ckpt.get("max_length", None),
        "head": {"hidden_dim_inferred": hidden_dim, "keys": sorted(list(head_state.keys()))[:8] if isinstance(head_state, dict) else None},
        "lora": {
            "enabled": bool(isinstance(lora_cfg, dict) and "lora" in ckpt),
            "lora_cfg": lora_cfg,
            "replaced_count": len(lora_replaced) if isinstance(lora_replaced, list) else None,
            "replaced_sample": lora_replaced[:10] if isinstance(lora_replaced, list) else None,
        },
        "best": _pick(metrics, ["best_metric", "best_score", "epoch"]),
        "calibrated_threshold": _extract_calibrated_threshold(metrics, dataset=str(args.dataset_for_threshold)),
    }

    if isinstance(metrics.get("eval", None), dict):
        evals = metrics["eval"]
        toxicn = evals.get("toxicn", None) if isinstance(evals, dict) else None
        if isinstance(toxicn, dict):
            out["eval_toxicn"] = _pick(
                toxicn,
                [
                    "macro_f1",
                    "fpr",
                    "non_toxic_f1",
                    "non_toxic_recall",
                    "calibrated_macro_f1",
                    "calibrated_fpr",
                    "calibrated_non_toxic_f1",
                    "calibrated_non_toxic_recall",
                    "calibrated_threshold",
                ],
            )
    else:
        out["eval"] = _pick(
            metrics,
            [
                "macro_f1",
                "fpr",
                "non_toxic_f1",
                "non_toxic_recall",
                "calibrated_macro_f1",
                "calibrated_fpr",
                "calibrated_non_toxic_f1",
                "calibrated_non_toxic_recall",
                "calibrated_threshold",
            ],
        )

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

