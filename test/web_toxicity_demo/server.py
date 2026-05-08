from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from train.offensive_infer import OffensivePredictor, load_offensive_predictor
from mamba_ssm.rag import AgentClient, FusionWeights, RagRequest, RagRetriever, fuse_scores


def _json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _parse_fusion_weights(data: Dict[str, Any], defaults: FusionWeights) -> FusionWeights:
    cfg = data.get("fusion", {}) if isinstance(data.get("fusion"), dict) else {}

    def _pick(name: str, fallback: float) -> float:
        if name in cfg:
            return float(cfg[name])
        alt = f"w_{name}"
        if alt in cfg:
            return float(cfg[alt])
        return float(fallback)

    return FusionWeights(
        model=_pick("model", defaults.model),
        bm25=_pick("bm25", defaults.bm25),
        vector=_pick("vector", defaults.vector),
        rule=_pick("rule", defaults.rule),
    )


def _build_rag_context(rag_result: "RagQueryResult", max_chars: int) -> str:
    lines: List[str] = []
    for hit in rag_result.hits:
        lines.append(f"[{hit.doc_type}] {hit.title}: {hit.text_snippet}")
    if rag_result.rule_hits:
        lines.append("Rules matched: " + ", ".join(rag_result.rule_hits))
    if not lines:
        return ""
    text = "Retrieved evidence:\n" + "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _rag_to_dict(rag_result: "RagQueryResult") -> Dict[str, Any]:
    return {
        "bm25_score": float(rag_result.bm25_score),
        "vector_score": float(rag_result.vector_score),
        "rule_score": float(rag_result.rule_score),
        "rule_hits": list(rag_result.rule_hits),
        "hits": [
            {
                "doc_id": h.doc_id,
                "title": h.title,
                "doc_type": h.doc_type,
                "score_bm25": float(h.score_bm25),
                "score_vec": float(h.score_vec),
                "score_fused": float(h.score_fused),
                "text_snippet": h.text_snippet,
                "source": h.source,
            }
            for h in rag_result.hits
        ],
        "meta": dict(rag_result.meta),
        "query": rag_result.query,
    }


def _parse_mode(data: Dict[str, Any]) -> str:
    mode = str(data.get("mode", "ocr_asr")).strip().lower()
    return mode or "ocr_asr"


def _resolve_pipeline_flags(mode: str, *, rag_enabled: bool) -> Dict[str, bool]:
    if mode in {"baseline", "base"}:
        return {"ocr_asr": False, "rag": False}
    if mode in {"ocr_asr_rag", "ocr-asr-rag", "rag", "ocr_asr+rag", "ocr+rag"}:
        return {"ocr_asr": True, "rag": bool(rag_enabled)}
    return {"ocr_asr": True, "rag": bool(rag_enabled)}


def _run_prediction(
    *,
    app: "App",
    predictor: OffensivePredictor,
    rag_retriever: Optional[RagRetriever],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    texts = data.get("texts", [])
    if not isinstance(texts, list):
        raise ValueError("texts 必须是数组")
    texts = [str(x) for x in texts]

    mode = _parse_mode(data)
    rag_requested = _as_bool(data.get("rag_enabled"), app.rag_enabled_default)
    flags = _resolve_pipeline_flags(mode, rag_enabled=rag_requested)
    enable_ocr_asr = bool(flags["ocr_asr"])
    rag_enabled = bool(flags["rag"]) and rag_retriever is not None
    rag_warning = None
    if bool(flags["rag"]) and rag_retriever is None:
        rag_warning = "RAG index not loaded or unavailable"

    rag_top_k = int(data.get("rag_top_k", app.rag_top_k))
    rag_min_score = float(data.get("rag_min_score", 0.0))
    rag_with_rules = _as_bool(data.get("rag_with_rules"), True)
    rag_context_max_chars = int(data.get("rag_context_max_chars", 480))
    rag_augment_input = _as_bool(data.get("rag_augment_input"), True)

    fusion_weights = _parse_fusion_weights(data, app.fusion_weights)
    fusion_threshold = float(data.get("fusion_threshold", app.fusion_threshold))

    image_b64 = data.get("image_b64", "")
    audio_b64 = data.get("audio_b64", "")

    image_path = None
    audio_path = None
    cleanup_paths: List[str] = []
    try:
        if image_b64:
            import base64
            import tempfile

            b64_str = image_b64.split(",")[-1]
            img_bytes = base64.b64decode(b64_str)
            img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img_temp.write(img_bytes)
            img_temp.close()
            image_path = img_temp.name
            cleanup_paths.append(image_path)
        if audio_b64:
            import base64
            import tempfile

            b64_str = audio_b64.split(",")[-1]
            aud_bytes = base64.b64decode(b64_str)
            aud_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            aud_temp.write(aud_bytes)
            aud_temp.close()
            audio_path = aud_temp.name
            cleanup_paths.append(audio_path)

        num_inferences = max(
            len([t for t in texts if t.strip()]),
            len(texts),
            1 if image_path else 0,
            1 if audio_path else 0,
        )
        if num_inferences == 0:
            raise ValueError("请输入至少一种模态数据。")

        display_texts = texts + [""] * (num_inferences - len(texts))
        images_list = [image_path] + [None] * (num_inferences - 1) if image_path else [None] * num_inferences
        audios_list = [audio_path] + [None] * (num_inferences - 1) if audio_path else [None] * num_inferences

        ocr_texts_list = [""] * num_inferences
        asr_texts_list = [""] * num_inferences
        if enable_ocr_asr:
            for i in range(num_inferences):
                img_file = images_list[i]
                aud_file = audios_list[i]
                extra_text = ""

                if img_file and app.ocr is not None:
                    try:
                        result = app.ocr.readtext(img_file)
                        ocr_texts = [res[1] for res in result]
                        if ocr_texts:
                            extracted_ocr = " ".join(ocr_texts)
                            ocr_texts_list[i] = extracted_ocr
                            extra_text += "。图片包含文字：" + extracted_ocr
                    except Exception as e:
                        print(f"OCR Error: {e}")

                if aud_file and app.asr is not None:
                    try:
                        result = app.asr.transcribe(aud_file, language="zh", initial_prompt="这是一段中文语音，请全部识别为简体中文。")
                        asr_text = result.get("text", "")
                        if asr_text:
                            asr_texts_list[i] = asr_text
                            extra_text += "。音频包含文字：" + asr_text
                    except Exception as e:
                        print(f"ASR Error: {e}")

                if extra_text:
                    display_texts[i] = display_texts[i] + extra_text

        rag_results: List[Optional[object]] = [None] * num_inferences
        rag_contexts: List[str] = [""] * num_inferences
        inference_texts = list(display_texts)
        if rag_enabled and rag_retriever is not None:
            for i in range(num_inferences):
                req = RagRequest(
                    query=display_texts[i],
                    top_k=rag_top_k,
                    with_rules=rag_with_rules,
                    max_snippet_chars=200,
                    min_score=rag_min_score,
                )
                rag_result = rag_retriever.query(req)
                rag_results[i] = rag_result
                rag_context = _build_rag_context(rag_result, rag_context_max_chars)
                rag_contexts[i] = rag_context
                if rag_context and rag_augment_input:
                    inference_texts[i] = display_texts[i] + "\n\n" + rag_context

        threshold_mode = str(data.get("threshold_mode", "calibrated"))
        threshold = float(data.get("threshold", 0.5))
        t0 = time.time()
        pred = predictor.predict(
            inference_texts,
            images=images_list,
            audios=audios_list,
            threshold_mode=threshold_mode,
            threshold=threshold,
        )
        dt = (time.time() - t0) * 1000.0

        results = []
        for i in range(num_inferences):
            model_score = float(pred.p_toxic[i])
            model_label = int(pred.labels[i])
            rag_result = rag_results[i]

            fusion_score = None
            fusion_label = None
            fusion_components = None
            final_label = model_label
            if rag_result is not None:
                fusion = fuse_scores(
                    model_score=model_score,
                    bm25_score=rag_result.bm25_score,
                    vector_score=rag_result.vector_score,
                    rule_score=rag_result.rule_score,
                    weights=fusion_weights,
                    threshold=fusion_threshold,
                )
                fusion_score = float(fusion.score)
                fusion_label = int(fusion.label)
                fusion_components = dict(fusion.components)
                final_label = fusion_label

            results.append({
                "text": display_texts[i],
                "model_text": inference_texts[i] if inference_texts[i] != display_texts[i] else "",
                "p_toxic": model_score,
                "label": int(final_label),
                "model_label": int(model_label),
                "fusion_score": fusion_score,
                "fusion_label": fusion_label,
                "fusion_components": fusion_components,
                "rag": _rag_to_dict(rag_result) if rag_result is not None else None,
                "rag_context": rag_contexts[i] if rag_contexts[i] else "",
                "has_text": bool(display_texts[i].strip()),
                "has_image": bool(images_list[i]),
                "has_audio": bool(audios_list[i]),
                "ocr_text": ocr_texts_list[i],
                "asr_text": asr_texts_list[i],
            })

        out: Dict[str, Any] = {
            "n": num_inferences,
            "mode": mode,
            "rag_enabled": bool(rag_enabled),
            "rag_warning": rag_warning,
            "threshold_mode": pred.threshold_mode,
            "threshold": pred.threshold,
            "fusion_threshold": float(fusion_threshold),
            "fusion_weights": {
                "model": float(fusion_weights.model),
                "bm25": float(fusion_weights.bm25),
                "vector": float(fusion_weights.vector),
                "rule": float(fusion_weights.rule),
            },
            "latency_ms": float(dt),
            "results": results,
        }
        internal = {
            "mode": mode,
            "rag_enabled": bool(rag_enabled),
            "rag_warning": rag_warning,
            "rag_results": rag_results,
            "display_texts": display_texts,
            "inference_texts": inference_texts,
            "fusion_threshold": float(fusion_threshold),
            "fusion_weights": fusion_weights,
        }
        return {"out": out, "internal": internal}
    finally:
        if cleanup_paths:
            import os

            for path in cleanup_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass


class App:
    def __init__(
        self,
        *,
        index_html: str,
        agent_html: str,
        run_dir: str,
        device: str,
        dtype: str,
        dataset_for_threshold: str,
        pretrained_dir: Optional[str],
        rag_index_dir: Optional[str],
        rag_device: str,
        rag_top_k: int,
        rag_enabled_default: bool,
        fusion_weights: FusionWeights,
        fusion_threshold: float,
    ):
        self.index_html = index_html
        self.agent_html = agent_html
        self.run_dir = str(run_dir)
        self.device = str(device)
        self.dtype = str(dtype)
        self.dataset_for_threshold = str(dataset_for_threshold)
        self.pretrained_dir = str(pretrained_dir) if pretrained_dir is not None else None
        self.rag_index_dir = str(rag_index_dir) if rag_index_dir else None
        self.rag_device = str(rag_device)
        self.rag_top_k = int(rag_top_k)
        self.rag_enabled_default = bool(rag_enabled_default)
        self.fusion_weights = fusion_weights
        self.fusion_threshold = float(fusion_threshold)

        # Load external reasoning tools on CPU or Device to help inference
        # Use easyocr instead of PaddleOCR due to paddle segfault issues
        try:
            import easyocr
            self.ocr = easyocr.Reader(['ch_sim', 'en'], gpu=(self.device=="cuda"))
        except ImportError:
            self.ocr = None
            print("easyocr is not installed, ignoring OCR.")

        try:
            import whisper
            # 升级为更大的 small 或 medium 模型，由于中文表现 base 较差
            self.asr = whisper.load_model("small", device=self.device)
        except ImportError:
            self.asr = None
            print("Whisper is not installed, ignoring ASR.")
        self._lock = threading.Lock()
        self.predictor: Optional[OffensivePredictor] = None
        self.rag_retriever: Optional[RagRetriever] = None
        self.agent = AgentClient.from_env()
        self.load_state: Dict[str, Any] = {
            "status": "loading",
            "run_dir": self.run_dir,
            "device": self.device,
            "dtype": self.dtype,
            "dataset_for_threshold": self.dataset_for_threshold,
            "rag_index_dir": self.rag_index_dir,
            "rag_status": "disabled" if not self.rag_index_dir else "loading",
            "rag_error": None,
            "error": None,
            "started_at": time.time(),
            "ready_at": None,
        }

    def start_loading(self) -> None:
        t = threading.Thread(target=self._load, daemon=True)
        t.start()

    def _load(self) -> None:
        try:
            pred = load_offensive_predictor(
                run_dir=self.run_dir,
                device=self.device,
                dtype=self.dtype,
                dataset_for_threshold=self.dataset_for_threshold,
                pretrained_dir=self.pretrained_dir,
            )
            rag = None
            rag_error = None
            if self.rag_index_dir:
                try:
                    rag = RagRetriever.load(self.rag_index_dir, device=self.rag_device)
                except Exception as e:
                    rag_error = str(e)
            with self._lock:
                self.predictor = pred
                self.rag_retriever = rag
                self.load_state["status"] = "ready"
                self.load_state["ready_at"] = time.time()
                if self.rag_index_dir:
                    self.load_state["rag_status"] = "ready" if rag is not None else "error"
                    self.load_state["rag_error"] = rag_error
        except Exception as e:
            with self._lock:
                self.predictor = None
                self.load_state["status"] = "error"
                self.load_state["error"] = str(e)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._send(200, b"", "text/plain; charset=utf-8")

        def do_GET(self) -> None:
            if self.path == "/" or self.path.startswith("/index.html"):
                body = app.index_html.encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
                return
            if self.path == "/agent" or self.path.startswith("/agent.html"):
                body = app.agent_html.encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
                return
            if self.path.startswith("/api/status"):
                with app._lock:
                    st = dict(app.load_state)
                    if st.get("ready_at") is not None:
                        try:
                            st["ready_in_ms"] = float(st["ready_at"]) * 1000.0 - float(st["started_at"]) * 1000.0
                        except Exception:
                            pass
                self._send(200, _json_bytes(st), "application/json; charset=utf-8")
                return
            self._send(404, b"not found\n", "text/plain; charset=utf-8")

        def do_POST(self) -> None:
            if self.path.startswith("/api/agent"):
                try:
                    with app._lock:
                        predictor = app.predictor
                        rag_retriever = app.rag_retriever
                        st = dict(app.load_state)
                    if predictor is None:
                        self._send(503, _json_bytes({"error": "模型仍在加载中", "status": st}), "application/json; charset=utf-8")
                        return

                    n = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(n)
                    data = json.loads(raw.decode("utf-8"))

                    message = str(data.get("message", "")).strip()
                    if message and not data.get("texts"):
                        data["texts"] = [message]
                    if not message:
                        texts = data.get("texts", [])
                        if isinstance(texts, list) and texts:
                            message = str(texts[0]).strip()

                    bundle = _run_prediction(app=app, predictor=predictor, rag_retriever=rag_retriever, data=data)
                    out = bundle["out"]
                    internal = bundle["internal"]

                    if not out.get("results"):
                        self._send(400, _json_bytes({"error": "缺少可分析的文本"}), "application/json; charset=utf-8")
                        return

                    result0 = out["results"][0]
                    rag_result = None
                    rag_results = internal.get("rag_results", [])
                    if rag_results:
                        rag_result = rag_results[0]

                    model_score = float(result0.get("p_toxic", 0.0))
                    fusion_score = result0.get("fusion_score")
                    fusion_label = result0.get("label", 0)
                    analysis = app.agent.analyze(
                        query=internal.get("display_texts", [message])[0] if internal.get("display_texts") else message,
                        rag=rag_result,
                        model_score=model_score,
                        fusion_score=float(fusion_score) if fusion_score is not None else model_score,
                        fusion_label=int(fusion_label),
                    )

                    payload = {
                        "analysis": analysis.content,
                        "agent_model": analysis.model,
                        "agent_latency_ms": analysis.latency_ms,
                        "agent_used_llm": analysis.used_llm,
                        "detection": result0,
                        "pipeline": {
                            "mode": out.get("mode"),
                            "rag_enabled": out.get("rag_enabled"),
                            "rag_warning": out.get("rag_warning"),
                            "fusion_threshold": out.get("fusion_threshold"),
                            "fusion_weights": out.get("fusion_weights"),
                        },
                    }
                    self._send(200, _json_bytes(payload), "application/json; charset=utf-8")
                except Exception as e:
                    self._send(400, _json_bytes({"error": str(e)}), "application/json; charset=utf-8")
                return

            if not self.path.startswith("/api/predict"):
                self._send(404, b"not found\n", "text/plain; charset=utf-8")
                return
            try:
                with app._lock:
                    predictor = app.predictor
                    rag_retriever = app.rag_retriever
                    st = dict(app.load_state)
                if predictor is None:
                    self._send(503, _json_bytes({"error": "模型仍在加载中", "status": st}), "application/json; charset=utf-8")
                    return
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                data = json.loads(raw.decode("utf-8"))

                bundle = _run_prediction(app=app, predictor=predictor, rag_retriever=rag_retriever, data=data)
                self._send(200, _json_bytes(bundle["out"]), "application/json; charset=utf-8")
            except Exception as e:
                self._send(400, _json_bytes({"error": str(e)}), "application/json; charset=utf-8")

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16")
    ap.add_argument("--dataset_for_threshold", type=str, default="toxicn")
    ap.add_argument("--pretrained_dir", type=str, default="")
    ap.add_argument("--rag_index_dir", type=str, default="")
    ap.add_argument("--rag_device", type=str, default="cpu")
    ap.add_argument("--rag_top_k", type=int, default=5)
    ap.add_argument("--rag_enabled_default", action="store_true")
    ap.add_argument("--fusion_threshold", type=float, default=0.5)
    ap.add_argument("--fusion_weight_model", type=float, default=0.6)
    ap.add_argument("--fusion_weight_bm25", type=float, default=0.2)
    ap.add_argument("--fusion_weight_vector", type=float, default=0.15)
    ap.add_argument("--fusion_weight_rule", type=float, default=0.05)
    args = ap.parse_args()

    index_html = (Path(__file__).resolve().parent / "index.html").read_text(encoding="utf-8")
    agent_html = (Path(__file__).resolve().parent / "agent.html").read_text(encoding="utf-8")
    fusion_weights = FusionWeights(
        model=float(args.fusion_weight_model),
        bm25=float(args.fusion_weight_bm25),
        vector=float(args.fusion_weight_vector),
        rule=float(args.fusion_weight_rule),
    )
    app = App(
        index_html=index_html,
        agent_html=agent_html,
        run_dir=str(args.run_dir),
        device=str(args.device),
        dtype=str(args.dtype),
        dataset_for_threshold=str(args.dataset_for_threshold),
        pretrained_dir=str(args.pretrained_dir) if str(args.pretrained_dir).strip() else None,
        rag_index_dir=str(args.rag_index_dir) if str(args.rag_index_dir).strip() else None,
        rag_device=str(args.rag_device),
        rag_top_k=int(args.rag_top_k),
        rag_enabled_default=bool(args.rag_enabled_default),
        fusion_weights=fusion_weights,
        fusion_threshold=float(args.fusion_threshold),
    )
    app.start_loading()
    handler = make_handler(app)
    server = HTTPServer((str(args.host), int(args.port)), handler)
    print(f"http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
