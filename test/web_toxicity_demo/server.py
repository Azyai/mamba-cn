from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from train.offensive_infer import OffensivePredictor, load_offensive_predictor


def _json_bytes(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


class App:
    def __init__(
        self,
        *,
        index_html: str,
        run_dir: str,
        device: str,
        dtype: str,
        dataset_for_threshold: str,
        pretrained_dir: Optional[str],
    ):
        self.index_html = index_html
        self.run_dir = str(run_dir)
        self.device = str(device)
        self.dtype = str(dtype)
        self.dataset_for_threshold = str(dataset_for_threshold)
        self.pretrained_dir = str(pretrained_dir) if pretrained_dir is not None else None

        # Load external reasoning tools on CPU or Device to help inference
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=(self.device=="cuda"))
        except ImportError:
            self.ocr = None
            print("PaddleOCR is not installed, ignoring OCR.")

        try:
            import whisper
            self.asr = whisper.load_model("base", device=self.device)
        except ImportError:
            self.asr = None
            print("Whisper is not installed, ignoring ASR.")
        self._lock = threading.Lock()
        self.predictor: Optional[OffensivePredictor] = None
        self.load_state: Dict[str, Any] = {
            "status": "loading",
            "run_dir": self.run_dir,
            "device": self.device,
            "dtype": self.dtype,
            "dataset_for_threshold": self.dataset_for_threshold,
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
            with self._lock:
                self.predictor = pred
                self.load_state["status"] = "ready"
                self.load_state["ready_at"] = time.time()
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
            if not self.path.startswith("/api/predict"):
                self._send(404, b"not found\n", "text/plain; charset=utf-8")
                return
            try:
                with app._lock:
                    predictor = app.predictor
                    st = dict(app.load_state)
                if predictor is None:
                    self._send(503, _json_bytes({"error": "模型仍在加载中", "status": st}), "application/json; charset=utf-8")
                    return
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                data = json.loads(raw.decode("utf-8"))
                texts = data.get("texts", [])
                if not isinstance(texts, list):
                    raise ValueError("texts 必须是数组")
                texts = [str(x) for x in texts]
                
                image_b64 = data.get("image_b64", "")
                audio_b64 = data.get("audio_b64", "")
                
                image_path = None
                img_temp = None
                if image_b64:
                    import tempfile
                    import base64
                    try:
                        b64_str = image_b64.split(",")[-1]
                        img_bytes = base64.b64decode(b64_str)
                        img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        img_temp.write(img_bytes)
                        img_temp.close()
                        image_path = img_temp.name
                    except Exception as e:
                        pass
                
                audio_path = None
                aud_temp = None
                if audio_b64:
                    import tempfile
                    import base64
                    try:
                        b64_str = audio_b64.split(",")[-1]
                        aud_bytes = base64.b64decode(b64_str)
                        aud_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        aud_temp.write(aud_bytes)
                        aud_temp.close()
                        audio_path = aud_temp.name
                    except:
                        pass

                num_inferences = max(len([t for t in texts if t.strip()]), len(texts), 1 if image_path else 0, 1 if audio_path else 0)
                if num_inferences == 0:
                    raise ValueError("请输入至少一种模态数据。")

                final_texts = texts + [""] * (num_inferences - len(texts))
                
                images_list = [image_path] + [None] * (num_inferences - 1) if image_path else [None] * num_inferences
                audios_list = [audio_path] + [None] * (num_inferences - 1) if audio_path else [None] * num_inferences
                
                # Apply OCR & ASR processing as textual supplements during inference time
                for i in range(num_inferences):
                    img_file = images_list[i]
                    aud_file = audios_list[i]
                    extra_text = ""
                    
                    if img_file and app.ocr is not None:
                        try:
                            result = app.ocr.ocr(img_file, cls=True)
                            ocr_texts = [line[1][0] for res in result[0] if res is not None and result[0] is not None]
                            if ocr_texts:
                                extra_text += "。图片包含文字：" + " ".join(ocr_texts)
                        except Exception as e:
                            print(f"OCR Error: {e}")
                            
                    if aud_file and app.asr is not None:
                        try:
                            result = app.asr.transcribe(aud_file)
                            asr_text = result.get("text", "")
                            if asr_text:
                                extra_text += "。音频包含文字：" + asr_text
                        except Exception as e:
                            print(f"ASR Error: {e}")
                            
                    if extra_text:
                        final_texts[i] = final_texts[i] + extra_text
                
                threshold_mode = str(data.get("threshold_mode", "calibrated"))
                threshold = float(data.get("threshold", 0.5))
                t0 = time.time()
                pred = predictor.predict(final_texts, images=images_list, audios=audios_list, threshold_mode=threshold_mode, threshold=threshold)
                dt = (time.time() - t0) * 1000.0
                
                results = []
                for i in range(num_inferences):
                    t = final_texts[i]
                    p = float(pred.p_toxic[i])
                    y = int(pred.labels[i])
                    results.append({
                        "text": t,
                        "p_toxic": p,
                        "label": y,
                        "has_text": bool(t.strip()),
                        "has_image": bool(images_list[i]),
                        "has_audio": bool(audios_list[i])
                    })
                
                out: Dict[str, Any] = {
                    "n": num_inferences,
                    "threshold_mode": pred.threshold_mode,
                    "threshold": pred.threshold,
                    "latency_ms": float(dt),
                    "results": results,
                }
                self._send(200, _json_bytes(out), "application/json; charset=utf-8")
                
                if image_path:
                    import os
                    try:
                        os.remove(image_path)
                    except:
                        pass
                if audio_path:
                    import os
                    try:
                        os.remove(audio_path)
                    except:
                        pass
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
    args = ap.parse_args()

    index_html = (Path(__file__).resolve().parent / "index.html").read_text(encoding="utf-8")
    app = App(
        index_html=index_html,
        run_dir=str(args.run_dir),
        device=str(args.device),
        dtype=str(args.dtype),
        dataset_for_threshold=str(args.dataset_for_threshold),
        pretrained_dir=str(args.pretrained_dir) if str(args.pretrained_dir).strip() else None,
    )
    app.start_loading()
    handler = make_handler(app)
    server = HTTPServer((str(args.host), int(args.port)), handler)
    print(f"http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
