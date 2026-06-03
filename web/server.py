"""Process Modeling Lab - Lightweight web server"""
import base64
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent
PORT = int(os.environ.get("PORT", 8503))

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".json": "application/json",
    ".png": "image/png",
}

jobs = {}  # job_id -> {"status": ..., "images": [...]}

EXPERIMENTS = {
    "regression": "単回帰 — 標準化傾きと相関係数の一致",
    "multicollinearity": "多重共線性とPLS — 係数安定性の比較",
    "jit": "JIT型ソフトセンサー — 局所モデルの効果",
    "graybox": "グレーボックスモデル — 物理 + 統計の融合",
    "transfer": "転移学習 — ドメイン差を超えた予測",
    "rawdata": "生データ診断 — 外れ値・ドリフト・張り付き検出",
}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            path = "/index.html"
        file_path = ROOT / path.lstrip("/")
        if file_path.is_file():
            ext = file_path.suffix.lower()
            mime = MIME.get(ext, "application/octet-stream")
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}") if length else {}

        if path == "/api/run":
            exp = body.get("exp", "")
            if exp not in EXPERIMENTS:
                self.send_json(400, {"error": "unknown experiment"})
                return
            job_id = f"{exp}_{id(body)}"
            jobs[job_id] = {"status": "running", "images": []}
            threading.Thread(target=run_exp, args=(job_id, exp), daemon=True).start()
            self.send_json(200, {"job_id": job_id})

        elif path == "/api/status":
            job_id = body.get("job_id", "")
            self.send_json(200, jobs.get(job_id, {"status": "not_found"}))

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_exp(job_id, exp):
    try:
        outdir = tempfile.mkdtemp()
        result = subprocess.run(
            [sys.executable, str(ROOT / "run_experiment.py"), exp, outdir],
            capture_output=True, timeout=120,
            cwd=str(ROOT.parent),
        )
        images = []
        for png in sorted(Path(outdir).glob("*.png")):
            data = base64.b64encode(png.read_bytes()).decode()
            images.append({"name": png.stem, "data": f"data:image/png;base64,{data}"})
        stderr = result.stderr.decode()[-500:] if result.returncode else ""
        jobs[job_id] = {"status": "done", "images": images, "stderr": stderr}
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e), "images": []}


if __name__ == "__main__":
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Process Modeling Lab at http://127.0.0.1:{PORT}")
    server.serve_forever()
