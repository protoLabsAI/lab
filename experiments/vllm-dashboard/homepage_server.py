"""Lightweight homepage server with reverse proxy to local services.

Serves the homepage on port 80 and proxies to backend services.
"""

from __future__ import annotations

import json
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

HOMEPAGE = Path(__file__).parent / "homepage.html"
PORT = 80

# Service proxy routes
PROXIES = {
    "/dashboard/": "http://127.0.0.1:7860/",
}


def get_gpu_json() -> str:
    """Get GPU info as JSON."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": float(parts[2]),
                    "memory_total_mb": float(parts[3]),
                    "power_draw_w": float(parts[4]),
                    "power_limit_w": float(parts[5]),
                    "temp_c": float(parts[6]),
                })
        return json.dumps(gpus)
    except Exception as e:
        return json.dumps([])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HOMEPAGE.read_bytes())

        elif self.path == "/api/gpu":
            data = get_gpu_json()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        else:
            # Try proxy
            for prefix, target in PROXIES.items():
                if self.path.startswith(prefix):
                    return self.proxy_request(prefix, target)

            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not found")

    def proxy_request(self, prefix: str, target: str):
        """Simple proxy — redirect to the actual service port."""
        # For simplicity, redirect instead of full proxy
        # (Gradio/WebUI need websockets which are hard to proxy in stdlib)
        remainder = self.path[len(prefix):]
        host = self.headers.get("Host", "protolabs").split(":")[0]
        port = target.split(":")[-1].rstrip("/")
        redirect_url = f"http://{host}:{port}/{remainder}"
        self.send_response(302)
        self.send_header("Location", redirect_url)
        self.end_headers()

    def log_message(self, format, *args):
        # Quiet logging
        pass


def main():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Homepage serving on http://0.0.0.0:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()


if __name__ == "__main__":
    main()
