"""Local viewer for Spread worlds.

Serves ui/index.html and proxies a small set of read-mostly API routes, so the
team key stays on this machine and never reaches the browser.

    python ui/server.py            # then open http://localhost:8765
    python ui/server.py --port 9000
"""
import argparse
import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def load_env() -> None:
    """Read KEY=VALUE lines from the project's .env without overriding the shell."""
    env = PROJECT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()
SPREAD_URL = os.environ.get("SPREAD_URL", "https://spread.bahnexpress.com").rstrip("/")
SPREAD_KEY = os.environ.get("SPREAD_KEY", "")

WORLD = r"/v1/worlds/[A-Za-z0-9_-]+"
# (method, path pattern) pairs the browser may call. Only /human sends actions.
ALLOWED = [
    ("GET", r"/v1/map"),
    ("GET", r"/v1/worlds"),
    ("POST", r"/v1/worlds"),
    ("GET", WORLD),
    ("DELETE", WORLD),
    ("GET", WORLD + r"/(events|drivers|score|reveal)"),
    ("POST", WORLD + r"/advance"),
    ("POST", WORLD + r"/actions"),
    ("GET", r"/v1/final"),
    ("GET", r"/v1/leaderboards/practice"),
]


def allowed(method: str, path: str) -> bool:
    return any(m == method and re.fullmatch(p, path) for m, p in ALLOWED)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        pass

    def send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error(self, status: int, message: str) -> None:
        self.send(status, json.dumps({"error": message}).encode())

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path in ("/human", "/human/"):
            return self.send(200, (ROOT / "human.html").read_bytes(), "text/html; charset=utf-8")
        if self.path.startswith("/human-log/"):
            world = re.sub(r"[^A-Za-z0-9_-]", "", self.path.rsplit("/", 1)[-1])
            f = PROJECT / "data" / "human" / f"{world}.jsonl"
            lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
            return self.send(200, json.dumps(lines).encode())
        if self.path == "/config":
            return self.send(200, json.dumps({"url": SPREAD_URL, "hasKey": bool(SPREAD_KEY)}).encode())
        self.proxy("GET")

    def do_POST(self):
        if self.path == "/human-log":
            return self.save_log()
        self.proxy("POST")

    def save_log(self) -> None:
        """Append one decision of the human dispatcher to data/human/<world>.jsonl."""
        length = int(self.headers.get("Content-Length") or 0)
        entry = json.loads(self.rfile.read(length) or b"{}")
        world = re.sub(r"[^A-Za-z0-9_-]", "", str(entry.get("worldId", "unknown"))) or "unknown"
        folder = PROJECT / "data" / "human"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{world}.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.send(200, b'{"ok": true}')

    def do_DELETE(self):
        self.proxy("DELETE")

    def proxy(self, method: str) -> None:
        if not self.path.startswith("/api/"):
            return self.error(404, "Not found")
        target = self.path[len("/api"):]
        path = target.split("?", 1)[0]
        if not allowed(method, path):
            return self.error(403, f"{method} {path} is not allowed by the viewer")
        body = None
        if method == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw or b"{}")
            body = json.dumps(data).encode()
        req = urllib.request.Request(SPREAD_URL + target, data=body, method=method)
        if SPREAD_KEY:
            req.add_header("Authorization", f"Bearer {SPREAD_KEY}")
        if body is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                self.send(resp.status, resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            payload = e.read() or json.dumps({"error": f"HTTP {e.code}"}).encode()
            self.send(e.code, payload)
        except (urllib.error.URLError, TimeoutError) as e:
            self.error(502, f"Spread server unreachable: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Spread viewer")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not SPREAD_KEY:
        print("Warning: SPREAD_KEY is not set (.env or shell). Only the map will load.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Spread viewer on http://localhost:{args.port}  (server: {SPREAD_URL})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
