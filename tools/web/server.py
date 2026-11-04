"""Minimal HTTP server for the browser workbench (no third-party deps).

  python -m tools.web.server --port 8765

The server only shells into the repository's own modules; it never writes outside
`runs/web`.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import params as params_module
from . import pipeline

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
JOBS = {}
LOCK = threading.Lock()
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


def _job(job_id):
    with LOCK:
        return JOBS.get(job_id)


class Handler(BaseHTTPRequestHandler):
    server_version = "DPanelWorkbench/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name):
        target = (STATIC / name).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            self._send(404, {"error": "not found: " + name})
            return
        self._send(200, target.read_bytes(), TYPES.get(target.suffix, "application/octet-stream"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._static("index.html")
            return
        if path.startswith("/static/"):
            self._static(path[len("/static/"):])
            return
        if path == "/api/schema":
            self._send(200, dict(params=params_module.SCHEMA,
                                 defaults=params_module.normalise({})[0],
                                 reference=params_module.reference(params_module.normalise({})[0]),
                                 runs=str(pipeline.RUNS)))
            return
        if path == "/api/job":
            job = _job(parse_qs(parsed.query).get("id", [""])[0])
            if job is None:
                self._send(404, {"error": "unknown job"})
                return
            self._send(200, dict(id=job.run_dir.name, state=job.state, error=job.error,
                                 log=job.log, result=job.result,
                                 build=job.build, elapsed=time.time() - job.started))
            return
        self._send(404, {"error": "not found: " + path})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": "invalid JSON: %s" % exc})
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/build":
                job = pipeline.new_job()
                payload = pipeline.build(job, body.get("params") or {})
                with LOCK:
                    JOBS[job.run_dir.name] = job
                self._send(200, dict(job=job.run_dir.name, **payload))
                return
            if path == "/api/solve":
                job = _job(str(body.get("job", "")))
                if job is None:
                    self._send(404, {"error": "unknown job"})
                    return
                if job.state in ("meshing", "solving"):
                    self._send(409, {"error": "job already running"})
                    return
                job.cancel.clear()
                job.state = "meshing"
                thread = threading.Thread(target=pipeline.run_job, args=(job, body), daemon=True)
                thread.start()
                self._send(200, dict(job=job.run_dir.name, state=job.state))
                return
            if path == "/api/cancel":
                job = _job(str(body.get("job", "")))
                if job is None:
                    self._send(404, {"error": "unknown job"})
                    return
                job.cancel.set()
                self._send(200, dict(job=job.run_dir.name, state=job.state, cancelled=True))
                return
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc),
                             "traceback": traceback.format_exc()[-4000:]})
            return
        self._send(404, {"error": "not found: " + path})


def main(argv=None):
    parser = argparse.ArgumentParser(description="DPanel browser workbench")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    pipeline.RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("workbench: http://%s:%d  (runs: %s)" % (args.host, args.port, pipeline.RUNS), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
