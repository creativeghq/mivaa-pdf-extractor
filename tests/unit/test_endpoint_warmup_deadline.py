"""A dead or endlessly-redirecting endpoint must fail warm-up at the deadline."""

import importlib.util
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "app/services/pdf/endpoint_providers.py"
_spec = importlib.util.spec_from_file_location("endpoint_providers_under_test", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ModalEndpointProvider = _mod.ModalEndpointProvider


def _serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


class _Ok(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


class _RedirectForever(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(0.3)
        self.send_response(303)
        self.send_header("Location", self.path)
        self.end_headers()

    def log_message(self, *a):
        pass


class _Silent(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(5)

    def log_message(self, *a):
        pass


def test_healthy_endpoint_warms():
    server, url = _serve(_Ok)
    try:
        assert ModalEndpointProvider(base_url=url, warmup_timeout=5).warmup() is True
    finally:
        server.shutdown()


def test_redirect_loop_gives_up_at_the_deadline():
    server, url = _serve(_RedirectForever)
    try:
        start = time.time()
        assert ModalEndpointProvider(base_url=url, warmup_timeout=2).warmup() is False
        assert time.time() - start < 4
    finally:
        server.shutdown()


def test_silent_endpoint_gives_up_at_the_deadline():
    server, url = _serve(_Silent)
    try:
        start = time.time()
        assert ModalEndpointProvider(base_url=url, warmup_timeout=1).warmup() is False
        assert time.time() - start < 3
    finally:
        server.shutdown()
