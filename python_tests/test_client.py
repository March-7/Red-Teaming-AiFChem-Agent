from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aifchem_shim_eval import ShimClient


def test_build_payload_matches_openai_chat_shape() -> None:
    client = ShimClient(model="test-model")

    payload = client.build_payload("hello")

    assert payload == {
        "model": "test-model",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": "hello",
            }
        ],
    }


def test_chat_completion_reads_content_from_openai_response() -> None:
    capture: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers["Content-Length"])
            raw_body = self.rfile.read(content_length)
            capture["path"] = self.path
            capture["authorization"] = self.headers.get("Authorization")
            capture["body"] = json.loads(raw_body.decode("utf-8"))

            response = {
                "choices": [
                    {
                        "message": {
                            "content": "shim says hi",
                        }
                    }
                ]
            }
            encoded = json.dumps(response).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        client = ShimClient(base_url=f"http://{host}:{port}", api_key="sk-test", model="shim-model")

        response_text = client.chat_completion("hello from pytest")

        assert response_text == "shim says hi"
        assert capture["path"] == "/v1/chat/completions"
        assert capture["authorization"] == "Bearer sk-test"
        assert capture["body"] == {
            "model": "shim-model",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "hello from pytest",
                }
            ],
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
