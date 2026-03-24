from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:4010"
DEFAULT_API_KEY = "sk-local-test"
DEFAULT_MODEL = "aifchem-agent"


@dataclass(frozen=True)
class ShimClient:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL

    def build_payload(self, user_text: str, *, stream: bool = False) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": user_text,
                }
            ],
        }

    def chat_completion(self, user_text: str, *, timeout: int = 120) -> str:
        payload = self.build_payload(user_text, stream=False)
        request = urllib.request.Request(
            url=self.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {details}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Request failed: {error}") from error

        parsed = json.loads(body)
        try:
            return parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"Unexpected response payload: {parsed!r}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal Python client for the local AIFChem shim.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Local shim API key placeholder")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to send")
    parser.add_argument("prompt", help="User prompt to send")
    args = parser.parse_args()

    client = ShimClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    print(client.chat_completion(args.prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
