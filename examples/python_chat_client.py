#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid


def call_chat_completion(base_url: str, api_key: str, model: str, messages: list[dict], session_id: str | None = None) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
    }
    if session_id:
        payload["metadata"] = {"session_id": session_id}

    request = urllib.request.Request(
        url=base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Request failed: {error}") from error

    parsed = json.loads(body)
    return parsed["choices"][0]["message"]["content"]


def repl(base_url: str, api_key: str, model: str, session_id: str | None) -> int:
    print(f"Connected to {base_url}/v1/chat/completions")
    print("Type your message and press Enter. Type 'exit' to quit.")
    if session_id:
        print(f"Using session_id={session_id}")

    messages: list[dict] = []

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            return 0

        messages.append({"role": "user", "content": user_text})
        try:
            content = call_chat_completion(base_url, api_key, model, messages, session_id=session_id)
        except Exception as error:
            messages.pop()
            print(f"\nError> {error}")
            continue

        messages.append({"role": "assistant", "content": content})
        print("\nAgent>")
        print(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple OpenAI-compatible chat client for the local AIFChem shim.")
    parser.add_argument("--base-url", default="http://127.0.0.1:4010", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="sk-local-test", help="Local shim API key placeholder")
    parser.add_argument("--model", default="aifchem-agent", help="Model name to send")
    parser.add_argument("--session-id", help="Optional shim session id for thread reuse")
    parser.add_argument("--once", help="Send one prompt and exit")
    args = parser.parse_args()

    if args.once:
        print(
            call_chat_completion(
                args.base_url,
                args.api_key,
                args.model,
                [{"role": "user", "content": args.once}],
                session_id=args.session_id,
            )
        )
        return 0

    session_id = args.session_id or f"example-session-{uuid.uuid4()}"
    return repl(args.base_url, args.api_key, args.model, session_id)


if __name__ == "__main__":
    sys.exit(main())
