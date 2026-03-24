from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience import
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = "logs"
DEFAULT_PROVIDER = "local"


def load_repo_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv(REPO_ROOT / ".env")


def inspect_model_name(model: str, provider: str = DEFAULT_PROVIDER) -> str:
    return f"openai-api/{provider}/{model}"


def ensure_log_dir(log_dir: str | Path = DEFAULT_LOG_DIR) -> Path:
    path = Path(log_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_eval_log_summary(log: Any, *, grader: bool = False) -> None:
    print(f"status={log.status}")
    print(f"task={log.eval.task}")
    print(f"model={log.eval.model}")
    if grader:
        print(f"grader={log.eval.model_roles['grader'].model}")
    if log.results:
        print(f"samples={log.results.total_samples}")
        print(f"completed={log.results.completed_samples}")
    print(f"log={Path(log.location)}")
