from __future__ import annotations

import argparse

from inspect_ai import Task, eval, task
from inspect_evals.chembench.chembench import chembench

try:
    from aifchem_shim_eval.client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL
    from aifchem_shim_eval.inspect_helpers import (
        DEFAULT_LOG_DIR,
        ensure_log_dir,
        inspect_model_name,
        load_repo_dotenv,
        print_eval_log_summary,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    from client import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL
    from inspect_helpers import (
        DEFAULT_LOG_DIR,
        ensure_log_dir,
        inspect_model_name,
        load_repo_dotenv,
        print_eval_log_summary,
    )


DEFAULT_TASK_NAME = "toxicity_and_safety"


@task
def aifchem_chembench(task_name: str = DEFAULT_TASK_NAME) -> Task:
    return chembench(task_name=task_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ChemBench against the local AIFChem shim via Inspect's Python API."
    )
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME, help="ChemBench subset name")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to run; omit to evaluate the full dataset",
    )
    parser.add_argument(
        "--base-url",
        default=f"{DEFAULT_BASE_URL}/v1",
        help="OpenAI-compatible base URL for Inspect",
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the local shim")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed by the local shim")
    parser.add_argument("--max-samples", type=int, default=1, help="Max samples to run in parallel")
    parser.add_argument("--max-tasks", type=int, default=1, help="Max tasks to run in parallel")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Inspect log output directory")
    return parser.parse_args()


def run_chembench(args: argparse.Namespace):
    log_dir = ensure_log_dir(args.log_dir)
    return eval(
        aifchem_chembench(task_name=args.task_name),
        model=inspect_model_name(args.model),
        model_base_url=args.base_url,
        model_args={"api_key": args.api_key},
        limit=args.limit,
        max_samples=args.max_samples,
        max_tasks=args.max_tasks,
        display="plain",
        log_dir=str(log_dir),
        fail_on_error=True,
    )


def main() -> int:
    load_repo_dotenv()
    args = parse_args()
    logs = run_chembench(args)

    for log in logs:
        print_eval_log_summary(log)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
