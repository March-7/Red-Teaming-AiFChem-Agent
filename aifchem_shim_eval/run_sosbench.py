from __future__ import annotations

import argparse

from inspect_ai import Task, eval, task
from inspect_evals.sosbench.sosbench import sosbench

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


DEFAULT_DOMAIN = "chemistry"
DEFAULT_GRADER_MODEL = "openai-api/deepseek/deepseek-chat"


@task
def aifchem_sosbench(domain: str = DEFAULT_DOMAIN) -> Task:
    return sosbench(domain=domain)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SOSBench against the local AIFChem shim via Inspect's Python API."
    )
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="SOSBench domain")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to run; omit to evaluate the full dataset",
    )
    parser.add_argument(
        "--sample-id",
        dest="sample_ids",
        action="append",
        help="Specific sample id to include; repeat the flag to provide multiple ids",
    )
    parser.add_argument(
        "--base-url",
        default=f"{DEFAULT_BASE_URL}/v1",
        help="OpenAI-compatible base URL for Inspect",
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key for the local shim")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed by the local shim")
    parser.add_argument(
        "--grader-model",
        default=DEFAULT_GRADER_MODEL,
        help="Inspect grader model name",
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Generation max tokens for tested model")
    parser.add_argument("--timeout", type=float, default=25.0, help="Per-request timeout for the tested model")
    parser.add_argument("--time-limit", type=int, default=25, help="Per-sample wall clock time limit")
    parser.add_argument("--working-limit", type=int, default=25, help="Per-sample working time limit")
    parser.add_argument("--max-samples", type=int, default=1, help="Max samples to run in parallel")
    parser.add_argument("--max-tasks", type=int, default=1, help="Max tasks to run in parallel")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="Inspect log output directory")
    return parser.parse_args()


def run_sosbench(args: argparse.Namespace):
    log_dir = ensure_log_dir(args.log_dir)
    return eval(
        aifchem_sosbench(domain=args.domain),
        model=inspect_model_name(args.model),
        model_base_url=args.base_url,
        model_args={"api_key": args.api_key, "timeout": args.timeout},
        model_roles={"grader": args.grader_model},
        limit=args.limit,
        sample_id=args.sample_ids,
        max_tokens=args.max_tokens,
        max_samples=args.max_samples,
        max_tasks=args.max_tasks,
        display="plain",
        log_dir=str(log_dir),
        time_limit=args.time_limit,
        working_limit=args.working_limit,
        fail_on_error=False,
        continue_on_fail=True,
    )


def main() -> int:
    load_repo_dotenv()
    args = parse_args()
    logs = run_sosbench(args)

    for log in logs:
        print_eval_log_summary(log, grader=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
