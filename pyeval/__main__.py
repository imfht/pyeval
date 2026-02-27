"""CLI entry point for PyEval: python -m pyeval"""

import argparse
import json
import sys

from .config import EvalConfig, load_config, merge_cli_args
from .problems.loader import load_bugfix_problems, load_problems
from .reporter import generate_markdown_report, print_terminal_report
from .runner import run_evaluation
from .scorer import compute_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pyeval",
        description="PyEval: Python coding ability evaluation framework",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--api-base", type=str, default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for generation")
    parser.add_argument("--timeout", type=int, default=None, help="API request timeout (seconds)")
    parser.add_argument("--sandbox-timeout", type=int, default=None, help="Sandbox execution timeout (seconds)")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["standard", "bugfix", "multiturn"],
        help="Evaluation mode: standard, bugfix, or multiturn",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help="Max attempts for multiturn mode (default: 3)",
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Comma-separated categories to evaluate (e.g. algorithms,oop)",
    )
    parser.add_argument(
        "--difficulties", type=str, default=None,
        help="Comma-separated difficulties (easy,medium,hard)",
    )
    parser.add_argument(
        "--problem-ids", type=str, default=None,
        help="Comma-separated problem IDs to evaluate",
    )
    parser.add_argument(
        "--extra-body", type=str, default=None,
        help='Extra JSON body params for API (e.g. \'{"enable_thinking": false}\')',
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--dry-run", action="store_true", help="Run reference solutions to validate tests")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--problems-dir", type=str, default=None, help="Path to problems bank directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Map CLI args to config fields
    cli_overrides = {
        "api_base": args.api_base,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "sandbox_timeout": args.sandbox_timeout,
        "mode": args.mode,
        "max_attempts": args.max_attempts,
        "categories": args.categories.split(",") if args.categories else None,
        "difficulties": args.difficulties.split(",") if args.difficulties else None,
        "problem_ids": args.problem_ids.split(",") if args.problem_ids else None,
        "extra_body": json.loads(args.extra_body) if args.extra_body else None,
        "verbose": args.verbose or None,
        "dry_run": args.dry_run or None,
        "output_dir": args.output_dir,
        "problems_dir": args.problems_dir,
    }
    config = merge_cli_args(config, cli_overrides)

    # Load problems
    try:
        if config.mode == "bugfix":
            problems = load_bugfix_problems(
                config.problems_dir,
                difficulties=config.difficulties or None,
                problem_ids=config.problem_ids or None,
            )
        else:
            problems = load_problems(
                config.problems_dir,
                categories=config.categories or None,
                difficulties=config.difficulties or None,
                problem_ids=config.problem_ids or None,
            )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not problems:
        print("No problems matched the given filters.", file=sys.stderr)
        sys.exit(1)

    mode_labels = {"standard": "standard", "bugfix": "bugfix", "multiturn": "multiturn"}
    mode_str = mode_labels.get(config.mode, config.mode)
    if config.dry_run:
        mode_display = f"dry-run (reference solutions) [{mode_str}]"
    else:
        mode_display = f"model={config.model} [{mode_str}]"
    print(f"PyEval: Evaluating {len(problems)} problems [{mode_display}]")
    if config.verbose:
        print(f"  API base: {config.api_base}")
        print(f"  Mode: {config.mode}")
        if config.mode == "multiturn":
            print(f"  Max attempts: {config.max_attempts}")
        print(f"  Categories: {config.categories or 'all'}")
        print(f"  Difficulties: {config.difficulties or 'all'}")

    # Progress callback
    def on_progress(result, done, total):
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{done}/{total}] {result.problem.id}: {status}")

    callback = on_progress if config.verbose else None

    # Run evaluation
    results = run_evaluation(problems, config, progress_callback=callback)

    # Score
    scores = compute_scores(results, config.model, mode=config.mode)

    # Report
    print_terminal_report(scores, results, verbose=config.verbose, mode=config.mode)
    md_path = generate_markdown_report(scores, results, config.output_dir, mode=config.mode)
    print(f"Report saved to: {md_path}")


if __name__ == "__main__":
    main()
