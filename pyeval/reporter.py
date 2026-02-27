"""Terminal table and Markdown report generation."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .runner import MultiturnResult, ProblemResult
from .scorer import ATTEMPT_WEIGHTS, CATEGORY_DISPLAY, EvalScore, MultiturnEvalScore


DIFFICULTIES = ["easy", "medium", "hard"]
# Ordered categories for display
CATEGORY_ORDER = [
    "basic_syntax", "data_structures", "algorithms", "stdlib", "oop",
    "exceptions", "file_io", "string_processing", "functional", "concurrency",
    "bugfix",
]


def _fmt_score(passed: int, total: int) -> str:
    if total == 0:
        return "-"
    return f"{passed}/{total}"


def _ordered_cats(score):
    """Get ordered category list from score."""
    ordered = [c for c in CATEGORY_ORDER if c in score.categories]
    for c in score.categories:
        if c not in ordered:
            ordered.append(c)
    return ordered


def _total_by_diff(score):
    """Compute totals by difficulty across all categories."""
    result = {}
    for diff in DIFFICULTIES:
        p = sum(
            score.categories[c].by_difficulty.get(diff, type("D", (), {"passed": 0, "total": 0})).passed
            for c in score.categories
        )
        t = sum(
            score.categories[c].by_difficulty.get(diff, type("D", (), {"passed": 0, "total": 0})).total
            for c in score.categories
        )
        result[diff] = (p, t)
    return result


def print_terminal_report(score: EvalScore | MultiturnEvalScore, results: list, verbose: bool = False, mode: str = "standard"):
    """Print a formatted report to the terminal."""
    sep = "=" * 60
    dash = "-" * 60

    mode_label = {"standard": "Standard", "bugfix": "Bug Fix", "multiturn": "Multi-turn"}
    print(f"\n{sep}")
    print(f"  PyEval Results: {score.model} [{mode_label.get(mode, mode)}]")
    print(sep)
    print(f"{'Category':<22} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Total':>8}")
    print(dash)

    ordered_cats = _ordered_cats(score)

    for cat in ordered_cats:
        cat_score = score.categories[cat]
        display = cat_score.display_name
        cols = []
        for diff in DIFFICULTIES:
            ds = cat_score.by_difficulty.get(diff)
            if ds:
                cols.append(_fmt_score(ds.passed, ds.total))
            else:
                cols.append("-")
        total = _fmt_score(cat_score.total_passed, cat_score.total_count)
        print(f"{display:<22} {cols[0]:>8} {cols[1]:>8} {cols[2]:>8} {total:>8}")

    print(dash)

    # Totals row
    tbd = _total_by_diff(score)
    cols = [_fmt_score(tbd[d][0], tbd[d][1]) for d in DIFFICULTIES]
    total_passed = sum(c.total_passed for c in score.categories.values())
    total_count = sum(c.total_count for c in score.categories.values())
    total_str = _fmt_score(total_passed, total_count)
    print(f"{'TOTAL':<22} {cols[0]:>8} {cols[1]:>8} {cols[2]:>8} {total_str:>8}")

    print()

    if mode == "multiturn" and isinstance(score, MultiturnEvalScore):
        print(f"Strict Pass Rate (1st try):    {score.strict_pass_rate * 100:5.1f}%")
        print(f"With Retries Pass Rate:        {score.pass_rate * 100:5.1f}%")
        print(f"Weighted Score (attempt-adj):  {score.weighted_score * 100:5.1f}%")
        print()
        print("  Attempt Distribution:")
        max_attempt = max(score.attempt_distribution.keys()) if score.attempt_distribution else 0
        for attempt_num in range(1, max(max_attempt + 1, 4)):
            w = ATTEMPT_WEIGHTS.get(attempt_num, 0.0)
            count = score.attempt_distribution.get(attempt_num, 0)
            print(f"    Attempt {attempt_num} (weight={w}): {count} passed")
    else:
        pass_rate = total_passed / total_count if total_count > 0 else 0.0
        print(f"Overall Pass Rate: {pass_rate * 100:5.1f}%")
        print(f"Weighted Score:    {score.weighted_score * 100:5.1f}%  (easy=1x, medium=2x, hard=3x)")

    print(sep)

    if verbose:
        print("\nDetailed Results:")
        print(dash)
        for r in sorted(results, key=lambda x: x.problem.id):
            if mode == "multiturn" and isinstance(r, MultiturnResult):
                if r.passed:
                    status = f"PASS (attempt {r.successful_attempt})"
                else:
                    status = "FAIL"
                print(f"  [{status}] {r.problem.id}: {r.problem.title}")
                if not r.passed and r.attempts:
                    last = r.attempts[-1]
                    if last.error:
                        print(f"         Error: {last.error}")
                    if last.execution and last.execution.stderr:
                        lines = last.execution.stderr.strip().split("\n")
                        for line in lines[-5:]:
                            print(f"         {line}")
            else:
                status = "PASS" if r.passed else "FAIL"
                print(f"  [{status}] {r.problem.id}: {r.problem.title}")
                if not r.passed and r.error:
                    print(f"         Error: {r.error}")
                if not r.passed and r.execution and r.execution.stderr:
                    lines = r.execution.stderr.strip().split("\n")
                    for line in lines[-5:]:
                        print(f"         {line}")
        print()


def generate_markdown_report(
    score: EvalScore | MultiturnEvalScore,
    results: list,
    output_dir: str,
    mode: str = "standard",
) -> str:
    """Generate a Markdown report file and return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_model = score.model.replace("/", "_").replace(" ", "_")
    md_path = out / f"report_{safe_model}_{timestamp}.md"

    mode_label = {"standard": "Standard", "bugfix": "Bug Fix", "multiturn": "Multi-turn"}
    total_passed = sum(c.total_passed for c in score.categories.values())
    total_count = sum(c.total_count for c in score.categories.values())
    pass_rate = total_passed / total_count * 100 if total_count > 0 else 0.0

    lines = [
        f"# PyEval Report: {score.model} [{mode_label.get(mode, mode)}]",
        f"",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Mode:** {mode_label.get(mode, mode)}  ",
        f"**Overall Pass Rate:** {pass_rate:.1f}%  ",
        f"**Weighted Score:** {score.weighted_score * 100:.1f}%  ",
        f"**Total:** {total_passed}/{total_count}",
        f"",
    ]

    if mode == "multiturn" and isinstance(score, MultiturnEvalScore):
        lines.extend([
            f"**Strict Pass Rate (1st try):** {score.strict_pass_rate * 100:.1f}%  ",
            f"**With Retries Pass Rate:** {score.pass_rate * 100:.1f}%  ",
            f"",
        ])

    lines.extend([
        f"## Summary",
        f"",
        f"| Category | Easy | Medium | Hard | Total |",
        f"|----------|------|--------|------|-------|",
    ])

    ordered_cats = _ordered_cats(score)

    for cat in ordered_cats:
        cs = score.categories[cat]
        cols = []
        for diff in DIFFICULTIES:
            ds = cs.by_difficulty.get(diff)
            cols.append(_fmt_score(ds.passed, ds.total) if ds else "-")
        total = _fmt_score(cs.total_passed, cs.total_count)
        lines.append(f"| {cs.display_name} | {cols[0]} | {cols[1]} | {cols[2]} | {total} |")

    # Total row
    tbd = _total_by_diff(score)
    cols = [_fmt_score(tbd[d][0], tbd[d][1]) for d in DIFFICULTIES]
    total_str = _fmt_score(total_passed, total_count)
    lines.append(f"| **TOTAL** | **{cols[0]}** | **{cols[1]}** | **{cols[2]}** | **{total_str}** |")

    if mode == "multiturn" and isinstance(score, MultiturnEvalScore):
        lines.extend([
            f"",
            f"## Attempt Distribution",
            f"",
        ])
        max_attempt = max(score.attempt_distribution.keys()) if score.attempt_distribution else 0
        for attempt_num in range(1, max(max_attempt + 1, 4)):
            w = ATTEMPT_WEIGHTS.get(attempt_num, 0.0)
            count = score.attempt_distribution.get(attempt_num, 0)
            lines.append(f"- Attempt {attempt_num} (weight={w}): {count} passed")

    lines.extend([
        f"",
        f"## Detailed Results",
        f"",
    ])

    for r in sorted(results, key=lambda x: x.problem.id):
        if mode == "multiturn" and isinstance(r, MultiturnResult):
            if r.passed:
                status = f"PASS (attempt {r.successful_attempt})"
            else:
                status = "FAIL"
            lines.append(f"### [{status}] {r.problem.id}: {r.problem.title}")
            lines.append(f"")
            lines.append(f"- **Category:** {r.problem.category}")
            lines.append(f"- **Difficulty:** {r.problem.difficulty}")
            lines.append(f"- **Attempts:** {len(r.attempts)}")
            if r.error:
                lines.append(f"- **Error:** {r.error}")
            if r.llm_response.code:
                lines.append(f"")
                lines.append(f"<details><summary>Final Generated Code</summary>")
                lines.append(f"")
                lines.append(f"```python")
                lines.append(r.llm_response.code)
                lines.append(f"```")
                lines.append(f"")
                lines.append(f"</details>")
        else:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"### [{status}] {r.problem.id}: {r.problem.title}")
            lines.append(f"")
            lines.append(f"- **Category:** {r.problem.category}")
            lines.append(f"- **Difficulty:** {r.problem.difficulty}")
            if r.error:
                lines.append(f"- **Error:** {r.error}")
            if r.llm_response.code:
                lines.append(f"")
                lines.append(f"<details><summary>Generated Code</summary>")
                lines.append(f"")
                lines.append(f"```python")
                lines.append(r.llm_response.code)
                lines.append(f"```")
                lines.append(f"")
                lines.append(f"</details>")
        lines.append(f"")

    md_path.write_text("\n".join(lines))

    # Also save raw results as JSON
    json_path = out / f"results_{safe_model}_{timestamp}.json"
    raw_results = []
    for r in sorted(results, key=lambda x: x.problem.id):
        entry = {
            "id": r.problem.id,
            "category": r.problem.category,
            "difficulty": r.problem.difficulty,
            "title": r.problem.title,
            "passed": r.passed,
            "error": r.error,
            "generated_code": r.llm_response.code,
            "stdout": r.execution.stdout if r.execution else "",
            "stderr": r.execution.stderr if r.execution else "",
        }
        if mode == "multiturn" and isinstance(r, MultiturnResult):
            entry["attempts"] = len(r.attempts)
            entry["successful_attempt"] = r.successful_attempt
            entry["score_weight"] = r.score_weight
        raw_results.append(entry)
    json_path.write_text(json.dumps(raw_results, indent=2, ensure_ascii=False))

    return str(md_path)
