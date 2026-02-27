"""Problem loading from JSON files."""

from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class Problem:
    id: str
    category: str
    difficulty: str
    title: str
    prompt: str
    reference_solution: str
    test_code: str
    tags: list[str] = field(default_factory=list)
    time_limit_seconds: int = 5


@dataclass
class BugfixProblem:
    id: str
    category: str
    difficulty: str
    title: str
    buggy_code: str
    bug_description: str
    reference_solution: str
    test_code: str
    tags: list[str] = field(default_factory=list)
    time_limit_seconds: int = 5


def load_problems(
    problems_dir: str,
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
    problem_ids: list[str] | None = None,
) -> list[Problem]:
    """Load problems from JSON files in the bank directory, with optional filtering."""
    bank_path = Path(problems_dir)
    if not bank_path.exists():
        raise FileNotFoundError(f"Problems directory not found: {problems_dir}")

    problems: list[Problem] = []
    for json_file in sorted(bank_path.glob("*.json")):
        # Skip bugfix problem files
        if json_file.name.startswith("bugfix_"):
            continue
        with open(json_file) as f:
            data = json.load(f)

        problem_list = data if isinstance(data, list) else [data]
        for item in problem_list:
            prob = Problem(**{k: v for k, v in item.items() if k in Problem.__dataclass_fields__})
            problems.append(prob)

    # Apply filters
    if problem_ids:
        problems = [p for p in problems if p.id in problem_ids]
    if categories:
        cats = {c.lower() for c in categories}
        problems = [p for p in problems if p.category.lower() in cats]
    if difficulties:
        diffs = {d.lower() for d in difficulties}
        problems = [p for p in problems if p.difficulty.lower() in diffs]

    return problems


def load_bugfix_problems(
    problems_dir: str,
    difficulties: list[str] | None = None,
    problem_ids: list[str] | None = None,
) -> list[BugfixProblem]:
    """Load bugfix problems from bugfix_*.json files in the bank directory."""
    bank_path = Path(problems_dir)
    if not bank_path.exists():
        raise FileNotFoundError(f"Problems directory not found: {problems_dir}")

    problems: list[BugfixProblem] = []
    for json_file in sorted(bank_path.glob("bugfix_*.json")):
        with open(json_file) as f:
            data = json.load(f)

        problem_list = data if isinstance(data, list) else [data]
        for item in problem_list:
            prob = BugfixProblem(**{k: v for k, v in item.items() if k in BugfixProblem.__dataclass_fields__})
            problems.append(prob)

    # Apply filters
    if problem_ids:
        problems = [p for p in problems if p.id in problem_ids]
    if difficulties:
        diffs = {d.lower() for d in difficulties}
        problems = [p for p in problems if p.difficulty.lower() in diffs]

    return problems
