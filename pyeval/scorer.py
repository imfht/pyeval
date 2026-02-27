"""Scoring and aggregation logic."""

from dataclasses import dataclass, field
from .runner import MultiturnResult, ProblemResult


DIFFICULTY_WEIGHTS = {"easy": 1, "medium": 2, "hard": 3}

ATTEMPT_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.3}

CATEGORY_DISPLAY = {
    "basic_syntax": "Basic Syntax",
    "data_structures": "Data Structures",
    "algorithms": "Algorithms",
    "stdlib": "Standard Library",
    "oop": "OOP",
    "exceptions": "Exceptions",
    "file_io": "File I/O",
    "string_processing": "String Processing",
    "functional": "Functional",
    "concurrency": "Concurrency",
    "bugfix": "Bug Fix",
}


@dataclass
class DifficultyScore:
    passed: int = 0
    total: int = 0

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class CategoryScore:
    category: str
    display_name: str
    by_difficulty: dict[str, DifficultyScore] = field(default_factory=dict)

    @property
    def total_passed(self) -> int:
        return sum(d.passed for d in self.by_difficulty.values())

    @property
    def total_count(self) -> int:
        return sum(d.total for d in self.by_difficulty.values())

    @property
    def pass_rate(self) -> float:
        return self.total_passed / self.total_count if self.total_count > 0 else 0.0


@dataclass
class EvalScore:
    model: str
    categories: dict[str, CategoryScore] = field(default_factory=dict)

    @property
    def total_passed(self) -> int:
        return sum(c.total_passed for c in self.categories.values())

    @property
    def total_count(self) -> int:
        return sum(c.total_count for c in self.categories.values())

    @property
    def pass_rate(self) -> float:
        return self.total_passed / self.total_count if self.total_count > 0 else 0.0

    @property
    def weighted_score(self) -> float:
        total_weighted = 0
        total_weight = 0
        for cat in self.categories.values():
            for diff, score in cat.by_difficulty.items():
                w = DIFFICULTY_WEIGHTS.get(diff, 1)
                total_weighted += score.passed * w
                total_weight += score.total * w
        return total_weighted / total_weight if total_weight > 0 else 0.0


@dataclass
class MultiturnEvalScore:
    model: str
    categories: dict[str, CategoryScore] = field(default_factory=dict)
    strict_passed: int = 0      # Passed on first attempt
    retry_passed: int = 0       # Passed on retry (attempt 2+)
    total_count: int = 0
    attempt_distribution: dict[int, int] = field(default_factory=dict)  # attempt_num -> count

    @property
    def total_passed(self) -> int:
        return self.strict_passed + self.retry_passed

    @property
    def pass_rate(self) -> float:
        return self.total_passed / self.total_count if self.total_count > 0 else 0.0

    @property
    def strict_pass_rate(self) -> float:
        return self.strict_passed / self.total_count if self.total_count > 0 else 0.0

    @property
    def retry_pass_rate(self) -> float:
        return self.retry_passed / self.total_count if self.total_count > 0 else 0.0

    @property
    def weighted_score(self) -> float:
        """Weighted score: sum(attempt_weight * difficulty_weight) / sum(difficulty_weight)."""
        total_weighted = 0.0
        total_weight = 0
        for cat in self.categories.values():
            for diff, score in cat.by_difficulty.items():
                w = DIFFICULTY_WEIGHTS.get(diff, 1)
                total_weight += score.total * w
        # We need per-result info for attempt weights, so compute from attempt_distribution + difficulty
        # This is tracked via _weighted_sum and _total_weight set during compute
        return self._weighted_sum / self._total_weight if self._total_weight > 0 else 0.0

    # Internal tracking for weighted score computation
    _weighted_sum: float = 0.0
    _total_weight: float = 0.0


def compute_scores(results: list, model: str, mode: str = "standard") -> EvalScore | MultiturnEvalScore:
    """Aggregate individual problem results into category/difficulty scores."""
    if mode == "multiturn":
        return _compute_multiturn_scores(results, model)

    eval_score = EvalScore(model=model)

    for r in results:
        cat = r.problem.category
        diff = r.problem.difficulty

        if cat not in eval_score.categories:
            eval_score.categories[cat] = CategoryScore(
                category=cat,
                display_name=CATEGORY_DISPLAY.get(cat, cat),
            )

        cat_score = eval_score.categories[cat]
        if diff not in cat_score.by_difficulty:
            cat_score.by_difficulty[diff] = DifficultyScore()

        cat_score.by_difficulty[diff].total += 1
        if r.passed:
            cat_score.by_difficulty[diff].passed += 1

    return eval_score


def _compute_multiturn_scores(results: list[MultiturnResult], model: str) -> MultiturnEvalScore:
    """Compute scores for multiturn mode with attempt weighting."""
    score = MultiturnEvalScore(model=model)
    score.total_count = len(results)

    weighted_sum = 0.0
    total_weight = 0.0

    for r in results:
        cat = r.problem.category
        diff = r.problem.difficulty
        diff_w = DIFFICULTY_WEIGHTS.get(diff, 1)

        if cat not in score.categories:
            score.categories[cat] = CategoryScore(
                category=cat,
                display_name=CATEGORY_DISPLAY.get(cat, cat),
            )

        cat_score = score.categories[cat]
        if diff not in cat_score.by_difficulty:
            cat_score.by_difficulty[diff] = DifficultyScore()

        cat_score.by_difficulty[diff].total += 1
        total_weight += diff_w

        if r.passed:
            cat_score.by_difficulty[diff].passed += 1
            attempt = r.successful_attempt or 1
            attempt_w = ATTEMPT_WEIGHTS.get(attempt, 0.0)
            weighted_sum += attempt_w * diff_w

            if attempt == 1:
                score.strict_passed += 1
            else:
                score.retry_passed += 1

            score.attempt_distribution[attempt] = score.attempt_distribution.get(attempt, 0) + 1

    score._weighted_sum = weighted_sum
    score._total_weight = total_weight

    return score
