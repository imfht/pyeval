"""Evaluation pipeline orchestration."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .client import LLMResponse, call_llm, call_llm_bugfix, call_llm_multiturn
from .config import EvalConfig
from .problems.loader import BugfixProblem, Problem
from .sandbox import ExecutionResult, execute_in_sandbox


@dataclass
class ProblemResult:
    problem: Problem | BugfixProblem
    llm_response: LLMResponse
    execution: ExecutionResult | None
    passed: bool
    error: str | None = None


@dataclass
class MultiturnResult:
    problem: Problem
    attempts: list[ProblemResult] = field(default_factory=list)
    passed: bool = False
    successful_attempt: int | None = None  # 1-indexed
    score_weight: float = 0.0

    @property
    def llm_response(self) -> LLMResponse:
        """Return the LLM response from the last attempt for compatibility."""
        if self.attempts:
            return self.attempts[-1].llm_response
        return LLMResponse(code="", raw_response="")

    @property
    def execution(self) -> ExecutionResult | None:
        """Return the execution from the last attempt for compatibility."""
        if self.attempts:
            return self.attempts[-1].execution
        return None

    @property
    def error(self) -> str | None:
        """Return the error from the last attempt for compatibility."""
        if self.attempts:
            return self.attempts[-1].error
        return None


ATTEMPT_WEIGHTS = {1: 1.0, 2: 0.6, 3: 0.3}


def run_single(problem: Problem, config: EvalConfig) -> ProblemResult:
    """Run evaluation for a single problem: call LLM, then execute in sandbox."""
    # Step 1: Call LLM to generate code
    llm_resp = call_llm(
        prompt=problem.prompt,
        api_base=config.api_base,
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        extra_body=config.extra_body or None,
    )

    if llm_resp.error:
        return ProblemResult(
            problem=problem,
            llm_response=llm_resp,
            execution=None,
            passed=False,
            error=llm_resp.error,
        )

    if not llm_resp.code.strip():
        return ProblemResult(
            problem=problem,
            llm_response=llm_resp,
            execution=None,
            passed=False,
            error="Empty code response from LLM",
        )

    # Step 2: Execute in sandbox
    timeout = problem.time_limit_seconds or config.sandbox_timeout
    exec_result = execute_in_sandbox(
        generated_code=llm_resp.code,
        test_code=problem.test_code,
        timeout=timeout,
    )

    return ProblemResult(
        problem=problem,
        llm_response=llm_resp,
        execution=exec_result,
        passed=exec_result.passed,
    )


def run_single_bugfix(problem: BugfixProblem, config: EvalConfig) -> ProblemResult:
    """Run bugfix evaluation: call LLM to fix buggy code, then execute in sandbox."""
    llm_resp = call_llm_bugfix(
        buggy_code=problem.buggy_code,
        bug_description=problem.bug_description,
        api_base=config.api_base,
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        extra_body=config.extra_body or None,
    )

    if llm_resp.error:
        return ProblemResult(
            problem=problem,
            llm_response=llm_resp,
            execution=None,
            passed=False,
            error=llm_resp.error,
        )

    if not llm_resp.code.strip():
        return ProblemResult(
            problem=problem,
            llm_response=llm_resp,
            execution=None,
            passed=False,
            error="Empty code response from LLM",
        )

    timeout = problem.time_limit_seconds or config.sandbox_timeout
    exec_result = execute_in_sandbox(
        generated_code=llm_resp.code,
        test_code=problem.test_code,
        timeout=timeout,
    )

    return ProblemResult(
        problem=problem,
        llm_response=llm_resp,
        execution=exec_result,
        passed=exec_result.passed,
    )


def run_single_multiturn(problem: Problem, config: EvalConfig) -> MultiturnResult:
    """Run multi-turn evaluation: retry with error feedback on failure."""
    mt_result = MultiturnResult(problem=problem)

    for attempt_num in range(1, config.max_attempts + 1):
        if attempt_num == 1:
            # First attempt: standard call
            llm_resp = call_llm(
                prompt=problem.prompt,
                api_base=config.api_base,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                extra_body=config.extra_body or None,
            )
        else:
            # Subsequent attempts: include previous code and error
            prev = mt_result.attempts[-1]
            error_output = ""
            if prev.execution:
                error_output = prev.execution.stderr
            elif prev.error:
                error_output = prev.error

            llm_resp = call_llm_multiturn(
                prompt=problem.prompt,
                previous_code=prev.llm_response.code,
                error_output=error_output,
                attempt=attempt_num,
                api_base=config.api_base,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                extra_body=config.extra_body or None,
            )

        if llm_resp.error:
            result = ProblemResult(
                problem=problem,
                llm_response=llm_resp,
                execution=None,
                passed=False,
                error=llm_resp.error,
            )
            mt_result.attempts.append(result)
            continue

        if not llm_resp.code.strip():
            result = ProblemResult(
                problem=problem,
                llm_response=llm_resp,
                execution=None,
                passed=False,
                error="Empty code response from LLM",
            )
            mt_result.attempts.append(result)
            continue

        timeout = problem.time_limit_seconds or config.sandbox_timeout
        exec_result = execute_in_sandbox(
            generated_code=llm_resp.code,
            test_code=problem.test_code,
            timeout=timeout,
        )

        result = ProblemResult(
            problem=problem,
            llm_response=llm_resp,
            execution=exec_result,
            passed=exec_result.passed,
        )
        mt_result.attempts.append(result)

        if exec_result.passed:
            mt_result.passed = True
            mt_result.successful_attempt = attempt_num
            mt_result.score_weight = ATTEMPT_WEIGHTS.get(attempt_num, 0.0)
            break

    return mt_result


def run_dry(problem: Problem, config: EvalConfig) -> ProblemResult:
    """Dry-run: execute reference solution against tests (no LLM call)."""
    timeout = problem.time_limit_seconds or config.sandbox_timeout
    exec_result = execute_in_sandbox(
        generated_code=problem.reference_solution,
        test_code=problem.test_code,
        timeout=timeout,
    )

    llm_resp = LLMResponse(code=problem.reference_solution, raw_response="[dry-run]")

    return ProblemResult(
        problem=problem,
        llm_response=llm_resp,
        execution=exec_result,
        passed=exec_result.passed,
        error=None if exec_result.passed else f"Reference solution failed: {exec_result.stderr}",
    )


def run_dry_multiturn(problem: Problem, config: EvalConfig) -> MultiturnResult:
    """Dry-run for multiturn: execute reference solution, wrap as MultiturnResult."""
    timeout = problem.time_limit_seconds or config.sandbox_timeout
    exec_result = execute_in_sandbox(
        generated_code=problem.reference_solution,
        test_code=problem.test_code,
        timeout=timeout,
    )

    llm_resp = LLMResponse(code=problem.reference_solution, raw_response="[dry-run]")
    pr = ProblemResult(
        problem=problem,
        llm_response=llm_resp,
        execution=exec_result,
        passed=exec_result.passed,
        error=None if exec_result.passed else f"Reference solution failed: {exec_result.stderr}",
    )

    mt = MultiturnResult(
        problem=problem,
        attempts=[pr],
        passed=exec_result.passed,
        successful_attempt=1 if exec_result.passed else None,
        score_weight=1.0 if exec_result.passed else 0.0,
    )
    return mt


def run_dry_bugfix(problem: BugfixProblem, config: EvalConfig) -> ProblemResult:
    """Dry-run for bugfix: execute reference solution against tests."""
    timeout = problem.time_limit_seconds or config.sandbox_timeout
    exec_result = execute_in_sandbox(
        generated_code=problem.reference_solution,
        test_code=problem.test_code,
        timeout=timeout,
    )

    llm_resp = LLMResponse(code=problem.reference_solution, raw_response="[dry-run]")

    return ProblemResult(
        problem=problem,
        llm_response=llm_resp,
        execution=exec_result,
        passed=exec_result.passed,
        error=None if exec_result.passed else f"Reference solution failed: {exec_result.stderr}",
    )


def run_evaluation(
    problems: list,
    config: EvalConfig,
    progress_callback=None,
) -> list:
    """Run the full evaluation pipeline with concurrency."""
    results: list = []

    if config.mode == "bugfix":
        run_fn = run_dry_bugfix if config.dry_run else run_single_bugfix
    elif config.mode == "multiturn":
        if config.dry_run:
            run_fn = run_dry_multiturn
        else:
            run_fn = run_single_multiturn
    else:
        run_fn = run_dry if config.dry_run else run_single

    max_workers = config.max_concurrent_sandbox if config.dry_run else config.max_concurrent_api

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_problem = {
            executor.submit(run_fn, prob, config): prob
            for prob in problems
        }

        for future in as_completed(future_to_problem):
            problem = future_to_problem[future]
            try:
                result = future.result()
            except Exception as e:
                result = ProblemResult(
                    problem=problem,
                    llm_response=LLMResponse(code="", raw_response=""),
                    execution=None,
                    passed=False,
                    error=str(e),
                )
            results.append(result)
            if progress_callback:
                progress_callback(result, len(results), len(problems))

    # Sort results by problem id for consistent ordering
    results.sort(key=lambda r: r.problem.id)
    return results
