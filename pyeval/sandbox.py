"""Sandboxed code execution via subprocess."""

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecutionResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def execute_in_sandbox(
    generated_code: str,
    test_code: str,
    timeout: int = 10,
) -> ExecutionResult:
    """Execute generated code + test code in an isolated subprocess.

    Combines the generated code with the test code and runs it.
    Returns pass/fail based on the process return code.
    """
    # Combine generated code with test code
    full_code = f"{generated_code}\n\n{test_code}\n\nif __name__ == '__main__':\n    unittest.main()\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(
            passed=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False,
            stdout="",
            stderr=f"Execution timed out after {timeout}s",
            returncode=-1,
            timed_out=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
