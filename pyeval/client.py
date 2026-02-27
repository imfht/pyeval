"""OpenAI-compatible API client using only stdlib."""

import ast
import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass


@dataclass
class LLMResponse:
    code: str
    raw_response: str
    error: str | None = None


def _send_chat_request(
    url: str,
    api_key: str,
    payload: dict,
    timeout: int = 30,
) -> dict | str:
    """Send a chat completion request. Returns response body dict or error string."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        no_proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(no_proxy_handler)
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return f"API error: {e}"
    except TimeoutError:
        return "API request timed out"


def call_llm(
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 30,
    extra_body: dict | None = None,
) -> LLMResponse:
    """Call an OpenAI-compatible chat completion API and extract code from response."""
    url = f"{api_base.rstrip('/')}/chat/completions"

    system_message = (
        "You are a Python coding assistant. Complete the given function implementation. "
        "Return ONLY the complete function definition (including the signature and docstring provided). "
        "Do not include any test code, examples, or explanations outside the code block."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Complete this Python function:\n\n```python\n{prompt}\n```"},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)

    result = _send_chat_request(url, api_key, payload, timeout)
    if isinstance(result, str):
        return LLMResponse(code="", raw_response="", error=result)

    raw = result["choices"][0]["message"]["content"]
    code = extract_code(raw)
    return LLMResponse(code=code, raw_response=raw)


def call_llm_bugfix(
    buggy_code: str,
    bug_description: str,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 30,
    extra_body: dict | None = None,
) -> LLMResponse:
    """Call LLM to fix a bug in the given code."""
    url = f"{api_base.rstrip('/')}/chat/completions"

    system_message = (
        "You are a Python debugging assistant. You will be given buggy code and a description of the bug. "
        "Fix the bug and return ONLY the complete corrected code. "
        "Do not include any test code, examples, or explanations outside the code block."
    )

    user_message = (
        f"The following Python code has a bug:\n\n```python\n{buggy_code}\n```\n\n"
        f"Bug description: {bug_description}\n\n"
        f"Fix the bug and return the complete corrected code."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)

    result = _send_chat_request(url, api_key, payload, timeout)
    if isinstance(result, str):
        return LLMResponse(code="", raw_response="", error=result)

    raw = result["choices"][0]["message"]["content"]
    code = extract_code(raw)
    return LLMResponse(code=code, raw_response=raw)


def call_llm_multiturn(
    prompt: str,
    previous_code: str,
    error_output: str,
    attempt: int,
    api_base: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 30,
    extra_body: dict | None = None,
) -> LLMResponse:
    """Call LLM with conversation history for multi-turn error correction."""
    url = f"{api_base.rstrip('/')}/chat/completions"

    system_message = (
        "You are a Python coding assistant. Complete the given function implementation. "
        "Return ONLY the complete function definition (including the signature and docstring provided). "
        "Do not include any test code, examples, or explanations outside the code block."
    )

    feedback_message = (
        f"Your previous code (attempt {attempt - 1}) failed with the following error:\n\n"
        f"```python\n{previous_code}\n```\n\n"
        f"Error output:\n```\n{error_output}\n```\n\n"
        f"Please fix the code and try again. Return ONLY the complete corrected function."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Complete this Python function:\n\n```python\n{prompt}\n```"},
        {"role": "assistant", "content": f"```python\n{previous_code}\n```"},
        {"role": "user", "content": feedback_message},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)

    result = _send_chat_request(url, api_key, payload, timeout)
    if isinstance(result, str):
        return LLMResponse(code="", raw_response="", error=result)

    raw = result["choices"][0]["message"]["content"]
    code = extract_code(raw)
    return LLMResponse(code=code, raw_response=raw)


def extract_code(text: str) -> str:
    """Extract Python code from LLM response.

    Handles:
    - <think>...</think> blocks (Qwen3 thinking mode — stripped)
    - ```python ... ``` blocks
    - ``` ... ``` blocks
    - Plain text that looks like code
    """
    # Strip thinking blocks (e.g. Qwen3 <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try markdown python code block first
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[0].strip()
        if _is_valid_python(code):
            return code

    # Try generic code block
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = matches[0].strip()
        if _is_valid_python(code):
            return code

    # Try the whole text as code
    stripped = text.strip()
    if _is_valid_python(stripped):
        return stripped

    # Last resort: find lines that look like Python code (starting with def, class, import, etc.)
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(("def ", "class ", "import ", "from ")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        code = "\n".join(code_lines)
        if _is_valid_python(code):
            return code

    return stripped


def _is_valid_python(code: str) -> bool:
    """Check if a string is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
