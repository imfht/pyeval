# PyEval Report: MiniMax-M2.5 [Bug Fix]

**Date:** 2026-02-27 07:56:11 UTC  
**Mode:** Bug Fix  
**Overall Pass Rate:** 53.3%  
**Weighted Score:** 40.0%  
**Total:** 8/15

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| Bug Fix | 4/5 | 4/5 | 0/5 | 8/15 |
| **TOTAL** | **4/5** | **4/5** | **0/5** | **8/15** |

## Detailed Results

### [PASS] bf_001: Off-by-one in range

- **Category:** bugfix
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def sum_1_to_n(n: int) -> int:
    """Return the sum of integers from 1 to n (inclusive)."""
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
```

</details>

### [PASS] bf_002: Comparison operator reversed

- **Category:** bugfix
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def find_max(numbers: list[int]) -> int:
    """Return the maximum value in a non-empty list of integers."""
    result = numbers[0]
    for num in numbers[1:]:
        if num > result:
            result = num
    return result
```

</details>

### [PASS] bf_003: Missing return statement

- **Category:** bugfix
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome (case-insensitive, ignoring spaces)."""
    cleaned = s.lower().replace(' ', '')
    return cleaned == cleaned[::-1]
```

</details>

### [PASS] bf_004: Variable name typo

- **Category:** bugfix
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def count_vowels(text: str) -> int:
    """Count the number of vowels (a, e, i, o, u) in the text (case-insensitive)."""
    vowels = 'aeiou'
    count = 0
    for char in text:
        if char.lower() in vowels:
            count += 1
    return count
```

</details>

### [FAIL] bf_005: Wrong condition logic

- **Category:** bugfix
- **Difficulty:** easy
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_006: Shallow copy vs deep copy

- **Category:** bugfix
- **Difficulty:** medium
- **Error:** API request timed out

### [PASS] bf_007: Mutable default argument

- **Category:** bugfix
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
def append_to_list(value: int, lst: list[int] | None = None) -> list[int]:
    """Append a value to the list and return it.
    If no list is provided, create a new empty list."""
    if lst is None:
        lst = []
    lst.append(value)
    return lst
```

</details>

### [PASS] bf_008: Wrong exception type caught

- **Category:** bugfix
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
def safe_divide(a: float, b: float) -> dict:
    """Safely divide a by b. Returns {'result': value} on success,
    or {'error': message} on failure.
    Handle both division by zero and non-numeric inputs."""
    try:
        result = a / b
        return {'result': result}
    except ZeroDivisionError:
        return {'error': 'Division by zero'}
    except TypeError:
        return {'error': f'Invalid types: {type(a).__name__}, {type(b).__name__}'}
```

</details>

### [PASS] bf_009: Closure variable capture

- **Category:** bugfix
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
def make_multipliers(n: int) -> list:
    """Create a list of n multiplier functions.
    make_multipliers(5) should return [f0, f1, f2, f3, f4]
    where fi(x) returns x * i."""
    multipliers = []
    for i in range(n):
        multipliers.append(lambda x, i=i: x * i)
    return multipliers
```

</details>

### [PASS] bf_010: Integer division vs float division

- **Category:** bugfix
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
def compute_average(numbers: list[int]) -> float:
    """Compute the arithmetic average of a list of integers.
    Return 0.0 for an empty list."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
```

</details>

### [FAIL] bf_011: Thread race condition - missing lock

- **Category:** bugfix
- **Difficulty:** hard
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_012: Generator exhaustion on second iteration

- **Category:** bugfix
- **Difficulty:** hard
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_013: Decorator missing functools.wraps

- **Category:** bugfix
- **Difficulty:** hard
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_014: Diamond inheritance MRO issue

- **Category:** bugfix
- **Difficulty:** hard
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_015: Async context manager misuse

- **Category:** bugfix
- **Difficulty:** hard
- **Error:** API error: HTTP Error 429: Too Many Requests
