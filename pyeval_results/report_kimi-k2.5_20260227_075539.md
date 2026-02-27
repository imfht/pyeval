# PyEval Report: kimi-k2.5 [Bug Fix]

**Date:** 2026-02-27 07:55:39 UTC  
**Mode:** Bug Fix  
**Overall Pass Rate:** 26.7%  
**Weighted Score:** 13.3%  
**Total:** 4/15

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| Bug Fix | 4/5 | 0/5 | 0/5 | 4/15 |
| **TOTAL** | **4/5** | **0/5** | **0/5** | **4/15** |

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
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_007: Mutable default argument

- **Category:** bugfix
- **Difficulty:** medium
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_008: Wrong exception type caught

- **Category:** bugfix
- **Difficulty:** medium
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_009: Closure variable capture

- **Category:** bugfix
- **Difficulty:** medium
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bf_010: Integer division vs float division

- **Category:** bugfix
- **Difficulty:** medium
- **Error:** API error: HTTP Error 429: Too Many Requests

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
