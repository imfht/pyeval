# PyEval Report: qwen3.5-flash

**Date:** 2026-02-27 04:42:53 UTC  
**Overall Pass Rate:** 100.0%  
**Weighted Score:** 100.0% (easy=1x, medium=2x, hard=3x)  
**Total:** 1/1

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| Basic Syntax | 1/1 | - | - | 1/1 |
| **TOTAL** | **1/1** | **-** | **-** | **1/1** |

## Detailed Results

### [PASS] bs_001: FizzBuzz

- **Category:** basic_syntax
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def fizzbuzz(n: int) -> list:
    """
    Given a positive integer n, return a list of strings from 1 to n where:
    - Multiples of 3 are replaced with "Fizz"
    - Multiples of 5 are replaced with "Buzz"
    - Multiples of both 3 and 5 are replaced with "FizzBuzz"
    - All other numbers are converted to their string representation

    Example:
        fizzbuzz(5) -> ["1", "2", "Fizz", "4", "Buzz"]
        fizzbuzz(15) -> ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]
    """
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
```

</details>
