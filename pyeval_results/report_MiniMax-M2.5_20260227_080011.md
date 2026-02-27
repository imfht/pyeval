# PyEval Report: MiniMax-M2.5 [Multi-turn]

**Date:** 2026-02-27 08:00:11 UTC  
**Mode:** Multi-turn  
**Overall Pass Rate:** 66.7%  
**Weighted Score:** 59.3%  
**Total:** 30/45

**Strict Pass Rate (1st try):** 48.9%  
**With Retries Pass Rate:** 66.7%  

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| Basic Syntax | 0/3 | 1/1 | 0/1 | 1/5 |
| Data Structures | 0/2 | 0/2 | 0/1 | 0/5 |
| Algorithms | 0/2 | 2/3 | 2/2 | 4/7 |
| Standard Library | 2/2 | 1/2 | 0/1 | 3/5 |
| OOP | 1/1 | 2/2 | 2/2 | 5/5 |
| Exceptions | 1/1 | 1/1 | 0/1 | 2/3 |
| File I/O | 1/1 | 1/1 | 1/1 | 3/3 |
| String Processing | 2/2 | 1/1 | 1/1 | 4/4 |
| Functional | 1/1 | 2/2 | 1/1 | 4/4 |
| Concurrency | 1/1 | 2/2 | 1/1 | 4/4 |
| **TOTAL** | **9/16** | **13/17** | **8/12** | **30/45** |

## Attempt Distribution

- Attempt 1 (weight=1.0): 22 passed
- Attempt 2 (weight=0.6): 6 passed
- Attempt 3 (weight=0.3): 2 passed

## Detailed Results

### [FAIL] algo_001: Binary Search

- **Category:** algorithms
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] algo_002: Bubble Sort

- **Category:** algorithms
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [PASS (attempt 3)] algo_003: Merge Sort

- **Category:** algorithms
- **Difficulty:** medium
- **Attempts:** 3

<details><summary>Final Generated Code</summary>

```python
def merge_sort(arr: list[int]) -> list[int]:
    """Sort a list of integers in ascending order using merge sort.

    Args:
        arr: A list of integers to sort.

    Returns:
        A new sorted list. The original list must not be modified.
    """
    if len(arr) <= 1:
        return list(arr)
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
```

</details>

### [FAIL] algo_004: Longest Common Subsequence

- **Category:** algorithms
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 500: Internal Server Error

### [PASS (attempt 1)] algo_005: Valid Parentheses

- **Category:** algorithms
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def valid_parentheses(s: str) -> bool:
    """Check if a string of brackets is valid.

    A string is valid if:
    - Open brackets are closed by the same type of brackets.
    - Open brackets are closed in the correct order.
    - Every close bracket has a corresponding open bracket of the same type.

    Supported bracket types: (), [], {}

    Args:
        s: A string containing only the characters '(', ')', '[', ']', '{', '}'.

    Returns:
        True if the string is valid, False otherwise.
    """
    stack = []
    bracket_map = {')': '(', ']': '[', '}': '{'}
    opening_brackets = set('([{')
    
    for char in s:
        if char in opening_brackets:
            stack.append(char)
        elif char in bracket_map:
            if not stack or stack[-1] != bracket_map[char]:
                return False
            stack.pop()
    
    return len(stack) == 0
```

</details>

### [PASS (attempt 1)] algo_006: 0/1 Knapsack

- **Category:** algorithms
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """Solve the 0/1 knapsack problem.

    Given a set of items, each with a weight and a value, determine the
    maximum total value that can be achieved by selecting items such that
    the total weight does not exceed the given capacity. Each item can
    either be included or excluded (no fractions).

    Args:
        weights: A list of positive integer weights for each item.
        values: A list of positive integer values for each item.
        capacity: The maximum total weight the knapsack can hold (non-negative integer).

    Returns:
        The maximum total value achievable within the weight capacity.
    """
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0
    
    # Create a 2D DP table where dp[i][w] represents the maximum value
    # achievable using the first i items with a capacity of w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Option 1: Don't include item i-1
            dp[i][w] = dp[i - 1][w]
            
            # Option 2: Include item i-1 if it fits
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
    
    return dp[n][capacity]
```

</details>

### [PASS (attempt 1)] algo_007: Topological Sort

- **Category:** algorithms
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Return a topological ordering of a directed acyclic graph (DAG).

    Args:
        graph: An adjacency list representation of a directed graph.
               Keys are node names (strings), values are lists of nodes
               that the key node has edges TO. For example,
               {'a': ['b', 'c']} means edges a->b and a->c.
               All nodes in the graph appear as keys (even if they
               have no outgoing edges).

    Returns:
        A list of node names in a valid topological order. If the graph
        contains a cycle, return an empty list.
    """
    from collections import deque
    
    # Calculate in-degree for each node
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # Initialize queue with nodes that have no incoming edges
    queue = deque([node for node in graph if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If we processed all nodes, return the result; otherwise, there's a cycle
    if len(result) == len(graph):
        return result
    return []
```

</details>

### [FAIL] bs_001: FizzBuzz

- **Category:** basic_syntax
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bs_002: Swap Variables

- **Category:** basic_syntax
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 500: Internal Server Error

### [FAIL] bs_003: Number to Words

- **Category:** basic_syntax
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [PASS (attempt 1)] bs_004: Flatten Nested List

- **Category:** basic_syntax
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def flatten_nested(lst: list) -> list:
    """
    Flatten an arbitrarily nested list into a single flat list.
    The order of elements should be preserved (depth-first, left-to-right).
    Non-list elements should be kept as-is.

    Examples:
        flatten_nested([1, [2, 3], [4, [5, 6]]]) -> [1, 2, 3, 4, 5, 6]
        flatten_nested([1, [2, [3, [4, [5]]]]]) -> [1, 2, 3, 4, 5]
        flatten_nested([]) -> []
        flatten_nested([1, 'hello', [2, [True, None]]]) -> [1, 'hello', 2, True, None]
    """
    result = []
    for element in lst:
        if isinstance(element, list):
            result.extend(flatten_nested(element))
        else:
            result.append(element)
    return result
```

</details>

### [FAIL] bs_005: Matrix Spiral Order

- **Category:** basic_syntax
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [PASS (attempt 1)] con_001: Thread-Safe Counter

- **Category:** concurrency
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import threading

class ThreadSafeCounter:
    """
    A thread-safe counter that can be safely incremented and decremented
    from multiple threads simultaneously.

    Methods:
        __init__(self, initial=0): Initialize the counter with an optional
            starting value (default 0).
        increment(self): Atomically increase the counter by 1.
        decrement(self): Atomically decrease the counter by 1.
        get_value(self): Return the current counter value.

    The counter must use threading.Lock to ensure thread safety.

    Examples:
        counter = ThreadSafeCounter()
        counter.increment()
        counter.increment()
        counter.get_value()  # -> 2
        counter.decrement()
        counter.get_value()  # -> 1

        counter = ThreadSafeCounter(10)
        counter.get_value()  # -> 10
    """
    
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def decrement(self):
        with self._lock:
            self._value -= 1
    
    def get_value(self):
        with self._lock:
            return self._value
```

</details>

### [PASS (attempt 2)] con_002: Parallel Map

- **Category:** concurrency
- **Difficulty:** medium
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_map(func, items: list, max_workers: int = 4) -> list:
    """
    Apply a function to each item in a list using parallel execution with
    a thread pool. Return the results in the same order as the input items.

    Args:
        func: A callable to apply to each item.
        items: A list of items to process.
        max_workers: Maximum number of threads to use (default 4).

    Returns:
        A list of results in the same order as the input.

    Examples:
        parallel_map(lambda x: x ** 2, [1, 2, 3, 4])  # -> [1, 4, 9, 16]        parallel_map(str.upper, ["hello", "world"])  # -> ["HELLO", "WORLD"]
        parallel_map(lambda x: x + 1, [], max_workers=2)  # -> []
    """
    if not items:
        return []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    
    return results
```

</details>

### [PASS (attempt 1)] con_003: Producer Consumer

- **Category:** concurrency
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import threading
import queue

def producer_consumer(producers: list, consumers: list, queue_size: int = 10) -> list:
    """
    Implement the producer-consumer pattern using threading and queue.Queue.

    Args:
        producers: A list of callables. Each producer callable, when called,
            returns a list of items to be placed on the queue.        consumers: A list of callables. Each consumer callable takes a single
            item from the queue and returns a processed result.
        queue_size: Maximum size of the shared queue (default 10).

    Behavior:
        - Create a bounded Queue with the given queue_size.
        - Start one thread per consumer FIRST. Each consumer thread loops:
          gets an item from the queue, and if it's None (sentinel), stops.
          Otherwise, calls its consumer callable on the item and collects
          the result.
        - Start one thread per producer. Each producer thread calls its callable
          to get items, then puts each item on the queue.
        - Wait for all producer threads to finish.
        - Put one sentinel None per consumer on the queue.
        - Wait for all consumer threads to finish.
        - Return ALL consumed (processed) results as a single flat list.
          The order of results does not need to match input order.

    Examples:
        results = producer_consumer(
            producers=[lambda: [1, 2, 3], lambda: [4, 5]],
            consumers=[lambda x: x * 2],
            queue_size=5
        )
        sorted(results)  # -> [2, 4, 6, 8, 10]
    """
    shared_queue = queue.Queue(maxsize=queue_size)
    results = []
    results_lock = threading.Lock()
    
    def consumer_wrapper(consumer_func):
        while True:
            item = shared_queue.get()
            if item is None:
                shared_queue.task_done()
                break
            result = consumer_func(item)
            with results_lock:
                results.append(result)
            shared_queue.task_done()
    
    def producer_wrapper(producer_func):
        items = producer_func()
        for item in items:
            shared_queue.put(item)
    
    # Start consumer threads FIRST
    consumer_threads = []
    for consumer in consumers:
        t = threading.Thread(target=consumer_wrapper, args=(consumer,))
        t.start()
        consumer_threads.append(t)
    
    # Start producer threads
    producer_threads = []
    for producer in producers:
        t = threading.Thread(target=producer_wrapper, args=(producer,))
        t.start()
        producer_threads.append(t)
    
    # Wait for all producer threads to finish
    for t in producer_threads:
        t.join()
    
    # Put one sentinel None per consumer on the queue
    for _ in range(len(consumers)):
        shared_queue.put(None)
    
    # Wait for all consumer threads to finish
    for t in consumer_threads:
        t.join()
    
    return results
```

</details>

### [PASS (attempt 2)] con_004: Async Gather with Timeout

- **Category:** concurrency
- **Difficulty:** hard
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
import asyncio

async def async_gather_results(async_callables: list, timeout: float = None) -> list:
    """
    Run a list of async callables (coroutine functions) concurrently and
    return their results.

    Args:
        async_callables: A list of async callables (coroutine functions that
            take no arguments). Each one, when called, returns a coroutine.
        timeout: Optional timeout in seconds. If provided and the total
            execution exceeds this duration, raise asyncio.TimeoutError.
            If None, no timeout is applied.

    Returns:
        A list of results in the same order as the input callables.

    Behavior:
        - Call each async callable to create a coroutine.
        - Run all coroutines concurrently using asyncio.gather.
        - If timeout is specified, use asyncio.wait_for to enforce the timeout
          around the gather call.
        - Return the list of results.

    Examples:
        async def make_value(v):
            await asyncio.sleep(0.01)
            return v

        results = await async_gather_results([
            lambda: make_value(1),
            lambda: make_value(2),
            lambda: make_value(3)
        ])
        # results -> [1, 2, 3]

        # With timeout (this would raise asyncio.TimeoutError if too slow):
        results = await async_gather_results(
            [lambda: make_value(42)],
            timeout=5.0
        )
        # results -> [42]
    """
    # Create coroutines by calling each async callable
    coroutines = [callable() for callable in async_callables]
    
    # Run all coroutines concurrently
    if timeout is not None:
        # Use wait_for to apply timeout
        results = await asyncio.wait_for(
            asyncio.gather(*coroutines),
            timeout=timeout
        )
    else:
        results = await asyncio.gather(*coroutines)
    
    return results
```

</details>

### [FAIL] ds_001: Two Sum

- **Category:** data_structures
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] ds_002: Reverse Linked List

- **Category:** data_structures
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] ds_003: LRU Cache

- **Category:** data_structures
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] ds_004: Group Anagrams

- **Category:** data_structures
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] ds_005: Merge K Sorted Lists

- **Category:** data_structures
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [PASS (attempt 2)] exc_001: Safe Divide

- **Category:** exceptions
- **Difficulty:** easy
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
def safe_divide(a, b):
    """Divide a by b safely.

    Args:
        a: The numerator.
        b: The denominator.

    Returns:
        The result of a / b, or None if b is zero.

    Raises:
        TypeError: If either a or b is not a number (int or float).
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both a and b must be numbers (int or float)")
    
    if b == 0:
        return None
    
    return a / b
```

</details>

### [PASS (attempt 1)] exc_002: Retry Decorator

- **Category:** exceptions
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import time
import functools


def retry(max_attempts=3, exceptions=(Exception,), delay=0):
    """A decorator that retries a function on failure.

    Args:
        max_attempts (int): Maximum number of attempts (including the first call).
            Must be >= 1.
        exceptions (tuple): A tuple of exception types to catch and retry on.
            Any other exception types should propagate immediately.
        delay (float): Seconds to wait between retries. Defaults to 0.

    Returns:
        A decorator that wraps the target function with retry logic.

    Behavior:
        - Call the decorated function. If it succeeds, return its result.
        - If it raises one of the specified exception types and attempts remain,
          wait `delay` seconds, then retry.
        - If all attempts are exhausted, raise the last exception.
        - The decorator should preserve the original function's name and docstring
          (use functools.wraps).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        if delay > 0:
                            time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator
```

</details>

### [FAIL] exc_003: Custom Exception Hierarchy and Schema Validation

- **Category:** exceptions
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API request timed out

### [PASS (attempt 1)] fio_001: Read CSV Data

- **Category:** file_io
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import csv
import io

def read_csv_data(csv_string: str) -> list:
    """
    Parse a CSV-formatted string into a list of dictionaries.

    The first row of the CSV string contains the headers, which become
    the dictionary keys. Each subsequent row becomes a dictionary with
    values mapped to the corresponding headers.

    Use the csv module for parsing. Handle quoted fields correctly.

    Args:
        csv_string: A string containing CSV-formatted data

    Returns:
        A list of dictionaries, one per data row

    Examples:
        read_csv_data('name,age\nAlice,30\nBob,25')
        -> [{'name': 'Alice', 'age': '30'}, {'name': 'Bob', 'age': '25'}]

        read_csv_data('a,b\n1,2')
        -> [{'a': '1', 'b': '2'}]
    """
    # Create a StringIO object from the csv_string to treat it as a file
    csv_file = io.StringIO(csv_string)
    
    # Use DictReader to parse the CSV - it automatically uses the first row
    # as headers and handles quoted fields correctly
    reader = csv.DictReader(csv_file)
    
    # Convert the reader to a list of dictionaries and return
    return list(reader)
```

</details>

### [PASS (attempt 1)] fio_002: Merge JSON Files

- **Category:** file_io
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import json

def merge_json_files(json_strings: list) -> dict:
    """
    Merge a list of JSON strings into a single dictionary.

    Each JSON string represents a dictionary. Merge them left to right so
    that later values override earlier ones for the same keys. For nested
    dictionaries, perform a deep merge: if both the existing and new values
    for a key are dicts, merge them recursively instead of replacing.

    Non-dict values always override (even if the existing value is a dict).

    Args:
        json_strings: A list of strings, each a valid JSON object

    Returns:
        A single merged dictionary

    Examples:
        merge_json_files(['{"a": 1}', '{"b": 2}'])
        -> {'a': 1, 'b': 2}

        merge_json_files(['{"a": 1}', '{"a": 2}'])
        -> {'a': 2}

        merge_json_files(['{"x": {"a": 1, "b": 2}}', '{"x": {"b": 3, "c": 4}}'])
        -> {'x': {'a': 1, 'b': 3, 'c': 4}}
    """
    result = {}
    
    for json_str in json_strings:
        data = json.loads(json_str)
        result = _deep_merge(result, data)
    
    return result


def _deep_merge(existing: dict, new: dict) -> dict:
    """
    Recursively merge new dictionary into existing dictionary.
    Non-dict values in new override values in existing.
    """
    for key, value in new.items():
        if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
            # Both values are dicts, merge recursively
            existing[key] = _deep_merge(existing[key], value)
        else:
            # Non-dict value overrides (or key doesn't exist)
            existing[key] = value
    return existing
```

</details>

### [PASS (attempt 1)] fio_003: Log Parser

- **Category:** file_io
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import re

def log_parser(log_lines: list) -> dict:
    # Regular expression to parse the log lines
    pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (DEBUG|INFO|WARNING|ERROR|CRITICAL): (.*)'
    
    counts = {}
    errors = []
    timestamps = []
    
    for line in log_lines:
        match = re.match(pattern, line)
        if match:
            timestamp, level, message = match.groups()
            timestamps.append(timestamp)
            counts[level] = counts.get(level, 0) + 1
            
            if level in ('ERROR', 'CRITICAL'):
                errors.append(message)
    
    # Determine timestamp range
    if timestamps:
        timestamp_range = (min(timestamps), max(timestamps))
    else:
        timestamp_range = None
    
    return {
        'counts': counts,
        'errors': errors,
        'timestamp_range': timestamp_range
    }
```

</details>

### [PASS (attempt 1)] fn_001: Apply Pipeline

- **Category:** functional
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def apply_pipeline(value, functions: list):
    """
    Given an initial value and a list of single-argument functions,
    apply each function in sequence (left to right) to the value,
    passing the result of each function as input to the next.
    Return the final result.

    If the functions list is empty, return the original value.

    Examples:
        apply_pipeline(5, [lambda x: x + 1, lambda x: x * 2]) -> 12
        apply_pipeline("hello", [str.upper, lambda s: s + "!"]) -> "HELLO!"
        apply_pipeline(10, []) -> 10
        apply_pipeline(-3, [abs, lambda x: x ** 2]) -> 9
    """
    result = value
    for func in functions:
        result = func(result)
    return result
```

</details>

### [PASS (attempt 1)] fn_002: Memoize Decorator

- **Category:** functional
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def memoize(func):
    """
    Implement a memoize decorator that caches the results of function calls.
    If the function is called again with the same positional arguments,
    return the cached result instead of recomputing.

    Assume all arguments are hashable (can be used as dict keys via tuple).
    Only positional arguments need to be supported (no kwargs).

    The decorated function should also have a .cache attribute that is
    the dictionary used for caching (mapping argument tuples to results).

    Examples:
        @memoize
        def add(a, b):
            return a + b

        add(1, 2)  # computes and caches
        add(1, 2)  # returns cached result
        add.cache  # {(1, 2): 3}
    """
    cache = {}
    
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    
    wrapper.cache = cache
    return wrapper
```

</details>

### [PASS (attempt 1)] fn_003: Group By

- **Category:** functional
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def group_by(items: list, key_func) -> dict:
    """
    Group elements of a list by the result of applying key_func to each element.
    Return a dictionary where keys are the results of key_func and values are
    lists of items that produced that key, in their original order.

    Examples:
        group_by([1, 2, 3, 4, 5, 6], lambda x: x % 2 == 0)
        # -> {False: [1, 3, 5], True: [2, 4, 6]}

        group_by(["apple", "banana", "avocado", "blueberry"], lambda s: s[0])
        # -> {"a": ["apple", "avocado"], "b": ["banana", "blueberry"]}

        group_by([], lambda x: x)
        # -> {}

        group_by([1, 2, 3, 4, 5], lambda x: "even" if x % 2 == 0 else "odd")
        # -> {"odd": [1, 3, 5], "even": [2, 4]}
    """
    result = {}
    for item in items:
        key = key_func(item)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result
```

</details>

### [PASS (attempt 2)] fn_004: Lazy Evaluation

- **Category:** functional
- **Difficulty:** hard
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
class Lazy:
    """
    A lazy evaluation class that wraps an iterable and supports chaining
    map and filter operations. Operations are NOT executed until
    .evaluate() is called.

    Methods:
        __init__(self, iterable): Store the iterable and initialize an empty
            list of pending operations.
        map(self, func): Register a map operation. Returns self for chaining.
        filter(self, predicate): Register a filter operation. Returns self for chaining.
        evaluate(self): Execute all registered operations in order on the
            iterable and return the result as a list.

    The Lazy object should be reusable - after calling evaluate(), the
    pending operations are cleared but the original iterable is preserved,
    so new chains can be built.

    Examples:
        Lazy([1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).map(lambda x: x * 10).evaluate() # -> [20, 40]

        lazy = Lazy(range(10))
        lazy.map(lambda x: x ** 2).filter(lambda x: x > 10).evaluate()
        # -> [16, 25, 36, 49, 64, 81]

        Lazy([]).evaluate()  # -> []
    """
    
    def __init__(self, iterable):
        self.iterable = list(iterable)
        self.operations = []
    
    def map(self, func):
        self.operations.append(('map', func))
        return self
    
    def filter(self, predicate):
        self.operations.append(('filter', predicate))
        return self
    
    def evaluate(self):
        result = self.iterable.copy()
        for op_type, op_func in self.operations:
            if op_type == 'map':
                result = [op_func(x) for x in result]
            elif op_type == 'filter':
                result = [x for x in result if op_func(x)]
        self.operations = []
        return result
```

</details>

### [PASS (attempt 1)] oop_001: Stack Class

- **Category:** oop
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
class Stack:
    """A stack data structure with standard operations.

    Methods:
        push(item): Add an item to the top of the stack.
        pop(): Remove and return the top item. Raise IndexError if empty.
        peek(): Return the top item without removing it. Raise IndexError if empty.
        is_empty(): Return True if the stack is empty, False otherwise.
        size(): Return the number of items in the stack.
    """

    def __init__(self):
        """Initialize an empty stack."""
        self._items = []

    def push(self, item):
        """Add an item to the top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._items[-1]

    def is_empty(self):
        """Return True if the stack is empty, False otherwise."""
        return len(self._items) == 0

    def size(self):
        """Return the number of items in the stack."""
        return len(self._items)
```

</details>

### [PASS (attempt 3)] oop_002: Shape Hierarchy with Abstract Base Class

- **Category:** oop
- **Difficulty:** medium
- **Attempts:** 3

<details><summary>Final Generated Code</summary>

```python
from abc import ABC, abstractmethod
import math


class Shape(ABC):
    """Abstract base class for shapes.

    Abstract Methods:
        area() -> float: Return the area of the shape.
        perimeter() -> float: Return the perimeter of the shape.
    """

    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass


class Circle(Shape):
    """A circle defined by its radius.

    Args:
        radius (float): The radius of the circle. Must be positive.
            Raise ValueError if radius <= 0.
    """

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    """A rectangle defined by width and height.

    Args:
        width (float): The width of the rectangle. Must be positive.
        height (float): The height of the rectangle. Must be positive.
            Raise ValueError if width or height <= 0.
    """

    def __init__(self, width: float, height: float):
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


def calculate_total_area(shapes):
    """Return the sum of areas of all shapes in the list.

    Args:
        shapes: A list of Shape instances.

    Returns:
        float: The total area.
    """
    return sum(shape.area() for shape in shapes)
```

</details>

### [PASS (attempt 1)] oop_003: BankAccount with Transaction History

- **Category:** oop
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
class BankAccount:
    """A bank account with deposit, withdraw, transfer, and transaction history.

    Constructor:
        __init__(self, owner: str, initial_balance: float = 0.0)
            Store owner as a public attribute (self.owner).
            Store balance internally; expose via read-only property.
            Initialize an empty transaction history list.
            Raise ValueError if initial_balance < 0.
            Do NOT record the initial balance as a transaction.

    Public Attributes:
        owner (str): The name of the account owner (directly accessible).

    Properties:
        balance (float): The current balance (read-only property).

    Methods:
        deposit(amount) -> None:
            Add amount to balance. Raise ValueError if amount <= 0.
            Record transaction as ('deposit', amount) in history.

        withdraw(amount) -> None:
            Subtract amount from balance.
            Raise ValueError if amount <= 0.
            Raise ValueError if insufficient funds.
            Record transaction as ('withdraw', amount) in history.

        transfer_to(other_account, amount) -> None:
            Withdraw amount from this account and deposit into other_account.
            Raise ValueError if amount <= 0.
            Raise ValueError if insufficient funds.
            Record transaction as ('transfer_out', amount) in this account's history.
            Record transaction as ('transfer_in', amount) in other account's history.

        get_history() -> list:
            Return a copy of all transaction tuples in chronological order.
    """

    def __init__(self, owner: str, initial_balance: float = 0.0):
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative")
        self.owner = owner
        self._balance = initial_balance
        self._history = []

    @property
    def balance(self) -> float:
        return self._balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        self._history.append(('deposit', amount))

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        self._history.append(('withdraw', amount))

    def transfer_to(self, other_account: 'BankAccount', amount: float) -> None:
        if amount <= 0:
            raise ValueError("Transfer amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        other_account._balance += amount
        self._history.append(('transfer_out', amount))
        other_account._history.append(('transfer_in', amount))

    def get_history(self) -> list:
        return self._history.copy()
```

</details>

### [PASS (attempt 1)] oop_004: EventEmitter - Observer Pattern

- **Category:** oop
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
class EventEmitter:
    """An event emitter implementing the observer pattern.

    Methods:
        on(event: str, callback: callable) -> None:
            Register a callback for the given event.
            The same callback can be registered multiple times for the same event;
            each registration counts separately.

        off(event: str, callback: callable) -> None:
            Remove one registration of callback for the given event.
            If the callback is registered multiple times, only remove the first one.
            If callback is not registered for this event, do nothing (no error).

        emit(event: str, *args, **kwargs) -> None:
            Call all registered callbacks for the event with the given arguments.
            Callbacks are called in the order they were registered.
            If no callbacks are registered for the event, do nothing.

        once(event: str, callback: callable) -> None:
            Register a callback that is automatically removed after it fires once.

        listener_count(event: str) -> int:
            Return the number of listeners registered for the given event.
    """

    def __init__(self):
        self._events = {}  # event_name -> list of callbacks

    def on(self, event: str, callback: callable) -> None:
        """Register a callback for the given event."""
        if event not in self._events:
            self._events[event] = []
        self._events[event].append(callback)

    def off(self, event: str, callback: callable) -> None:
        """Remove one registration of callback for the given event."""
        if event in self._events:
            try:
                self._events[event].remove(callback)
            except ValueError:
                pass  # callback not registered, do nothing

    def emit(self, event: str, *args, **kwargs) -> None:
        """Call all registered callbacks for the event with the given arguments."""
        if event in self._events:
            # Iterate over a copy to handle callbacks that modify the list during iteration
            for callback in self._events[event][:]:
                callback(*args, **kwargs)

    def once(self, event: str, callback: callable) -> None:
        """Register a callback that is automatically removed after it fires once."""
        def wrapper(*args, **kwargs):
            callback(*args, **kwargs)
            self.off(event, wrapper)
        self.on(event, wrapper)

    def listener_count(self, event: str) -> int:
        """Return the number of listeners registered for the given event."""
        return len(self._events.get(event, []))
```

</details>

### [PASS (attempt 1)] oop_005: ValidatedField Descriptor

- **Category:** oop
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
class ValidatedField:
    """A descriptor that validates field values on assignment.

    Constructor:
        __init__(self, field_type, *, min_value=None, max_value=None, max_length=None)

    Descriptor Protocol:
        __set_name__(self, owner, name):
            Store the field name. Use a private storage attribute like
            '_validated_' + name on the instance to avoid conflicts.

        __get__(self, obj, objtype=None):
            If obj is None (class-level access), return the descriptor itself.
            Otherwise return the stored value.
            Raise AttributeError if the value has not been set yet.

        __set__(self, obj, value):
            Validate the value before storing:
            1. Check isinstance(value, field_type). Raise TypeError if not.
            2. If field_type is int or float and min_value is set, raise ValueError
               if value < min_value.
            3. If field_type is int or float and max_value is set, raise ValueError
               if value > max_value.
            4. If field_type is str and max_length is set, raise ValueError
               if len(value) > max_length.
    """

    def __init__(self, field_type, *, min_value=None, max_value=None, max_length=None):
        self.field_type = field_type
        self.min_value = min_value
        self.max_value = max_value
        self.max_length = max_length
        self.name = None

    def __set_name__(self, owner, name):
        self.name = '_validated_' + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.name):
            raise AttributeError(f"'{type(obj).__name__}' object has no attribute '{self.name[11:]}'")
        return getattr(obj, self.name)

    def __set__(self, obj, value):
        if not isinstance(value, self.field_type):
            raise TypeError(f"'{type(value).__name__}' is not an instance of '{self.field_type.__name__}'")

        if self.field_type in (int, float):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"Value must be >= {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"Value must be <= {self.max_value}")

        if self.field_type is str and self.max_length is not None and len(value) > self.max_length:
            raise ValueError(f"String length must be <= {self.max_length}")

        setattr(obj, self.name, value)


class Person:
    """A class using ValidatedField descriptors.

    Class-level descriptor fields:
        name = ValidatedField(str, max_length=50)
        age = ValidatedField(int, min_value=0, max_value=200)
        email = ValidatedField(str, max_length=100)

    Constructor:
        __init__(self, name: str, age: int, email: str)
            Assign all three arguments to self (triggering descriptor validation).
    """

    name = ValidatedField(str, max_length=50)
    age = ValidatedField(int, min_value=0, max_value=200)
    email = ValidatedField(str, max_length=100)

    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email
```

</details>

### [PASS (attempt 1)] std_001: Count Words

- **Category:** stdlib
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
from collections import Counter

def count_words(text: str) -> dict[str, int]:
    """Count the frequency of each word in a string.

    Words are separated by whitespace. Counting is case-sensitive
    (i.e., 'Hello' and 'hello' are different words).

    Args:
        text: A string of words separated by whitespace.

    Returns:
        A dictionary mapping each word to the number of times it appears.
    """
    words = text.split()
    return dict(Counter(words))
```

</details>

### [PASS (attempt 1)] std_002: Parse Date

- **Category:** stdlib
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
from datetime import datetime

def parse_date(date_str: str) -> str:
    """Parse a date string in 'YYYY-MM-DD' format and return the day of the week.

    Args:
        date_str: A date string in the format 'YYYY-MM-DD'.

    Returns:
        The day of the week as a full string, e.g. 'Monday', 'Tuesday', etc.
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.strftime('%A')
```

</details>

### [FAIL] std_003: Validate Email

- **Category:** stdlib
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API request timed out

### [PASS (attempt 2)] std_004: Cartesian Product Filter

- **Category:** stdlib
- **Difficulty:** medium
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
from itertools import product

def cartesian_filter(lists: list[list[int]], threshold: int) -> list[tuple[int, ...]]:
    """Generate the cartesian product of multiple lists of integers and
    filter to keep only tuples whose element sum exceeds a threshold.

    Args:
        lists: A list of lists of integers. The cartesian product is
               computed across all provided lists.
        threshold: An integer threshold. Only tuples whose sum is
                   strictly greater than this value are included.

    Returns:
        A list of tuples (in the order produced by itertools.product)
        where the sum of elements is strictly greater than the threshold.
        Returns an empty list if lists is empty.
    """
    if not lists:
        return []
    
    result = []
    for combo in product(*lists):
        if sum(combo) > threshold:
            result.append(combo)
    
    return result
```

</details>

### [FAIL] std_005: Parallel Map

- **Category:** stdlib
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 500: Internal Server Error

### [PASS (attempt 1)] str_001: Is Palindrome

- **Category:** string_processing
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import re

def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome, ignoring case, spaces, and punctuation.

    Only consider alphanumeric characters when determining if the string
    is a palindrome.

    Args:
        s: The input string

    Returns:
        True if the string is a palindrome, False otherwise

    Examples:
        is_palindrome('racecar') -> True
        is_palindrome('A man, a plan, a canal: Panama') -> True
        is_palindrome('hello') -> False
        is_palindrome('') -> True
    """
    # Remove all non-alphanumeric characters and convert to lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    # Check if the cleaned string equals its reverse
    return cleaned == cleaned[::-1]
```

</details>

### [PASS (attempt 1)] str_002: Caesar Cipher

- **Category:** string_processing
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def caesar_cipher(text: str, shift: int, mode: str = 'encrypt') -> str:
    """
    Encrypt or decrypt text using a Caesar cipher.

    Shift each letter by the given amount. Preserve case. Non-alphabetic
    characters (spaces, punctuation, digits) are left unchanged.

    For encryption, shift letters forward in the alphabet.
    For decryption, shift letters backward.    The shift value can be any integer and will be normalized mod 26.

    Args:
        text: The input string
        shift: Integer shift value
        mode: 'encrypt' or 'decrypt'

    Returns:
        The encrypted or decrypted string

    Examples:
        caesar_cipher('Hello, World!', 3, 'encrypt') -> 'Khoor, Zruog!'
        caesar_cipher('Khoor, Zruog!', 3, 'decrypt') -> 'Hello, World!'
        caesar_cipher('abc', 1, 'encrypt') -> 'bcd'
        caesar_cipher('xyz', 3, 'encrypt') -> 'abc'
    """
    result = []
    
    # Normalize shift based on mode
    if mode == 'decrypt':
        shift = -shift
    
    # Normalize shift to be within 0-25
    shift = shift % 26
    
    for char in text:
        if char.isalpha():
            # Determine the ASCII base (uppercase or lowercase)
            base = ord('A') if char.isupper() else ord('a')
            # Shift the character and wrap around using modulo
            shifted = (ord(char) - base + shift) % 26
            result.append(chr(base + shifted))
        else:
            # Non-alphabetic characters remain unchanged
            result.append(char)
    
    return ''.join(result)
```

</details>

### [PASS (attempt 2)] str_003: Extract URLs

- **Category:** string_processing
- **Difficulty:** medium
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
import re

def extract_urls(text: str) -> list:
    """
    Extract all URLs from a text string.

    Find all URLs that start with http:// or https:// followed by
    one or more non-whitespace characters. URLs should not end with
    trailing punctuation marks (period, comma, semicolon, exclamation
    mark, question mark, closing parenthesis) that are likely part of
    the surrounding text rather than the URL itself.

    Args:
        text: The input string that may contain URLs

    Returns:
        A list of URL strings found in the text, in order of appearance

    Examples:
        extract_urls('Visit https://example.com for more info')
        -> ['https://example.com']

        extract_urls('No URLs here')
        -> []

        extract_urls('Go to http://a.com and https://b.com/path')
        -> ['http://a.com', 'https://b.com/path']
    """
    # Pattern to match URLs starting with http:// or https://
    # followed by one or more non-whitespace characters
    pattern = r'https?://\S+'
    
    # Find all potential URL matches
    urls = re.findall(pattern, text)
    
    # Punctuation marks to strip from end of URLs
    trailing_punctuation = '.,;:!?)}'
    
    # Clean URLs by removing trailing punctuation
    cleaned_urls = []
    for url in urls:
        # Strip trailing punctuation characters
        while url and url[-1] in trailing_punctuation:
            url = url[:-1]
        if url:
            cleaned_urls.append(url)
    
    return cleaned_urls
```

</details>

### [PASS (attempt 1)] str_004: Template Engine

- **Category:** string_processing
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import re

def template_engine(template: str, context: dict) -> str:
    """
    Implement a simple template engine that replaces placeholders with values.

    Placeholders are enclosed in double curly braces: {{key}}

    Features:
    - Simple replacement: {{name}} is replaced with context['name']
    - Nested access: {{user.name}} accesses context['user']['name']
    - Default values: {{key|default}} uses 'default' if key is missing
    - If a key is not found and no default is given, leave the placeholder as-is
    - Strip whitespace inside braces: {{ name }} is the same as {{name}}
    - Values are converted to strings using str()

    Args:
        template: A string with {{placeholder}} markers
        context: A dictionary of values for substitution

    Returns:
        The template string with placeholders replaced

    Examples:
        template_engine('Hello {{name}}!', {'name': 'World'})
        -> 'Hello World!'

        template_engine('{{user.name}} is {{user.age}}',
                        {'user': {'name': 'Alice', 'age': 30}})
        -> 'Alice is 30'

        template_engine('{{missing|default_val}}', {})
        -> 'default_val'

        template_engine('{{missing}}', {})
        -> '{{missing}}'
    """
    def replace_placeholder(match):
        # Get the content inside {{ }}
        content = match.group(1)
        
        # Strip whitespace
        content = content.strip()
        
        # Check for default value
        default_value = None
        if '|' in content:
            parts = content.split('|', 1)
            key_part = parts[0].strip()
            default_value = parts[1].strip()
        else:
            key_part = content
        
        # Handle empty key - return original placeholder
        if not key_part:
            return match.group(0)
        
        # Try to get the value from context
        try:
            # Handle nested access
            keys = key_part.split('.')
            value = context
            for key in keys:
                value = value[key]
            return str(value)
        except (KeyError, TypeError, IndexError):
            # Key not found
            if default_value is not None:
                return default_value
            else:
                # Return original placeholder if no default
                return match.group(0)
    
    # Use regex to find all {{...}} patterns (non-greedy to handle multiple placeholders)
    result = re.sub(r'\{\{(.+?)\}\}', replace_placeholder, template)
    return result
```

</details>
