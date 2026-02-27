# PyEval Report: kimi-k2.5 [Multi-turn]

**Date:** 2026-02-27 07:56:31 UTC  
**Mode:** Multi-turn  
**Overall Pass Rate:** 37.8%  
**Weighted Score:** 34.3%  
**Total:** 17/45

**Strict Pass Rate (1st try):** 33.3%  
**With Retries Pass Rate:** 37.8%  

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| Basic Syntax | 2/3 | 0/1 | 0/1 | 2/5 |
| Data Structures | 0/2 | 0/2 | 0/1 | 0/5 |
| Algorithms | 0/2 | 0/3 | 0/2 | 0/7 |
| Standard Library | 0/2 | 0/2 | 0/1 | 0/5 |
| OOP | 0/1 | 0/2 | 0/2 | 0/5 |
| Exceptions | 0/1 | 0/1 | 0/1 | 0/3 |
| File I/O | 1/1 | 1/1 | 1/1 | 3/3 |
| String Processing | 2/2 | 1/1 | 1/1 | 4/4 |
| Functional | 1/1 | 2/2 | 1/1 | 4/4 |
| Concurrency | 1/1 | 2/2 | 1/1 | 4/4 |
| **TOTAL** | **7/16** | **6/17** | **4/12** | **17/45** |

## Attempt Distribution

- Attempt 1 (weight=1.0): 15 passed
- Attempt 2 (weight=0.6): 1 passed
- Attempt 3 (weight=0.3): 1 passed

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

### [FAIL] algo_003: Merge Sort

- **Category:** algorithms
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] algo_004: Longest Common Subsequence

- **Category:** algorithms
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] algo_005: Valid Parentheses

- **Category:** algorithms
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] algo_006: 0/1 Knapsack

- **Category:** algorithms
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] algo_007: Topological Sort

- **Category:** algorithms
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] bs_001: FizzBuzz

- **Category:** basic_syntax
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [PASS (attempt 1)] bs_002: Swap Variables

- **Category:** basic_syntax
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def swap_variables(d: dict) -> dict:
    """
    Given a dictionary with keys 'a' and 'b', return a new dictionary
    where the values of 'a' and 'b' are swapped.

    The original dictionary must not be modified.

    Example:
        swap_variables({'a': 1, 'b': 2}) -> {'a': 2, 'b': 1}
        swap_variables({'a': 'hello', 'b': 'world'}) -> {'a': 'world', 'b': 'hello'}
    """
    return {'a': d['b'], 'b': d['a']}
```

</details>

### [PASS (attempt 1)] bs_003: Number to Words

- **Category:** basic_syntax
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def number_to_words(n: int) -> str:
    """
    Convert an integer from 0 to 999 (inclusive) to its English words representation.

    Rules:
    - 0 -> "Zero"
    - Use standard English number names
    - Hyphenate compound numbers 21-99 (e.g., "Twenty-One")
    - Use title case (e.g., "One Hundred Forty-Five")
    - No "and" between hundred and tens/ones (e.g., "One Hundred Twelve", not "One Hundred and Twelve")

    Examples:
        number_to_words(0) -> "Zero"
        number_to_words(5) -> "Five"
        number_to_words(42) -> "Forty-Two"
        number_to_words(100) -> "One Hundred"
        number_to_words(999) -> "Nine Hundred Ninety-Nine"
        number_to_words(115) -> "One Hundred Fifteen"
    """
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    
    if n == 0:
        return "Zero"
    
    result = []
    
    # Handle hundreds
    if n >= 100:
        result.append(ones[n // 100])
        result.append("Hundred")
        n %= 100
    
    # Handle tens and ones
    if n >= 20:
        ten_word = tens[n // 10]
        n %= 10
        if n > 0:
            ten_word += "-" + ones[n]
        result.append(ten_word)
    elif n >= 10:
        result.append(teens[n - 10])
    elif n > 0:
        result.append(ones[n])
    
    return " ".join(result)
```

</details>

### [FAIL] bs_004: Flatten Nested List

- **Category:** basic_syntax
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

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

### [PASS (attempt 1)] con_002: Parallel Map

- **Category:** concurrency
- **Difficulty:** medium
- **Attempts:** 1

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
        parallel_map(lambda x: x ** 2, [1, 2, 3, 4])  # -> [1, 4, 9, 16]
        parallel_map(str.upper, ["hello", "world"])  # -> ["HELLO", "WORLD"]
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
            returns a list of items to be placed on the queue.
        consumers: A list of callables. Each consumer callable takes a single
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
    q = queue.Queue(maxsize=queue_size)
    results = []
    results_lock = threading.Lock()
    
    def consumer_worker(consumer_func):
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break
            result = consumer_func(item)
            with results_lock:
                results.append(result)
            q.task_done()
    
    def producer_worker(producer_func):
        items = producer_func()
        for item in items:
            q.put(item)
    
    # Start consumer threads first
    consumer_threads = []
    for consumer in consumers:
        t = threading.Thread(target=consumer_worker, args=(consumer,))
        t.start()
        consumer_threads.append(t)
    
    # Start producer threads
    producer_threads = []
    for producer in producers:
        t = threading.Thread(target=producer_worker, args=(producer,))
        t.start()
        producer_threads.append(t)
    
    # Wait for all producers to finish
    for t in producer_threads:
        t.join()
    
    # Send sentinel values to stop consumers
    for _ in consumers:
        q.put(None)
    
    # Wait for all consumers to finish
    for t in consumer_threads:
        t.join()
    
    return results
```

</details>

### [PASS (attempt 1)] con_004: Async Gather with Timeout

- **Category:** concurrency
- **Difficulty:** hard
- **Attempts:** 1

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
    coroutines = [callable_() for callable_ in async_callables]
    gather_task = asyncio.gather(*coroutines)
    
    if timeout is not None:
        return await asyncio.wait_for(gather_task, timeout=timeout)
    else:
        return await gather_task
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

### [FAIL] exc_001: Safe Divide

- **Category:** exceptions
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] exc_002: Retry Decorator

- **Category:** exceptions
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] exc_003: Custom Exception Hierarchy and Schema Validation

- **Category:** exceptions
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [PASS (attempt 3)] fio_001: Read CSV Data

- **Category:** file_io
- **Difficulty:** easy
- **Attempts:** 3

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
    f = io.StringIO(csv_string)
    reader = csv.DictReader(f)
    return list(reader)
```

</details>

### [PASS (attempt 2)] fio_002: Merge JSON Files

- **Category:** file_io
- **Difficulty:** medium
- **Attempts:** 2

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
    def deep_merge(base: dict, update: dict) -> dict:
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    result = {}
    for json_str in json_strings:
        data = json.loads(json_str)
        result = deep_merge(result, data)
    return result
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
    """
    Parse a list of log line strings and extract structured information.

    Each log line is in the format:
        [TIMESTAMP] LEVEL: message
    where TIMESTAMP is in 'YYYY-MM-DD HH:MM:SS' format and LEVEL is one
    of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns a dictionary with:
        - 'counts': dict mapping each log level to its count
        - 'errors': list of message strings from ERROR and CRITICAL lines
          (in order of appearance)
        - 'timestamp_range': tuple of (earliest, latest) timestamp strings,
          or None if no log lines are provided

    Args:
        log_lines: List of log line strings

    Returns:
        Dict with 'counts', 'errors', and 'timestamp_range' keys

    Examples:
        log_parser(['[2024-01-15 10:30:45] ERROR: Connection failed',
                     '[2024-01-15 10:31:00] INFO: Retrying'])
        -> {
            'counts': {'ERROR': 1, 'INFO': 1},
            'errors': ['Connection failed'],
            'timestamp_range': ('2024-01-15 10:30:45', '2024-01-15 10:31:00')
        }
    """
    # Pattern to match log lines: [TIMESTAMP] LEVEL: message
    pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (DEBUG|INFO|WARNING|ERROR|CRITICAL): (.+)'
    
    counts = {}
    errors = []
    timestamps = []
    
    for line in log_lines:
        match = re.match(pattern, line.strip())
        if match:
            timestamp, level, message = match.groups()
            
            # Update counts
            counts[level] = counts.get(level, 0) + 1
            
            # Collect error messages
            if level in ('ERROR', 'CRITICAL'):
                errors.append(message)
            
            # Collect timestamps
            timestamps.append(timestamp)
    
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
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
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

### [PASS (attempt 1)] fn_004: Lazy Evaluation

- **Category:** functional
- **Difficulty:** hard
- **Attempts:** 1

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
        Lazy([1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0).map(lambda x: x * 10).evaluate()
        # -> [20, 40]

        lazy = Lazy(range(10))
        lazy.map(lambda x: x ** 2).filter(lambda x: x > 10).evaluate()
        # -> [16, 25, 36, 49, 64, 81]

        Lazy([]).evaluate()  # -> []
    """

    def __init__(self, iterable):
        self._iterable = iterable
        self._operations = []

    def map(self, func):
        self._operations.append(('map', func))
        return self

    def filter(self, predicate):
        self._operations.append(('filter', predicate))
        return self

    def evaluate(self):
        result = list(self._iterable)
        for op_type, func in self._operations:
            if op_type == 'map':
                result = [func(x) for x in result]
            elif op_type == 'filter':
                result = [x for x in result if func(x)]
        self._operations = []
        return result
```

</details>

### [FAIL] oop_001: Stack Class

- **Category:** oop
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] oop_002: Shape Hierarchy with Abstract Base Class

- **Category:** oop
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] oop_003: BankAccount with Transaction History

- **Category:** oop
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] oop_004: EventEmitter - Observer Pattern

- **Category:** oop
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] oop_005: ValidatedField Descriptor

- **Category:** oop
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] std_001: Count Words

- **Category:** stdlib
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] std_002: Parse Date

- **Category:** stdlib
- **Difficulty:** easy
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] std_003: Validate Email

- **Category:** stdlib
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] std_004: Cartesian Product Filter

- **Category:** stdlib
- **Difficulty:** medium
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

### [FAIL] std_005: Parallel Map

- **Category:** stdlib
- **Difficulty:** hard
- **Attempts:** 3
- **Error:** API error: HTTP Error 429: Too Many Requests

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
    For decryption, shift letters backward.

    The shift value can be any integer and will be normalized mod 26.

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
    # Normalize shift to be within 0-25
    shift = shift % 26
    
    # Adjust shift for decryption
    if mode == 'decrypt':
        shift = -shift
    elif mode != 'encrypt':
        raise ValueError("mode must be 'encrypt' or 'decrypt'")
    
    result = []
    
    for char in text:
        if char.isalpha():
            # Determine base ASCII value ('A' for uppercase, 'a' for lowercase)
            base = ord('A') if char.isupper() else ord('a')
            # Calculate shifted position
            shifted = (ord(char) - base + shift) % 26
            # Convert back to character
            new_char = chr(base + shifted)
            result.append(new_char)
        else:
            # Non-alphabetic characters remain unchanged
            result.append(char)
    
    return ''.join(result)
```

</details>

### [PASS (attempt 1)] str_003: Extract URLs

- **Category:** string_processing
- **Difficulty:** medium
- **Attempts:** 1

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
    # followed by non-whitespace characters
    pattern = r'https?://\S+'
    
    # Find all potential URLs
    matches = re.findall(pattern, text)
    
    # Strip trailing punctuation that is likely not part of the URL
    result = []
    for url in matches:
        # Remove trailing punctuation marks
        stripped_url = url.rstrip('.,;!?)]')
        result.append(stripped_url)
    
    return result
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
    def get_value(key_path: str, ctx: dict):
        """Get value from nested dictionary using dot notation."""
        keys = key_path.split('.')
        current = ctx
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def replace_placeholder(match):
        content = match.group(1).strip()
        
        # Check for default value
        if '|' in content:
            key_path, default = content.split('|', 1)
            key_path = key_path.strip()
            default = default.strip()
        else:
            key_path = content
            default = None
        
        value = get_value(key_path, context)
        
        if value is None:
            if default is not None:
                return default
            else:
                # Return original placeholder if key not found and no default
                return match.group(0)
        else:
            return str(value)
    
    # Pattern to match {{...}}
    pattern = r'\{\{(.*?)\}\}'
    return re.sub(pattern, replace_placeholder, template)
```

</details>
