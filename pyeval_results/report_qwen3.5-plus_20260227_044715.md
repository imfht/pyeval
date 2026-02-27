# PyEval Report: qwen3.5-plus

**Date:** 2026-02-27 04:47:15 UTC  
**Overall Pass Rate:** 91.1%  
**Weighted Score:** 88.4% (easy=1x, medium=2x, hard=3x)  
**Total:** 41/45

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| Basic Syntax | 3/3 | 1/1 | 1/1 | 5/5 |
| Data Structures | 2/2 | 2/2 | 1/1 | 5/5 |
| Algorithms | 2/2 | 3/3 | 2/2 | 7/7 |
| Standard Library | 2/2 | 1/2 | 1/1 | 4/5 |
| OOP | 1/1 | 1/2 | 1/2 | 3/5 |
| Exceptions | 1/1 | 1/1 | 0/1 | 2/3 |
| File I/O | 1/1 | 1/1 | 1/1 | 3/3 |
| String Processing | 2/2 | 1/1 | 1/1 | 4/4 |
| Functional | 1/1 | 2/2 | 1/1 | 4/4 |
| Concurrency | 1/1 | 2/2 | 1/1 | 4/4 |
| **TOTAL** | **16/16** | **15/17** | **10/12** | **41/45** |

## Detailed Results

### [PASS] algo_001: Binary Search

- **Category:** algorithms
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def binary_search(arr: list[int], target: int) -> int:
    """Search for target in a sorted list of integers using binary search.

    Args:
        arr: A sorted list of integers in ascending order.
        target: The integer to search for.

    Returns:
        The index of target in arr if found, otherwise -1.
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

</details>

### [PASS] algo_002: Bubble Sort

- **Category:** algorithms
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def bubble_sort(arr: list[int]) -> list[int]:
    """Sort a list of integers in ascending order using bubble sort.

    Args:
        arr: A list of integers to sort.

    Returns:
        A new list containing the same elements sorted in ascending order.
        The original list must not be modified.
    """
    # Create a copy to avoid modifying the original list
    result = arr.copy()
    n = len(result)
    
    for i in range(n):
        # Flag to optimize: if no swaps occur, the list is already sorted
        swapped = False
        
        # Last i elements are already in place after i passes
        for j in range(0, n - 1 - i):
            if result[j] > result[j + 1]:
                # Swap elements
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        
        # If no two elements were swapped in the inner loop, break early
        if not swapped:
            break
    
    return result
```

</details>

### [PASS] algo_003: Merge Sort

- **Category:** algorithms
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
def merge_sort(arr: list[int]) -> list[int]:
    """Sort a list of integers in ascending order using merge sort.

    Args:
        arr: A list of integers to sort.

    Returns:
        A new sorted list. The original list must not be modified.
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    merged = []
    i, j = 0, 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged
```

</details>

### [PASS] algo_004: Longest Common Subsequence

- **Category:** algorithms
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
def longest_common_subsequence(s1: str, s2: str) -> int:
    """Find the length of the longest common subsequence of two strings.

    A subsequence is a sequence that can be derived from another sequence
    by deleting some or no elements without changing the order of the
    remaining elements.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The length of the longest common subsequence.
    """
    m, n = len(s1), len(s2)
    
    # Create a 2D DP table with (m+1) rows and (n+1) columns
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]
```

</details>

### [PASS] algo_005: Valid Parentheses

- **Category:** algorithms
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
    mapping = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # According to the docstring, input contains only bracket characters,
            # but this handles unexpected characters gracefully by returning False.
            return False

    return len(stack) == 0
```

</details>

### [PASS] algo_006: 0/1 Knapsack

- **Category:** algorithms
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
    if n == 0 or capacity <= 0:
        return 0

    # dp[w] will store the maximum value achievable with capacity w
    dp = [0] * (capacity + 1)

    for i in range(n):
        weight = weights[i]
        value = values[i]
        # Traverse backwards to ensure each item is only considered once
        for w in range(capacity, weight - 1, -1):
            dp[w] = max(dp[w], dp[w - weight] + value)

    return dp[capacity]
```

</details>

### [PASS] algo_007: Topological Sort

- **Category:** algorithms
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
    # Compute in-degrees for all nodes
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Initialize queue with nodes having in-degree 0
    queue = [node for node in graph if in_degree[node] == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)

        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If result contains all nodes, it's a valid topological sort; otherwise, there's a cycle
    if len(result) == len(graph):
        return result
    else:
        return []
```

</details>

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

### [PASS] bs_002: Swap Variables

- **Category:** basic_syntax
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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

### [PASS] bs_003: Number to Words

- **Category:** basic_syntax
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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
    if not 0 <= n <= 999:
        raise ValueError("Input must be between 0 and 999 inclusive.")

    ones = [
        "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
        "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
        "Seventeen", "Eighteen", "Nineteen"
    ]

    tens = [
        "", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"
    ]

    if n < 20:
        return ones[n]

    if n < 100:
        ten_digit = n // 10
        one_digit = n % 10
        if one_digit == 0:
            return tens[ten_digit]
        else:
            return f"{tens[ten_digit]}-{ones[one_digit]}"

    # n >= 100
    hundred_digit = n // 100
    remainder = n % 100

    if remainder == 0:
        return f"{ones[hundred_digit]} Hundred"
    else:
        return f"{ones[hundred_digit]} Hundred {number_to_words(remainder)}"
```

</details>

### [PASS] bs_004: Flatten Nested List

- **Category:** basic_syntax
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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

    def _flatten(item):
        if isinstance(item, list):
            for sub_item in item:
                _flatten(sub_item)
        else:
            result.append(item)

    _flatten(lst)
    return result
```

</details>

### [PASS] bs_005: Matrix Spiral Order

- **Category:** basic_syntax
- **Difficulty:** hard

<details><summary>Generated Code</summary>

```python
def matrix_spiral(matrix: list) -> list:
    """
    Given an MxN matrix (list of lists of integers), return all elements
    in spiral order (clockwise, starting from top-left).

    The spiral traversal goes: right across the top row, down the right column,
    left across the bottom row, up the left column, then inward.

    Examples:
        matrix_spiral([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]) -> [1, 2, 3, 6, 9, 8, 7, 4, 5]

        matrix_spiral([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]]) -> [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]

        matrix_spiral([]) -> []
        matrix_spiral([[1]]) -> [1]
    """
    if not matrix or not matrix[0]:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse right along the top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Traverse down along the right column
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        # Traverse left along the bottom row (if still valid)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        # Traverse up along the left column (if still valid)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result
```

</details>

### [PASS] con_001: Thread-Safe Counter

- **Category:** concurrency
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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

### [PASS] con_002: Parallel Map

- **Category:** concurrency
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
        # Submit all tasks and keep track of their order
        futures = [executor.submit(func, item) for item in items]
        # Collect results in the original order
        results = [future.result() for future in futures]

    return results
```

</details>

### [PASS] con_003: Producer Consumer

- **Category:** concurrency
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
            try:
                result = consumer_func(item)
                with results_lock:
                    results.append(result)
            finally:
                q.task_done()

    def producer_worker(producer_func):
        items = producer_func()
        for item in items:
            q.put(item)

    consumer_threads = []
    for _ in range(len(consumers)):
        t = threading.Thread(target=consumer_worker, args=(consumers[_ % len(consumers)],))
        t.start()
        consumer_threads.append(t)

    producer_threads = []
    for prod_func in producers:
        t = threading.Thread(target=producer_worker, args=(prod_func,))
        t.start()
        producer_threads.append(t)

    for t in producer_threads:
        t.join()

    for _ in consumers:
        q.put(None)

    for t in consumer_threads:
        t.join()

    return results
```

</details>

### [PASS] con_004: Async Gather with Timeout

- **Category:** concurrency
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
    
    if timeout is not None:
        return await asyncio.wait_for(asyncio.gather(*coroutines), timeout=timeout)
    else:
        return await asyncio.gather(*coroutines)
```

</details>

### [PASS] ds_001: Two Sum

- **Category:** data_structures
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def two_sum(nums: list, target: int) -> list:
    """
    Given a list of integers and a target integer, return the indices of the
    two numbers that add up to the target.

    Assumptions:
    - Each input has exactly one solution.
    - You may not use the same element twice.
    - Return the indices in ascending order.

    Examples:
        two_sum([2, 7, 11, 15], 9) -> [0, 1]
        two_sum([3, 2, 4], 6) -> [1, 2]
        two_sum([3, 3], 6) -> [0, 1]
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return sorted([seen[complement], i])
        seen[num] = i
    # According to the problem assumptions, this line should never be reached
    raise ValueError("No solution found")
```

</details>

### [PASS] ds_002: Reverse Linked List

- **Category:** data_structures
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
def reverse_linked_list(values: list) -> list:
    """
    Given a list of values representing a singly linked list (head to tail),
    return a new list with the values in reversed order.

    This simulates reversing a linked list using list representation.
    You should implement the reversal logic iteratively, simulating
    pointer manipulation (not just using built-in reverse).

    Examples:
        reverse_linked_list([1, 2, 3, 4, 5]) -> [5, 4, 3, 2, 1]
        reverse_linked_list([1]) -> [1]
        reverse_linked_list([]) -> []
    """
    # Simulate iterative linked list reversal using three pointers:
    # prev, current, and next_node. We'll build the reversed list by
    # prepending each node's value as we traverse.

    prev = None
    current = 0
    n = len(values)
    reversed_values = []

    # First pass: simulate traversal and collect values in reverse order
    # by iterating from head to tail and prepending each value.
    while current < n:
        # Save next index
        next_index = current + 1

        # Prepend current value to the front of reversed_values
        # This simulates changing the next pointer to point to previous node
        reversed_values.insert(0, values[current])

        # Move to next node
        current = next_index

    return reversed_values
```

</details>

### [PASS] ds_003: LRU Cache

- **Category:** data_structures
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
from collections import OrderedDict

class LRUCache:
    """
    Implement a Least Recently Used (LRU) Cache.

    The cache has a fixed capacity. When the cache is full and a new key
    is inserted, the least recently used key is evicted.

    Both get and put operations count as "using" a key.

    Methods:
        __init__(capacity: int) - Initialize the cache with positive capacity.
        get(key: int) -> int - Return the value if key exists, otherwise -1.
        put(key: int, value: int) -> None - Insert or update the key-value pair.

    Example:
        cache = LRUCache(2)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.get(1)       # returns 1
        cache.put(3, 3)    # evicts key 2
        cache.get(2)       # returns -1 (not found)
        cache.get(3)       # returns 3
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Move the accessed key to the end to mark it as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing key and move it to the end
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Pop the first item (least recently used)
            self.cache.popitem(last=False)
```

</details>

### [PASS] ds_004: Group Anagrams

- **Category:** data_structures
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
def group_anagrams(strs: list) -> list:
    """
    Given a list of strings, group the anagrams together.
    Return a list of groups, where each group is a sorted list of anagram strings.
    The groups themselves should be sorted by their first element.

    Two strings are anagrams if they contain the same characters with the
    same frequencies, regardless of order.

    Examples:
        group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
        -> [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]

        group_anagrams([""]) -> [[""]]
        group_anagrams(["a"]) -> [["a"]]
    """
    from collections import defaultdict

    anagram_groups = defaultdict(list)

    for s in strs:
        key = ''.join(sorted(s))
        anagram_groups[key].append(s)

    result = []
    for group in anagram_groups.values():
        sorted_group = sorted(group)
        result.append(sorted_group)

    result.sort(key=lambda x: x[0])

    return result
```

</details>

### [PASS] ds_005: Merge K Sorted Lists

- **Category:** data_structures
- **Difficulty:** hard

<details><summary>Generated Code</summary>

```python
import heapq

def merge_k_sorted(lists: list) -> list:
    """
    Merge k sorted lists into one sorted list.

    Use a min-heap for efficient merging. Each input list is sorted
    in ascending order. Return a single sorted list containing all elements.

    Examples:
        merge_k_sorted([[1, 4, 5], [1, 3, 4], [2, 6]])
        -> [1, 1, 2, 3, 4, 4, 5, 6]

        merge_k_sorted([]) -> []
        merge_k_sorted([[], []]) -> []
        merge_k_sorted([[1]]) -> [1]
    """
    if not lists:
        return []

    min_heap = []
    result = []

    # Initialize the heap with the first element of each non-empty list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))

    while min_heap:
        val, list_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)

        # If there's a next element in the same list, push it to the heap
        next_elem_idx = elem_idx + 1
        if next_elem_idx < len(lists[list_idx]):
            next_val = lists[list_idx][next_elem_idx]
            heapq.heappush(min_heap, (next_val, list_idx, next_elem_idx))

    return result
```

</details>

### [PASS] exc_001: Safe Divide

- **Category:** exceptions
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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
        raise TypeError("Both arguments must be numbers (int or float).")
    
    if b == 0:
        return None
    
    return a / b
```

</details>

### [PASS] exc_002: Retry Decorator

- **Category:** exceptions
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
                    else:
                        raise last_exception
            # This line should theoretically never be reached due to the raise above,
            # but included for completeness in case max_attempts < 1 (though spec says >= 1)
            if last_exception is not None:
                raise last_exception
        return wrapper
    return decorator
```

</details>

### [FAIL] exc_003: Custom Exception Hierarchy and Schema Validation

- **Category:** exceptions
- **Difficulty:** hard

<details><summary>Generated Code</summary>

```python
import re


class ValidationError(Exception):
    """Base exception for validation errors.

    Attributes:
        field (str): The name of the field that failed validation.
        message (str): A human-readable error message.
    """


class TypeValidationError(ValidationError):
    """Raised when a field has the wrong type.

    Attributes:
        field (str): The field name.
        expected_type (type): The expected type.
        actual_type (type): The actual type received.
    """


class RangeValidationError(ValidationError):
    """Raised when a numeric field is out of range.

    Attributes:
        field (str): The field name.
        value: The actual value.
        min_value: The minimum allowed value (or None).
        max_value: The maximum allowed value (or None).
    """


class PatternValidationError(ValidationError):
    """Raised when a string field does not match a regex pattern.

    Attributes:
        field (str): The field name.
        pattern (str): The regex pattern that was expected.
        value (str): The actual string value.
    """


def validate_data(schema, data):
    """Validate data against a schema, raising appropriate exceptions.

    Args:
        schema (dict): A dict mapping field names to rule dicts. Each rule dict
            can have:
            - 'type' (type): required type (e.g., int, str, float)
            - 'min' (number): minimum value (only for numeric types)
            - 'max' (number): maximum value (only for numeric types)
            - 'pattern' (str): regex pattern the string must match fully
            - 'required' (bool): if True, field must exist in data (default True)

        data (dict): The data to validate.

    Raises:
        TypeValidationError: if a field has the wrong type.
        RangeValidationError: if a numeric field is out of range.
        PatternValidationError: if a string field doesn't match the pattern.
        ValidationError: if a required field is missing.

    Returns:
        True if all validations pass.

    Note:
        Validate fields in the order they appear in the schema.
        Raise on the first validation failure (do not collect all errors).
    """
    for field, rules in schema.items():
        required = rules.get('required', True)
        
        # Check if field exists
        if field not in data:
            if required:
                raise ValidationError(field=field, message=f"Missing required field: {field}")
            else:
                continue
        
        value = data[field]
        
        # Type check
        if 'type' in rules:
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                raise TypeValidationError(
                    field=field,
                    expected_type=expected_type,
                    actual_type=type(value)
                )
        
        # Range check (only for numeric types)
        if 'min' in rules or 'max' in rules:
            min_val = rules.get('min')
            max_val = rules.get('max')
            
            if min_val is not None and value < min_val:
                raise RangeValidationError(
                    field=field,
                    value=value,
                    min_value=min_val,
                    max_value=max_val
                )
            
            if max_val is not None and value > max_val:
                raise RangeValidationError(
                    field=field,
                    value=value,
                    min_value=min_val,
                    max_value=max_val
                )
        
        # Pattern check (only for strings)
        if 'pattern' in rules:
            pattern = rules['pattern']
            if not re.fullmatch(pattern, value):
                raise PatternValidationError(
                    field=field,
                    pattern=pattern,
                    value=value
                )
    
    return True
```

</details>

### [PASS] fio_001: Read CSV Data

- **Category:** file_io
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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

### [PASS] fio_002: Merge JSON Files

- **Category:** file_io
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
    def deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    merged = {}
    for json_str in json_strings:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Each JSON string must represent a dictionary.")
        merged = deep_merge(merged, data)

    return merged
```

</details>

### [PASS] fio_003: Log Parser

- **Category:** file_io
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
    pattern = re.compile(
        r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (DEBUG|INFO|WARNING|ERROR|CRITICAL): (.*)$'
    )
    
    counts = {}
    errors = []
    timestamps = []
    
    for line in log_lines:
        match = pattern.match(line)
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
    timestamp_range = None
    if timestamps:
        timestamps_sorted = sorted(timestamps)
        timestamp_range = (timestamps_sorted[0], timestamps_sorted[-1])
    
    return {
        'counts': counts,
        'errors': errors,
        'timestamp_range': timestamp_range
    }
```

</details>

### [PASS] fn_001: Apply Pipeline

- **Category:** functional
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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

### [PASS] fn_002: Memoize Decorator

- **Category:** functional
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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

### [PASS] fn_003: Group By

- **Category:** functional
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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

### [PASS] fn_004: Lazy Evaluation

- **Category:** functional
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
                result = [func(item) for item in result]
            elif op_type == 'filter':
                result = [item for item in result if func(item)]
        self._operations = []
        return result
```

</details>

### [PASS] oop_001: Stack Class

- **Category:** oop
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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
        self._items = []

    def push(self, item):
        """Add an item to the top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item. Raise IndexError if empty."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it. Raise IndexError if empty."""
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

### [PASS] oop_002: Shape Hierarchy with Abstract Base Class

- **Category:** oop
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
            raise ValueError("Radius must be positive.")
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
            raise ValueError("Width and height must be positive.")
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

### [FAIL] oop_003: BankAccount with Transaction History

- **Category:** oop
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
class BankAccount:
    """A bank account with deposit, withdraw, transfer, and transaction history.

    Args:
        owner (str): The name of the account owner.
        initial_balance (float): Starting balance. Defaults to 0.0.
            Raise ValueError if initial_balance < 0.

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
            Return a list of all transaction tuples in chronological order.
    """

    def __init__(self, owner: str, initial_balance: float = 0.0):
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative.")
        self._owner = owner
        self._balance = initial_balance
        self._history = []

    @property
    def balance(self) -> float:
        return self._balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")
        self._balance += amount
        self._history.append(('deposit', amount))

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        if amount > self._balance:
            raise ValueError("Insufficient funds.")
        self._balance -= amount
        self._history.append(('withdraw', amount))

    def transfer_to(self, other_account: 'BankAccount', amount: float) -> None:
        if amount <= 0:
            raise ValueError("Transfer amount must be positive.")
        if amount > self._balance:
            raise ValueError("Insufficient funds.")
        self._balance -= amount
        other_account._balance += amount
        self._history.append(('transfer_out', amount))
        other_account._history.append(('transfer_in', amount))

    def get_history(self) -> list:
        return list(self._history)
```

</details>

### [PASS] oop_004: EventEmitter - Observer Pattern

- **Category:** oop
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
        self._events = {}

    def on(self, event: str, callback: callable) -> None:
        if event not in self._events:
            self._events[event] = []
        self._events[event].append(callback)

    def off(self, event: str, callback: callable) -> None:
        if event not in self._events:
            return
        try:
            self._events[event].remove(callback)
        except ValueError:
            pass

    def emit(self, event: str, *args, **kwargs) -> None:
        if event not in self._events:
            return
        # Copy the list to avoid issues if callbacks modify the listener list during emission
        callbacks = self._events[event][:]
        for callback in callbacks:
            callback(*args, **kwargs)

    def once(self, event: str, callback: callable) -> None:
        def wrapper(*args, **kwargs):
            self.off(event, wrapper)
            callback(*args, **kwargs)

        self.on(event, wrapper)

    def listener_count(self, event: str) -> int:
        return len(self._events.get(event, []))
```

</details>

### [FAIL] oop_005: ValidatedField Descriptor

- **Category:** oop
- **Difficulty:** hard

<details><summary>Generated Code</summary>

```python
class ValidatedField:
    """A descriptor that validates field values on assignment.

    Args:
        field_type (type): The required type (e.g., int, float, str).
        min_value (optional): Minimum value for numeric types. Ignored for non-numeric.
        max_value (optional): Maximum value for numeric types. Ignored for non-numeric.
        max_length (optional): Maximum length for str type. Ignored for non-str.

    Behavior:
        - On __set__, validate the value:
            1. Check that isinstance(value, field_type). Raise TypeError if not.
            2. If field_type is int or float and min_value is set, raise ValueError
               if value < min_value.
            3. If field_type is int or float and max_value is set, raise ValueError
               if value > max_value.
            4. If field_type is str and max_length is set, raise ValueError
               if len(value) > max_length.
        - On __get__, return the stored value (or raise AttributeError if not set).
        - Use __set_name__ to capture the field name for storage.
    """

    def __init__(self, field_type, min_value=None, max_value=None, max_length=None):
        self.field_type = field_type
        self.min_value = min_value
        self.max_value = max_value
        self.max_length = max_length
        self.field_name = None

    def __set_name__(self, owner, name):
        self.field_name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if not hasattr(instance, f"_validated_{self.field_name}"):
            raise AttributeError(f"'{owner.__name__}' object has no attribute '{self.field_name}'")
        return getattr(instance, f"_validated_{self.field_name}")

    def __set__(self, instance, value):
        # Type check
        if not isinstance(value, self.field_type):
            raise TypeError(f"Expected {self.field_type.__name__}, got {type(value).__name__}")

        # Numeric range checks
        if self.field_type in (int, float):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"Value {value} is less than minimum {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"Value {value} is greater than maximum {self.max_value}")

        # String length check
        if self.field_type is str and self.max_length is not None:
            if len(value) > self.max_length:
                raise ValueError(f"String length {len(value)} exceeds maximum {self.max_length}")

        setattr(instance, f"_validated_{self.field_name}", value)


class Person:
    """A class using ValidatedField descriptors.

    Fields:
        name: ValidatedField(str, max_length=50)
        age: ValidatedField(int, min_value=0, max_value=200)
        email: ValidatedField(str, max_length=100)
    """

    name = ValidatedField(str, max_length=50)
    age = ValidatedField(int, min_value=0, max_value=200)
    email = ValidatedField(str, max_length=100)
```

</details>

### [PASS] std_001: Count Words

- **Category:** stdlib
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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
    if not text:
        return {}
    
    words = text.split()
    return dict(Counter(words))
```

</details>

### [PASS] std_002: Parse Date

- **Category:** stdlib
- **Difficulty:** easy

<details><summary>Generated Code</summary>

```python
from datetime import datetime

def parse_date(date_str: str) -> str:
    """Parse a date string in 'YYYY-MM-DD' format and return the day of the week.

    Args:
        date_str: A date string in the format 'YYYY-MM-DD'.

    Returns:
        The day of the week as a full string, e.g. 'Monday', 'Tuesday', etc.
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%A")
```

</details>

### [FAIL] std_003: Validate Email

- **Category:** stdlib
- **Difficulty:** medium

<details><summary>Generated Code</summary>

```python
import re

def validate_email(email: str) -> bool:
    """Validate an email address using regular expressions.

    A valid email must satisfy:
    - Local part (before @): one or more alphanumeric characters, dots,
      underscores, or hyphens. Must not start or end with a dot/hyphen.
    - Exactly one '@' symbol.
    - Domain part (after @): one or more labels separated by dots.
      Each label must be one or more alphanumeric characters or hyphens,
      but must not start or end with a hyphen.
    - The last domain label (TLD) must be at least 2 characters long
      and consist of only alphabetic characters.

    Args:
        email: The email address string to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False

    # Pattern explanation:
    # ^(?![.\-])                 : Local part must not start with dot or hyphen
    # (?:[a-zA-Z0-9._-]+)        : Local part characters
    # (?<![.\-])                 : Local part must not end with dot or hyphen
    # @                          : Exactly one @
    # (?:                        : Start of domain part
    #   (?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?) : Domain label (no leading/trailing hyphen)
    #   \.                       : Dot separator
    # )*                         : Zero or more intermediate labels
    # (?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)? : Optional second-to-last label (to allow single-label TLD handling below)
    # ([a-zA-Z]{2,})             : TLD: at least 2 alphabetic chars
    # $                          : End of string
    
    # More precise construction:
    # Local part: cannot start/end with . or -, allowed chars: alnum, ., _, -
    local_part = r"(?![.\-])(?:[a-zA-Z0-9._-]+)(?<![.\-])"
    
    # Domain label: alnum or hyphen, but not starting/ending with hyphen
    domain_label = r"[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?"
    # For single-char labels, the above fails, so we adjust:
    domain_label = r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?|[a-zA-Z0-9])"
    # Actually simpler: [a-zA-Z0-9] optionally followed by ([a-zA-Z0-9-]*[a-zA-Z0-9])
    domain_label = r"[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?"
    # But single char is valid: so use (?:[a-zA-Z0-9]|(?:[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]))
    # Let's simplify: each label must be alnum, may contain hyphens inside, not at edges.
    # So: [a-zA-Z0-9] + optional ([a-zA-Z0-9\-]*[a-zA-Z0-9]) -> covers single char too if we make the second part optional correctly.
    # Correct pattern for a label: [a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])? 
    # This works for single char: first char matches, rest is optional and empty.
    
    # However, to avoid matching a trailing hyphen, we ensure the last char is alnum.
    # So domain_label = r"[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?" is okay because:
    #   - If length=1: matches [a-zA-Z0-9], rest skipped.
    #   - If length>1: must end with [a-zA-Z0-9].
    
    # Now, domain: one or more labels separated by dots, last label is TLD (alpha only, >=2)
    # We separate TLD: must be [a-zA-Z]{2,}
    # Other labels: domain_label as above
    
    # Full domain: (domain_label\.)* TLD
    # But note: there must be at least one label before TLD? No: example.com -> two labels.
    # Actually: domain must have at least two parts? Not necessarily per spec, but TLD is last label.
    # Spec says: "one or more labels separated by dots", and last is TLD.
    # So minimum: one label? But then it would be the TLD. However, common practice requires at least two.
    # Re-read spec: "Domain part: one or more labels separated by dots. Each label... The last domain label (TLD)..."
    # This implies at least one label, but then the TLD is that one label. However, TLD must be alpha-only and >=2.
    # But typical emails require at least one subdomain + TLD. Let's follow spec literally: 
    #   It says "labels separated by dots", plural implies multiple? But "one or more" is explicit.
    #   However, if there's only one label, then it's the TLD. Is "user@com" valid? Per spec: 
    #      - label: "com" -> valid (alnum, no hyphen issues), TLD: "com" -> alpha, len>=2 -> valid.
    #   But in reality, we usually require at least two labels. However, the problem statement does not specify that.
    #   Let me check: "one or more labels separated by dots" — if there's only one label, there are no dots, so it's just one label.
    #   So "user@localhost" would be invalid because TLD "localhost" is alpha and >=2, but is "localhost" a valid TLD? 
    #   The spec doesn't restrict TLD to known ones, just format. However, the problem says "The last domain label (TLD)", implying there might be others, but doesn't forbid single-label domains.
    #   BUT: standard email validation usually requires at least two labels. Let me re-read carefully:
    #      "Domain part (after @): one or more labels separated by dots."
    #      This phrasing is ambiguous. In English, "separated by dots" for one label doesn't make sense (no separators). 
    #      Typically, this means: the domain consists of labels, and if there are multiple, they are separated by dots. One label is allowed without dots.
    #   However, the TLD requirement: "last domain label" — if only one, it's the last. 
    #   But practical email standards (RFC) require at least two labels? Actually, RFC allows single-label domains in private contexts, but for public internet, usually two+.
    #   Since the problem doesn't specify, and to be safe, let's assume the domain must have at least two labels? 
    #   Wait, the problem says: "one or more labels", so one is allowed. But then the TLD is that one label. 
    #   However, the example constraints: TLD must be alpha-only and >=2. So "a@b" is invalid (TLD "b" len=1), "a@co" is valid? 
    #   But "co" is a valid TLD (Colombia). So per spec, "user@co" should be valid? 
    #   However, in reality, most validators require at least two labels. Given the ambiguity, I'll follow the literal spec: 
    #      Domain = (label\.)*label, where the last label is TLD (alpha, >=2), and other labels follow the hyphen rules.
    #   But note: if there's only one label, then the entire domain is the TLD. So pattern: 
    #      domain = (?:label\.)*TLD
    #   And label = [a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])? 
    #   TLD = [a-zA-Z]{2,}
    #
    # However, there's a catch: the non-TLD labels can have hyphens (but not at edges), TLD cannot have hyphens (only alpha).
    #
    # Let's construct:
    #   local_part: as above
    #   domain_label_non_tld: [a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?   (allows hyphens inside)
    #   tld: [a-zA-Z]{2,}
    #   domain: (?:domain_label_non_tld\.)*tld
    #
    # But what if there's only the TLD? Then (?:...)* matches zero times, and we have just tld. That's acceptable per "one or more labels" (the TLD is one label).
    #
    # However, the problem says: "each label must be ...", and the TLD is a label, so it must also satisfy the non-hyphen-start/end? 
    #   But the TLD rule overrides: "must consist of only alphabetic characters", so no hyphens allowed anyway. 
    #   And since it's all alpha, it won't start/end with hyphen. So TLD automatically satisfies the general label rule except the hyphen part is irrelevant.
    #
    # Therefore, we can define:
    #   label_with_hyphens = r'[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?'
    #   But note: this pattern for a single char: [a-zA-Z0-9] matches, and the rest is optional -> good.
    #   However, for two chars: [a-zA-Z0-9] then [a-zA-Z0-9] (since [a-zA-Z0-9\-]* can be empty) -> good.
    #   For "a-b": [a] then [\-b] -> but wait: [a-zA-Z0-9\-]* matches '-', then [a-zA-Z0-9] matches 'b'. -> good.
    #   For "a-": fails because after [a], we have '-' in the middle part, but then we need to end with [a-zA-Z0-9] -> so "a-" is rejected. Good.
    #
    # Now, the domain: 
    #   We need zero or more non-TLD labels (which can have hyphens) followed by a dot, then the TLD.
    #   But the TLD is also a label, but with stricter rules (no digits, no hyphens). 
    #   So: domain = (?:label_with_hyphens\.)*[a-zA-Z]{2,}
    #
    # However, what if there are multiple labels and the last one before TLD has a hyphen? That's allowed for non-TLD labels.
    #
    # Example: user@sub-domain.example.com -> 
    #   labels: "sub-domain", "example", "com"
    #   Here, "com" is TLD -> must be alpha, which it is.
    #   "example" and "sub-domain" are non-TLD -> can have hyphens (but not at edges).
    #
    # But in our pattern, the last label is forced to be TLD (alpha only). The preceding labels use label_with_hyphens.
    #
    # So full regex:
    pattern = rf'^{local_part}@(?:(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?\.)*)[a-zA-Z]{{2,}}$'
    
    # However, the above pattern for the domain part: 
    #   (?: ... )* allows zero or more occurrences of (label + dot). Then TLD.
    #   This means the domain can be just TLD (e.g., user@com) which per spec is allowed (one label).
    #
    # But wait: the label pattern [a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])? might not match a single character correctly in all cases? 
    #   Let's test mentally: 
    #      "a" -> [a-zA-Z0-9] matches 'a', the rest is optional -> matches.
    #      "ab" -> [a] then [b] (since [a-zA-Z0-9\-]* matches empty, then [a-zA-Z0-9] matches 'b') -> matches.
    #      "a-b" -> [a] then [\-b]: [a-zA-Z0-9\-]* matches '-', then [a-zA-Z0-9] matches 'b' -> matches.
    #      "a-" -> [a] then we try to match the optional part: [a-zA-Z0-9\-]* matches '-', but then we need [a-zA-Z0-9] at the end -> fails. Good.
    #      "-a" -> first char must be [a-zA-Z0-9] -> fails. Good.
    #
    # However, there's a potential issue: the pattern [a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])? 
    #   For a two-character string like "a-", it would try: 
    #       first char 'a' matches [a-zA-Z0-9]
    #       then the optional group: [a-zA-Z0-9\-]* matches '-', but then [a-zA-Z0-9] fails at the end -> so the whole optional group fails, and we accept just "a". 
    #   But that would match "a" in "a-", leaving "-" unmatched. However, in our domain pattern, each label is followed by a dot or is the last (but the last is handled by TLD). 
    #   In the non-TLD part, we have (label\.), so the label must be followed by a dot. Therefore, if we have "a-.", the label part would try to match "a-" and fail, so it wouldn't match. 
    #   Actually, the regex engine will backtrack. But our pattern for the label is greedy? 
    #
    # To avoid complexity, let's refine the label pattern to explicitly disallow trailing hyphen by structure:
    #   We can use: [a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])? 
    #   This already ensures that if there is more than one character, the last is alnum. And single char is alnum. So no trailing hyphen.
    #
    # But note: what about consecutive hyphens? "a--b" -> 
    #   [a] then [\-\-b]: [a-zA-Z0-9\-]* matches '--', then [a-zA-Z0-9] matches 'b' -> valid. And that's acceptable (hyphens inside are allowed).
    #
    # Now, the TLD: [a-zA-Z]{2,} -> ensures only letters, at least 2.
    #
    # One more edge: the entire email must not have extra characters. The ^ and $ ensure that.
    #
    # Let's write the regex:
    #   local_part: (?![.\-])(?:[a-zA-Z0-9._-]+)(?<![.\-])
    #   But note: the local part must not have consecutive dots? The spec doesn't say that. It only says must not start/end with dot/hyphen.
    #   So "a..b" is allowed per spec? The problem statement doesn't forbid it. So we allow.
    #
    # However, the pattern [a-zA-Z0-9._-]+ allows consecutive dots. That's acceptable per given rules.
    #
    # Final pattern assembly:
    regex_pattern = r'^(?![.\-])(?:[a-zA-Z0-9._-]+)(?<![.\-])@(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
    # Wait, I changed the domain part: now I require at least one non-TLD label? Because (?: ... \.)+ means one or more labels followed by dot, then TLD.
    # But the spec says "one or more labels" for the entire domain. If we do (label\.)*TLD, that allows zero non-TLD labels (so domain = TLD). 
    # But if we do (label\.)+TLD, that requires at least one non-TLD label (so domain has at least two labels: one non-TLD and TLD).
    #
    # Re-read spec: "Domain part (after @): one or more labels separated by dots." 
    #   This is ambiguous. In common interpretation, "labels separated by dots" implies that if there are n labels, there are n-1 dots. 
    #   So one label: no dots -> valid. Two labels: one dot -> valid.
    #   Therefore, we should allow one label (the TLD) as the entire domain.
    #
    # However, in practice, almost all email validators require at least two labels (e.g., example.com). 
    # But the problem statement does not specify a minimum of two labels. It says "one or more".
    #
    # Let me check the TLD rule: "The last domain label (TLD) must be at least 2 characters long and consist of only alphabetic characters."
    #   If there's only one label, it is the last (and only) label, so it must be TLD -> alpha, >=2.
    #   So "user@co" would be valid per spec.
    #
    # However
```

</details>

### [PASS] std_004: Cartesian Product Filter

- **Category:** stdlib
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
    
    return [t for t in product(*lists) if sum(t) > threshold]
```

</details>

### [PASS] std_005: Parallel Map

- **Category:** stdlib
- **Difficulty:** hard

<details><summary>Generated Code</summary>

```python
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar('T')
R = TypeVar('R')

def parallel_map(func: Callable[[T], R], items: list[T], max_workers: int = 4) -> list[R]:
    """Apply a function to each item in a list using parallel threads.

    Uses concurrent.futures.ThreadPoolExecutor to apply func to each
    element of items in parallel, then returns the results in the same
    order as the input items.

    Args:
        func: A callable that takes a single argument and returns a result.
        items: A list of items to process.
        max_workers: Maximum number of threads to use (default 4).

    Returns:
        A list of results in the same order as the input items.
        Returns an empty list if items is empty.
    """
    if not items:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))

    return results
```

</details>

### [PASS] str_001: Is Palindrome

- **Category:** string_processing
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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
    # Extract alphanumeric characters and convert to lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    
    # Check if the cleaned string equals its reverse
    return cleaned == cleaned[::-1]
```

</details>

### [PASS] str_002: Caesar Cipher

- **Category:** string_processing
- **Difficulty:** easy

<details><summary>Generated Code</summary>

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
    if mode == 'decrypt':
        shift = -shift

    shift = shift % 26
    result = []

    for char in text:
        if 'a' <= char <= 'z':
            base = ord('a')
            shifted = (ord(char) - base + shift) % 26 + base
            result.append(chr(shifted))
        elif 'A' <= char <= 'Z':
            base = ord('A')
            shifted = (ord(char) - base + shift) % 26 + base
            result.append(chr(shifted))
        else:
            result.append(char)

    return ''.join(result)
```

</details>

### [PASS] str_003: Extract URLs

- **Category:** string_processing
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
    # Pattern explanation:
    # (https?://\S+) - Capture group for http/https followed by non-whitespace chars
    # We then strip trailing punctuation in post-processing
    pattern = r'https?://\S+'
    
    matches = re.findall(pattern, text)
    
    cleaned_urls = []
    trailing_punct = {'.', ',', ';', '!', '?', ')'}
    
    for url in matches:
        # Strip trailing punctuation that's likely not part of the URL
        while url and url[-1] in trailing_punct:
            url = url[:-1]
        
        # Only add if there's still content after stripping
        if url and len(url) > len('http://'):  # Ensure minimum valid URL length
            cleaned_urls.append(url)
    
    return cleaned_urls
```

</details>

### [PASS] str_004: Template Engine

- **Category:** string_processing
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
    pattern = re.compile(r'\{\{\s*([^}]+?)\s*\}\}')

    def replace_match(match):
        expression = match.group(1).strip()
        
        # Check for default value
        if '|' in expression:
            parts = expression.split('|', 1)
            key_path = parts[0].strip()
            default_value = parts[1].strip()
        else:
            key_path = expression
            default_value = None
        
        # Navigate nested keys
        keys = key_path.split('.')
        value = context
        
        try:
            for key in keys:
                key = key.strip()
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    raise KeyError(key)
            
            return str(value)
        except (KeyError, TypeError):
            if default_value is not None:
                return default_value
            else:
                return match.group(0)  # Return original placeholder
    
    return pattern.sub(replace_match, template)
```

</details>
