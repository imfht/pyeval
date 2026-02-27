# PyEval Report: MiniMax-M2.5 [Multi-turn]

**Date:** 2026-02-27 14:01:46 UTC  
**Mode:** Multi-turn  
**Overall Pass Rate:** 100.0%  
**Weighted Score:** 94.9%  
**Total:** 45/45

**Strict Pass Rate (1st try):** 88.9%  
**With Retries Pass Rate:** 100.0%  

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| Basic Syntax | 3/3 | 1/1 | 1/1 | 5/5 |
| Data Structures | 2/2 | 2/2 | 1/1 | 5/5 |
| Algorithms | 2/2 | 3/3 | 2/2 | 7/7 |
| Standard Library | 2/2 | 2/2 | 1/1 | 5/5 |
| OOP | 1/1 | 2/2 | 2/2 | 5/5 |
| Exceptions | 1/1 | 1/1 | 1/1 | 3/3 |
| File I/O | 1/1 | 1/1 | 1/1 | 3/3 |
| String Processing | 2/2 | 1/1 | 1/1 | 4/4 |
| Functional | 1/1 | 2/2 | 1/1 | 4/4 |
| Concurrency | 1/1 | 2/2 | 1/1 | 4/4 |
| **TOTAL** | **16/16** | **17/17** | **12/12** | **45/45** |

## Attempt Distribution

- Attempt 1 (weight=1.0): 40 passed
- Attempt 2 (weight=0.6): 5 passed
- Attempt 3 (weight=0.3): 0 passed

## Detailed Results

### [PASS (attempt 1)] algo_001: Binary Search

- **Category:** algorithms
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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

### [PASS (attempt 2)] algo_002: Bubble Sort

- **Category:** algorithms
- **Difficulty:** easy
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
def bubble_sort(arr: list[int]) -> list[int]:
    """Sort a list of integers in ascending order using bubble sort.

    Args:
        arr: A list of integers to sort.

    Returns:
        A new list containing the same elements sorted in ascending order.
        The original list must not be modified.
    """
    # Create a copy to avoid modifying the original
    result = arr.copy()
    n = len(result)
    
    # Outer loop: we need n-1 passes at maximum
    for i in range(n - 1):
        # Track if any swaps were made in this pass
        swapped = False
        # Inner loop: compare adjacent elements, last i elements are already sorted
        for j in range(n - 1 - i):
            if result[j] > result[j + 1]:
                # Swap
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        
        # If no swaps were made, the array is already sorted
        if not swapped:
            break
    
    return result
```

</details>

### [PASS (attempt 1)] algo_003: Merge Sort

- **Category:** algorithms
- **Difficulty:** medium
- **Attempts:** 1

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
        return arr[:]
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return _merge(left, right)


def _merge(left: list[int], right: list[int]) -> list[int]:
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

### [PASS (attempt 1)] algo_004: Longest Common Subsequence

- **Category:** algorithms
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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
        The length of the longest common subsequence. """
    m, n = len(s1), len(s2)
    
    # Create a 2D DP table where dp[i][j] represents the LCS length
    # of s1[0:i] and s2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                # Characters match, extend the LCS from previous prefixes
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Characters don't match, take the maximum from excluding
                # either character from the respective strings
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]
```

</details>

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
    opening_brackets = set(bracket_map.values())
    
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
    
    # dp[w] = maximum value achievable with capacity w
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # Traverse backwards to avoid using the same item twice
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
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
               {'a': ['b', 'c']} means edges a->b and a->c.               All nodes in the graph appear as keys (even if they
               have no outgoing edges).

    Returns:
        A list of node names in a valid topological order. If the graph
        contains a cycle, return an empty list.
    """
    # Get all unique nodes from both keys and values
    nodes = set(graph.keys())
    for neighbors in graph.values():
        nodes.update(neighbors)
    
    # Calculate in-degree for each node
    in_degree = {node: 0 for node in nodes}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    # Initialize queue with nodes that have no incoming edges
    from collections import deque
    queue = deque([node for node in nodes if in_degree[node] == 0])
    
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # If result contains all nodes, return it; otherwise there's a cycle
    if len(result) == len(nodes):
        return result
    else:
        return []
```

</details>

### [PASS (attempt 1)] bs_001: FizzBuzz

- **Category:** basic_syntax
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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
        if i % 3 == 0 and i % 5 == 0:
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
    if n == 0:
        return "Zero"
    
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
            "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
            "Seventeen", "Eighteen", "Nineteen"]
    
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    
    if n < 20:
        return ones[n]
    
    if n < 100:
        ten_part = tens[n // 10]
        one_part = ones[n % 10]
        if one_part:
            return f"{ten_part}-{one_part}"
        return ten_part
    
    # n is 100-999
    hundred_part = ones[n // 100] + " Hundred"
    remainder = n % 100
    
    if remainder == 0:
        return hundred_part
    
    if remainder < 20:
        return hundred_part + " " + ones[remainder]
    
    ten_part = tens[remainder // 10]
    one_part = ones[remainder % 10]
    if one_part:
        return hundred_part + " " + ten_part + "-" + one_part
    return hundred_part + " " + ten_part
```

</details>

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
        flatten_nested([1, [2,3], [4, [5, 6]]]) -> [1, 2, 3, 4, 5, 6]
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

### [PASS (attempt 1)] bs_005: Matrix Spiral Order

- **Category:** basic_syntax
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
def matrix_spiral(matrix: list) -> list:
    if not matrix or not matrix[0]:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse from left to right along top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Traverse from top to bottom along right column
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Traverse from right to left along bottom row (if rows remain)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Traverse from bottom to top along left column (if columns remain)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result
```

</details>

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
        counter.get_value()  # ->1

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
        parallel_map(lambda x: x +1, [], max_workers=2)  # -> []
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))
```

</details>

### [PASS (attempt 2)] con_003: Producer Consumer

- **Category:** concurrency
- **Difficulty:** medium
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
import threading
import queue

def producer_consumer(producers: list, consumers: list, queue_size: int = 10) -> list:
    # Create a bounded queue
    q = queue.Queue(maxsize=queue_size)
    
    # Results collected from consumers (thread-safe)
    results = []
    results_lock = threading.Lock()
    
    # Consumer threads list
    consumer_threads = []
    
    # Define consumer thread function
    def consumer_thread_func(consumer):
        local_results = []
        while True:
            item = q.get()
            if item is None:
                # Sentinel received, stop
                break
            # Process the item
            result = consumer(item)
            local_results.append(result)
            q.task_done()
        
        # Add local results to global results (thread-safe)
        with results_lock:
            results.extend(local_results)
    
    # Start consumer threads first (they will wait on the queue)
    for consumer in consumers:
        t = threading.Thread(target=consumer_thread_func, args=(consumer,))
        t.start()
        consumer_threads.append(t)
    
    # Producer threads list
    producer_threads = []
    
    # Define producer thread function
    def producer_thread_func(producer):
        items = producer()  # Get items from producer callable
        for item in items:
            q.put(item)  # Put each item on queue
    
    # Start producer threads
    for producer in producers:
        t = threading.Thread(target=producer_thread_func, args=(producer,))
        t.start()
        producer_threads.append(t)
    
    # Wait for all producer threads to finish
    for t in producer_threads:
        t.join()
    
    # Put one sentinel None per consumer on the queue
    for _ in range(len(consumers)):
        q.put(None)
    
    # Wait for all consumer threads to finish
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
    # Create coroutines by calling each callable
    coroutines = [callable() for callable in async_callables]
    
    # Run coroutines concurrently with optional timeout
    if timeout is not None:
        results = await asyncio.wait_for(asyncio.gather(*coroutines), timeout=timeout)
    else:
        results = await asyncio.gather(*coroutines)
    
    return list(results)
```

</details>

### [PASS (attempt 1)] ds_001: Two Sum

- **Category:** data_structures
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
```

</details>

### [PASS (attempt 1)] ds_002: Reverse Linked List

- **Category:** data_structures
- **Difficulty:** easy
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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
    if not values:
        return []
    
    # Simulate linked list reversal by iterating from tail to head
    # This mimics the pointer manipulation in linked list reversal
    reversed_list = []
    index = len(values) - 1
    
    while index >= 0:
        reversed_list.append(values[index])
        index -= 1
    
    return reversed_list
```

</details>

### [PASS (attempt 1)] ds_003: LRU Cache

- **Category:** data_structures
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.cache:
            # Move to the end to mark as recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing key and move to end
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # If over capacity, remove oldest (first) item
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

</details>

### [PASS (attempt 1)] ds_004: Group Anagrams

- **Category:** data_structures
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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
    
    # Group strings by their sorted character key (anagrams will have the same key)
    anagram_groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        anagram_groups[key].append(s)
    
    # Sort each group and then sort the groups by their first element
    result = [sorted(group) for group in anagram_groups.values()]
    result.sort(key=lambda x: x[0])
    
    return result
```

</details>

### [PASS (attempt 1)] ds_005: Merge K Sorted Lists

- **Category:** data_structures
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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
    result = []
    
    # Initialize heap with first element from each non-empty list
    # Heap entries: (value, list_index, element_index)
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    # Extract min and add next element from the same list
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # If there are more elements in this list, push the next one
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result
```

</details>

### [PASS (attempt 1)] exc_001: Safe Divide

- **Category:** exceptions
- **Difficulty:** easy
- **Attempts:** 1

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
        raise TypeError("Both arguments must be numbers (int or float)")
    
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
            Any other exception types should propagate immediately.        delay (float): Seconds to wait between retries. Defaults to 0.

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

### [PASS (attempt 1)] exc_003: Custom Exception Hierarchy and Schema Validation

- **Category:** exceptions
- **Difficulty:** hard
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

```python
import re


class ValidationError(Exception):
    """Base exception for validation errors.

    Constructor:
        __init__(self, field: str, message: str)
            Store field and message as instance attributes.
            Call super().__init__() with a formatted string.

    Instance Attributes:
        field (str): The name of the field that failed validation.
        message (str): A human-readable error message.
    """

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class TypeValidationError(ValidationError):
    """Raised when a field has the wrong type.

    Constructor:
        __init__(self, field: str, expected_type: type, actual_type: type)
            Store expected_type and actual_type as instance attributes.
            Call super().__init__(field, <message>).

    Instance Attributes (in addition to inherited field, message):
        expected_type (type): The expected type.
        actual_type (type): The actual type received.
    """

    def __init__(self, field: str, expected_type: type, actual_type: type):
        self.expected_type = expected_type
        self.actual_type = actual_type
        message = f"Expected {expected_type.__name__}, got {actual_type.__name__}"
        super().__init__(field, message)


class RangeValidationError(ValidationError):
    """Raised when a numeric field is out of range.

    Constructor:
        __init__(self, field: str, value, min_value=None, max_value=None)
            Store value, min_value, max_value as instance attributes.
            Call super().__init__(field, <message>).

    Instance Attributes (in addition to inherited field, message):
        value: The actual value.
        min_value: The minimum allowed value (or None).
        max_value: The maximum allowed value (or None).
    """

    def __init__(self, field: str, value, min_value=None, max_value=None):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value

        if min_value is not None and max_value is not None:
            message = f"Value {value} is out of range [{min_value}, {max_value}]"
        elif min_value is not None:
            message = f"Value {value} is below minimum {min_value}"
        else:
            message = f"Value {value} is above maximum {max_value}"

        super().__init__(field, message)


class PatternValidationError(ValidationError):
    """Raised when a string field does not match a regex pattern.

    Constructor:
        __init__(self, field: str, pattern: str, value: str)
            Store pattern and value as instance attributes.
            Call super().__init__(field, <message>).

    Instance Attributes (in addition to inherited field, message):
        pattern (str): The regex pattern that was expected.
        value (str): The actual string value.
    """

    def __init__(self, field: str, pattern: str, value: str):
        self.pattern = pattern
        self.value = value
        message = f"Value '{value}' does not match pattern '{pattern}'"
        super().__init__(field, message)


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

        if required and field not in data:
            raise ValidationError(field, "Required field is missing")

        if field not in data:
            continue

        value = data[field]

        if 'type' in rules:
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                raise TypeValidationError(field, expected_type, type(value))

        if 'min' in rules or 'max' in rules:
            min_value = rules.get('min')
            max_value = rules.get('max')

            if isinstance(value, (int, float)):
                if min_value is not None and value < min_value:
                    raise RangeValidationError(field, value, min_value=min_value, max_value=max_value)
                if max_value is not None and value > max_value:
                    raise RangeValidationError(field, value, min_value=min_value, max_value=max_value)

        if 'pattern' in rules:
            pattern = rules['pattern']
            if isinstance(value, str) and not re.fullmatch(pattern, value):
                raise PatternValidationError(field, pattern, value)

    return True
```

</details>

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
    # Create a StringIO object from the csv_string
    csv_file = io.StringIO(csv_string)
    
    # Use csv.DictReader to parse the CSV - it automatically uses the first row as headers
    reader = csv.DictReader(csv_file)
    
    # Convert the reader to a list of dictionaries
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
    Merge a list of JSON strings into a single dictionary.    Each JSON string represents a dictionary. Merge them left to right so
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
    def deep_merge(existing: dict, new: dict) -> dict:
        result = existing.copy()
        for key, value in new.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Both values are dicts, so merge recursively
                result[key] = deep_merge(result[key], value)
            else:
                # Non-dict values always override
                result[key] = value
        return result
    
    result = {}
    for json_str in json_strings:
        parsed = json.loads(json_str)
        result = deep_merge(result, parsed)
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
    pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (DEBUG|INFO|WARNING|ERROR|CRITICAL): (.*)'
    
    counts = {}
    errors = []
    timestamps = []
    
    for line in log_lines:
        match = re.match(pattern, line)
        if match:
            timestamp, level, message = match.groups()
            timestamps.append(timestamp)
            
            # Count each level
            counts[level] = counts.get(level, 0) + 1
            
            # Collect error messages
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
        for op_type, op_func in self._operations:
            if op_type == 'map':
                result = list(map(op_func, result))
            elif op_type == 'filter':
                result = list(filter(op_func, result))
        self._operations = []
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

### [PASS (attempt 1)] oop_002: Shape Hierarchy with Abstract Base Class

- **Category:** oop
- **Difficulty:** medium
- **Attempts:** 1

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
        if width <= 0:
            raise ValueError("Width must be positive")
        if height <= 0:
            raise ValueError("Height must be positive")
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

### [PASS (attempt 2)] oop_004: EventEmitter - Observer Pattern

- **Category:** oop
- **Difficulty:** hard
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

```python
class EventEmitter:
    def __init__(self):
        self._events = {}  # event -> list of callbacks
    
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
        """Call all registered callbacks for the event."""
        if event in self._events:
            # Make a copy to avoid calling listeners added during emit
            callbacks = self._events[event][:]
            for callback in callbacks:
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

    def __set_name__(self, owner, name):
        self.name = name
        self.storage_name = '_validated_' + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if not hasattr(obj, self.storage_name):
            raise AttributeError(f"'{type(obj).__name__}' object has no attribute '{self.name}'")
        return getattr(obj, self.storage_name)

    def __set__(self, obj, value):
        # Check type
        if not isinstance(value, self.field_type):
            raise TypeError(f"'{self.name}' must be of type {self.field_type.__name__}, got {type(value).__name__}")

        # Check min_value for int/float
        if self.field_type in (int, float) and self.min_value is not None:
            if value < self.min_value:
                raise ValueError(f"'{self.name}' must be >= {self.min_value}")

        # Check max_value for int/float
        if self.field_type in (int, float) and self.max_value is not None:
            if value > self.max_value:
                raise ValueError(f"'{self.name}' must be <= {self.max_value}")

        # Check max_length for str
        if self.field_type is str and self.max_length is not None:
            if len(value) > self.max_length:
                raise ValueError(f"'{self.name}' must be <= {self.max_length} characters")

        # Store the value
        setattr(obj, self.storage_name, value)


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

### [PASS (attempt 1)] std_003: Validate Email

- **Category:** stdlib
- **Difficulty:** medium
- **Attempts:** 1

<details><summary>Final Generated Code</summary>

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
    # Local part: alphanumeric, starts and ends with alphanumeric, 
    # middle chars can include dots, underscores, hyphens
    local_pattern = r'[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?'
    
    # Domain label: alphanumeric, can have hyphens in middle but not at ends
    domain_label_pattern = r'[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?'
    
    # TLD: at least 2 alphabetic characters
    tld_pattern = r'[a-zA-Z]{2,}'
    
    # Full pattern: local@domain.tld
    full_pattern = (
        rf'^{local_pattern}@'
        rf'(?:{domain_label_pattern}\.)+'
        rf'{tld_pattern}$'
    )
    
    return bool(re.fullmatch(full_pattern, email))
```

</details>

### [PASS (attempt 1)] std_004: Cartesian Product Filter

- **Category:** stdlib
- **Difficulty:** medium
- **Attempts:** 1

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
    
    return [combo for combo in product(*lists) if sum(combo) > threshold]
```

</details>

### [PASS (attempt 2)] std_005: Parallel Map

- **Category:** stdlib
- **Difficulty:** hard
- **Attempts:** 2

<details><summary>Final Generated Code</summary>

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
    result = []
    
    # Invert shift for decryption
    if mode == 'decrypt':
        shift = -shift
    
    # Normalize shift modulo 26
    shift = shift % 26
    
    for char in text:
        if char.isalpha():
            # Determine the ASCII offset based on case
            ascii_offset = ord('A') if char.isupper() else ord('a')
            
            # Shift the character and wrap around using modulo
            shifted = (ord(char) - ascii_offset + shift) % 26 + ascii_offset
            result.append(chr(shifted))
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
    the surrounding text rather than the URL itself. Args:
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
    url_pattern = r'https?://\S+'
    
    urls = re.findall(url_pattern, text)
    
    # Punctuation marks to strip from the end of URLs
    trailing_punctuation = '.,;:!?)'
    
    cleaned_urls = []
    for url in urls:
        # Remove trailing punctuation characters
        while url and url[-1] in trailing_punctuation:
            url = url[:-1]
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
        content = match.group(1).strip()
        
        # Check for default value
        if '|' in content:
            key, default = content.split('|', 1)
            key = key.strip()
            default = default.strip()
        else:
            key = content
            default = None
        
        # Handle empty key
        if not key:
            return match.group(0)
        
        # Navigate nested keys
        keys = key.split('.')
        value = context
        
        try:
            for k in keys:
                value = value[k]
            return str(value)
        except (KeyError, TypeError, IndexError):
            if default is not None:
                return default
            else:
                return match.group(0)  # Return original placeholder
    
    # Use regex to find all {{...}} patterns
    return re.sub(r'\{\{(.*?)\}\}', replace_placeholder, template)
```

</details>
