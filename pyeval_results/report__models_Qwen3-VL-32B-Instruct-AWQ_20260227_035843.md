# PyEval Report: /models/Qwen3-VL-32B-Instruct-AWQ

**Date:** 2026-02-27 03:58:43 UTC  
**Overall Pass Rate:** 88.9%  
**Weighted Score:** 87.2% (easy=1x, medium=2x, hard=3x)  
**Total:** 40/45

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| Basic Syntax | 3/3 | 1/1 | 1/1 | 5/5 |
| Data Structures | 2/2 | 2/2 | 1/1 | 5/5 |
| Algorithms | 2/2 | 3/3 | 2/2 | 7/7 |
| Standard Library | 2/2 | 2/2 | 1/1 | 5/5 |
| OOP | 1/1 | 1/2 | 1/2 | 3/5 |
| Exceptions | 1/1 | 1/1 | 0/1 | 2/3 |
| File I/O | 1/1 | 1/1 | 1/1 | 3/3 |
| String Processing | 2/2 | 0/1 | 1/1 | 3/4 |
| Functional | 1/1 | 2/2 | 1/1 | 4/4 |
| Concurrency | 0/1 | 2/2 | 1/1 | 3/4 |
| **TOTAL** | **15/16** | **15/17** | **10/12** | **40/45** |

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
    # Create a copy of the input list to avoid modifying the original
    sorted_arr = arr.copy()
    n = len(sorted_arr)
    
    # Bubble sort algorithm
    for i in range(n):
        # Flag to optimize - if no swaps occur, array is already sorted
        swapped = False
        for j in range(0, n - i - 1):
            if sorted_arr[j] > sorted_arr[j + 1]:
                # Swap elements
                sorted_arr[j], sorted_arr[j + 1] = sorted_arr[j + 1], sorted_arr[j]
                swapped = True
        
        # If no swapping occurred, the array is sorted
        if not swapped:
            break
    
    return sorted_arr
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
        return arr.copy()
    
    # Split the array into two halves
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # Recursively sort both halves
    left_sorted = merge_sort(left)
    right_sorted = merge_sort(right)
    
    # Merge the sorted halves
    return merge(left_sorted, right_sorted)

def merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists into a single sorted list."""
    result = []
    i = j = 0
    
    # Compare elements and merge in sorted order
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements from left list
    while i < len(left):
        result.append(left[i])
        i += 1
    
    # Add remaining elements from right list
    while j < len(right):
        result.append(right[j])
        j += 1
    
    return result
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
    # Create a 2D DP table where dp[i][j] represents the length of LCS of s1[0:i] and s2[0:j]
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
    bracket_map = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in '([{':
            stack.append(char)
        else:  # char is one of ')]}'
            if not stack or stack.pop() != bracket_map[char]:
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
    if n == 0 or capacity == 0:
        return 0
    
    # Create a 2D DP table: dp[i][w] represents max value using first i items with weight limit w
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't include current item
            dp[i][w] = dp[i - 1][w]
            
            # Include current item if possible
            if weights[i - 1] <= w:
                include_value = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                dp[i][w] = max(dp[i][w], include_value)
    
    return dp[n][capacity]
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
    # Calculate in-degrees for all nodes
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Find all nodes with in-degree 0 (no incoming edges)
    queue = [node for node in in_degree if in_degree[node] == 0]
    result = []

    # Process nodes in topological order
    while queue:
        current = queue.pop(0)
        result.append(current)

        # Remove current node and update in-degrees of its neighbors
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If we couldn't process all nodes, there's a cycle
    if len(result) != len(graph):
        return []

    return result
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
    # Create a new dictionary with swapped values
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
    if n == 0:
        return "Zero"
    
    # Define number words
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    
    def convert_to_words(num):
        if num == 0:
            return ""
        elif num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            ten_part = num // 10
            one_part = num % 10
            if one_part == 0:
                return tens[ten_part]
            else:
                return f"{tens[ten_part]}-{ones[one_part]}"
        else:  # num >= 100
            hundred_part = num // 100
            remainder = num % 100
            if remainder == 0:
                return f"{ones[hundred_part]} Hundred"
            else:
                return f"{ones[hundred_part]} Hundred {convert_to_words(remainder)}"
    
    return convert_to_words(n)
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
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_nested(item))
        else:
            result.append(item)
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
    
    rows, cols = len(matrix), len(matrix[0])
    result = []
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1
    
    while top <= bottom and left <= right:
        # Traverse right along the top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Traverse down along the right column
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Traverse left along the bottom row (if there's a bottom row left)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Traverse up along the left column (if there's a left column left)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result
```

</details>

### [FAIL] con_001: Thread-Safe Counter

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
        """Initialize the counter with an optional starting value (default 0)."""
        self.value = initial
        self.lock = threading.Lock()

    def increment(self):
        """Atomically increase the counter by 1."""
        with self.lock:
            self.value += 1

    def decrement(self):
        """Atomically decrease the counter by 1."""
        with self.lock:
            self.value -= 1

    def get_value(self):
        """Return the current counter value."""
        with self.lock:
            return self.value
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
        # Submit all tasks and store futures in order
        futures = [executor.submit(func, item) for item in items]
        # Collect results in the same order as input
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
    # Create a bounded queue
    q = queue.Queue(maxsize=queue_size)
    
    # List to collect results from consumers
    results = []
    results_lock = threading.Lock()
    
    # Start consumer threads first
    consumer_threads = []
    for consumer in consumers:
        def consumer_worker(consumer_func, results_list, lock):
            while True:
                item = q.get()
                if item is None:  # sentinel
                    break
                # Process the item and collect result
                result = consumer_func(item)
                with lock:
                    results_list.append(result)
                q.task_done()
        
        thread = threading.Thread(target=consumer_worker, args=(consumer, results, results_lock))
        thread.start()
        consumer_threads.append(thread)
    
    # Start producer threads
    producer_threads = []
    for producer in producers:
        def producer_worker(producer_func, q):
            items = producer_func()
            for item in items:
                q.put(item)
        
        thread = threading.Thread(target=producer_worker, args=(producer, q))
        thread.start()
        producer_threads.append(thread)
    
    # Wait for all producer threads to finish
    for thread in producer_threads:
        thread.join()
    
    # Put sentinel None for each consumer
    for _ in consumers:
        q.put(None)
    
    # Wait for all consumer threads to finish
    for thread in consumer_threads:
        thread.join()
    
    # Return all results
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
    # Create a list of coroutines by calling each async callable
    coroutines = [callable() for callable in async_callables]
    
    # Use asyncio.gather to run all coroutines concurrently
    if timeout is not None:
        # Wrap with asyncio.wait_for to enforce timeout
        try:
            results = await asyncio.wait_for(asyncio.gather(*coroutines), timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Operation timed out after {timeout} seconds")
        return results
    else:
        # No timeout, just gather and return results
        results = await asyncio.gather(*coroutines)
        return results
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
    # Use a dictionary to store the value and its index
    value_to_index = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in value_to_index:
            # Return the indices in ascending order
            return [value_to_index[complement], i]
        value_to_index[num] = i
    
    # This should never be reached given the problem assumptions
    return []
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
    if not values:
        return []
    
    # Initialize pointers: prev, current, next
    prev = None
    current = values[0]
    next_node = None
    
    # Convert the list to a linked list structure for simulation
    # We'll use indices to simulate node traversal
    for i in range(len(values)):
        next_node = values[i] if i < len(values) else None
        # Simulate reversing: current.next = prev
        # In list terms, we're building the result in reverse order
        # We'll collect the values in reverse order as we traverse
        pass
    
    # Alternative approach: build result list by iterating backwards
    # But we need to simulate pointer reversal, so let's use iterative approach
    # with three pointers: prev, current, next
    
    # Initialize
    prev = None
    current = values[0] if values else None
    result = []
    
    # We'll simulate the reversal by iterating through the list
    # and building the result in reverse order
    for i in range(len(values)):
        # Store next node
        next_node = values[i + 1] if i + 1 < len(values) else None
        # Reverse the link: current points to prev
        # In our simulation, we'll add current to result (which will be in reverse order)
        result.append(current)
        # Move to next
        prev = current
        current = next_node
    
    # But this doesn't reverse the order. Let's do it properly.
    # We need to build the result in reverse order by traversing from end to start.
    # However, the requirement is to simulate pointer reversal.
    
    # Let's implement the classic iterative reversal algorithm:
    prev = None
    current = values[0] if values else None
    next_node = None
    
    # We'll create a new list to store the reversed values
    reversed_values = []
    
    # Simulate the linked list reversal
    for i in range(len(values)):
        # Store next node
        next_node = values[i + 1] if i + 1 < len(values) else None
        # Reverse the link: current points to prev
        # In our case, we're not actually changing pointers, but we can simulate
        # by building the result list in reverse order
        # We'll add current to the beginning of our result list
        reversed_values.insert(0, values[i])
        # Move to next
        prev = values[i]
        current = next_node
    
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
        """Initialize the cache with positive capacity."""
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        """Return the value if key exists, otherwise -1."""
        if key not in self.cache:
            return -1
        # Move the accessed key to the end to mark it as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        """Insert or update the key-value pair."""
        if key in self.cache:
            # Update the value and move to end to mark as recently used
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # If cache is full, remove the least recently used item (first item)
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            # Add the new key-value pair
            self.cache[key] = value
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
    if not strs:
        return []
    
    # Dictionary to store groups of anagrams
    anagram_groups = {}
    
    for s in strs:
        # Sort the characters in the string to get a key for anagrams
        sorted_s = ''.join(sorted(s))
        
        # Add the string to the corresponding group
        if sorted_s in anagram_groups:
            anagram_groups[sorted_s].append(s)
        else:
            anagram_groups[sorted_s] = [s]
    
    # Sort each group alphabetically
    for key in anagram_groups:
        anagram_groups[key].sort()
    
    # Convert to list of lists and sort groups by their first element
    result = [anagram_groups[key] for key in anagram_groups]
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
    # Handle edge case: empty lists
    if not lists:
        return []
    
    # Initialize min-heap with first element from each non-empty list
    # Each heap element is (value, list_index, element_index)
    heap = []
    result = []
    
    for i, lst in enumerate(lists):
        if lst:  # Only add non-empty lists
            heapq.heappush(heap, (lst[0], i, 0))
    
    # Process elements from heap until empty
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # If there are more elements in the same list, add next element to heap
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
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
                except Exception as e:
                    if isinstance(e, exceptions):
                        last_exception = e
                        if attempt < max_attempts - 1:
                            time.sleep(delay)
                        else:
                            raise last_exception
                    else:
                        # Propagate any other exception immediately
                        raise
            # This should never be reached due to the raise in the loop
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

    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(self.message)


class TypeValidationError(ValidationError):
    """Raised when a field has the wrong type.

    Attributes:
        field (str): The field name.
        expected_type (type): The expected type.
        actual_type (type): The actual type received.
    """

    def __init__(self, field, expected_type, actual_type):
        message = f"Expected type {expected_type.__name__}, got {actual_type.__name__}"
        super().__init__(field, message)
        self.expected_type = expected_type
        self.actual_type = actual_type


class RangeValidationError(ValidationError):
    """Raised when a numeric field is out of range.

    Attributes:
        field (str): The field name.
        value: The actual value.
        min_value: The minimum allowed value (or None).
        max_value: The maximum allowed value (or None).
    """

    def __init__(self, field, value, min_value, max_value):
        message_parts = []
        if min_value is not None and value < min_value:
            message_parts.append(f"less than minimum {min_value}")
        if max_value is not None and value > max_value:
            message_parts.append(f"greater than maximum {max_value}")
        message = f"Value {value} is {', '.join(message_parts)}" if message_parts else f"Value {value} is out of range"
        super().__init__(field, message)
        self.value = value
        self.min_value = min_value
        self.max_value = max_value


class PatternValidationError(ValidationError):
    """Raised when a string field does not match a regex pattern.

    Attributes:
        field (str): The field name.
        pattern (str): The regex pattern that was expected.
        value (str): The actual string value.
    """

    def __init__(self, field, pattern, value):
        message = f"Value '{value}' does not match pattern '{pattern}'"
        super().__init__(field, message)
        self.pattern = pattern
        self.value = value


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
            raise ValidationError(field, f"Required field '{field}' is missing")

        value = data[field]
        expected_type = rules.get('type')

        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeValidationError(field, expected_type, type(value))

        # Check range for numeric types
        if expected_type in (int, float) or isinstance(value, (int, float)):
            min_val = rules.get('min')
            max_val = rules.get('max')
            if min_val is not None and value < min_val:
                raise RangeValidationError(field, value, min_val, max_val)
            if max_val is not None and value > max_val:
                raise RangeValidationError(field, value, min_val, max_val)

        # Check pattern for string types
        if expected_type == str or isinstance(value, str):
            pattern = rules.get('pattern')
            if pattern is not None:
                if not re.fullmatch(pattern, str(value)):
                    raise PatternValidationError(field, pattern, str(value))

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
       认 -> [{'a': '1', 'b': '2'}]
    """
    if not csv_string:
        return []
    
    # Use StringIO to wrap the string for csv.reader
    file_like = io.StringIO(csv_string)
    reader = csv.DictReader(file_like)
    
    # Read all rows into a list of dictionaries
    data = list(reader)
    
    return data
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
    if not json_strings:
        return {}

    # Parse the first JSON string
    result = json.loads(json_strings[0])

    # Merge remaining JSON strings
    for i in range(1, len(json_strings)):
        current = json.loads(json_strings[i])
        merge_dicts(result, current)

    return result


def merge_dicts(dest: dict, src: dict):
    """
    Recursively merge two dictionaries.
    
    If both dest[key] and src[key] are dictionaries, merge them recursively.
    Otherwise, overwrite dest[key] with src[key].
    """
    for key, value in src.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
            merge_dicts(dest[key], value)
        else:
            dest[key] = value
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
    if not log_lines:
        return {
            'counts': {},
            'errors': [],
            'timestamp_range': None
        }

    # Regular expression to match log line format
    pattern = r'\[([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\] ([A-Z]+): (.+)'
    
    counts = {}
    errors = []
    timestamps = []
    
    for line in log_lines:
        match = re.match(pattern, line)
        if match:
            timestamp, level, message = match.groups()
            timestamps.append(timestamp)
            
            # Update counts
            counts[level] = counts.get(level, 0) + 1
            
            # Collect errors (ERROR and CRITICAL)
            if level in ['ERROR', 'CRITICAL']:
                errors.append(message)
    
    # Calculate timestamp range
    if timestamps:
        earliest = min(timestamps)
        latest = max(timestamps)
        timestamp_range = (earliest, latest)
    else:
        timestamp_range = None
    
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
        """Initialize with the given iterable and an empty list of operations."""
        self.iterable = iterable
        self.operations = []

    def map(self, func):
        """Register a map operation. Returns self for chaining."""
        self.operations.append(('map', func))
        return self

    def filter(self, predicate):
        """Register a filter operation. Returns self for chaining."""
        self.operations.append(('filter', predicate))
        return self

    def evaluate(self):
        """Execute all registered operations in order and return result as a list."""
        result = list(self.iterable)
        for op_type, func in self.operations:
            if op_type == 'map':
                result = [func(x) for x in result]
            elif op_type == 'filter':
                result = [x for x in result if func(x)]
        self.operations = []  # Clear operations for reusability
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
        """Initialize an empty stack using a list."""
        self.items = []

    def push(self, item):
        """Add an item to the top of the stack.

        Args:
            item: The item to be added to the stack.
        """
        self.items.append(item)

    def pop(self):
        """Remove and return the top item from the stack.

        Returns:
            The top item of the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self):
        """Return the top item without removing it.

        Returns:
            The top item of the stack.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]

    def is_empty(self):
        """Check if the stack is empty.

        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        return len(self.items) == 0

    def size(self):
        """Return the number of items in the stack.

        Returns:
            int: The number of items in the stack.
        """
        return len(self.items)
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
    total = 0.0
    for shape in shapes:
        if not isinstance(shape, Shape):
            raise TypeError("All items in shapes list must be instances of Shape.")
        total += shape.area()
    return total
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

        withdraw(amount) ->世 None:
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

    def __init__(self, owner, initial_balance=0.0):
        self.owner = owner
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative.")
        self._balance = initial_balance
        self._history = []

    @property
    def balance(self):
        return self._balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive.")
        self._balance += amount
        self._history.append(('deposit', amount))

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive.")
        if amount > self._balance:
            raise ValueError("Insufficient funds.")
        self._balance -= amount
        self._history.append(('withdraw', amount))

    def transfer_to(self, other_account, amount):
        if amount <= 0:
            raise ValueError("Transfer amount must be positive.")
        if amount > self._balance:
            raise ValueError("Insufficient funds.")
        self.withdraw(amount)
        other_account.deposit(amount)
        self._history.append(('transfer_out', amount))
        # Note: The 'transfer_in' is recorded in other_account.deposit()

    def get_history(self):
        return self._history.copy()
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
        """Register a callback for the given event."""
        if event not in self._events:
            self._events[event] = []
        self._events[event].append(callback)

    def off(self, event: str, callback: callable) -> None:
        """Remove one registration of callback for the given event."""
        if event not in self._events:
            return
        if callback in self._events[event]:
            self._events[event].remove(callback)

    def emit(self, event: str, *args, **kwargs) -> None:
        """Call all registered callbacks for the event with the given arguments."""
        if event not in self._events:
            return
        for callback in self._events[event][:]:  # Iterate over a copy to handle removals
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
        self._value = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self._value is None:
            raise AttributeError(f"'{self.name}' has not been set")
        return self._value

    def __set__(self, instance, value):
        if not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be of type {self.field_type.__name__}, got {type(value).__name__}")
        
        if self.field_type in (int, float):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"{self.name} must be at least {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"{self.name} must be at most {self.max_value}")
        
        if self.field_type is str and self.max_length is not None and len(value) > self.max_length:
            raise ValueError(f"{self.name} must be at most {self.max_length} characters long")
        
        self._value = value


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
    words = text.split()
    return Counter(words)
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
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.strftime('%A')
```

</details>

### [PASS] std_003: Validate Email

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
    # Regular expression pattern for email validation
    pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}$'
    
    # Check if email matches the pattern
    if not re.match(pattern, email):
        return False
    
    # Additional check: ensure domain part doesn't start or end with hyphen
    domain_parts = email.split('@')[1].split('.')
    for part in domain_parts:
        if len(part) == 0:
            return False
        if part.startswith('-') or part.endswith('-'):
            return False
    
    # Check that TLD is only alphabetic
    tld = domain_parts[-1]
    if not tld.isalpha():
        return False
    
    return True
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
    
    result = []
    for combo in product(*lists):
        if sum(combo) > threshold:
            result.append(combo)
    
    return result
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
        # Submit all tasks and store futures in order
        futures = [executor.submit(func, item) for item in items]
        
        # Collect results in the same order as input items
        results = [future.result() for future in futures]
    
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
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    
    # Check if the cleaned string reads the same forwards and backwards
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
    # Normalize shift to be within 0-25
    shift = shift % 26
    
    # If decrypting, reverse the shift
    if mode == 'decrypt':
        shift = -shift
    
    result = []
    
    for char in text:
        if char.isalpha():
            # Determine if uppercase or lowercase
            is_upper = char.isupper()
            # Convert to 0-25 range (A=0, B=1, ..., Z=25)
            char_code = ord(char.lower()) - ord('a')
            # Apply shift
            shifted_code = (char_code + shift) % 26
            # Convert back to character
            shifted_char = chr(shifted_code + ord('a'))
            # Restore case
            if is_upper:
                shifted_char = shifted_char.upper()
            result.append(shifted_char)
        else:
            # Non-alphabetic characters remain unchanged
            result.append(char)
    
    return ''.join(result)
```

</details>

### [FAIL] str_003: Extract URLs

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
    # Pattern: starts with http:// or https://, followed by non-whitespace chars,
    # and ends before any trailing punctuation (., !, ?, ;, ), or whitespace)
    pattern = r'https?://[^\\s\.,!?\);]+'
    urls = re.findall(pattern, text)
    return urls
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
    def resolve_key(key_str: str) -> str:
        """Resolve a key string to its value in context, with optional default."""
        # Split by pipe to separate key from default
        parts = key_str.split('|', 1)
        key = parts[0].strip()
        default_value = parts[1].strip() if len(parts) > 1 else None
        
        # Split key by dots for nested access
        keys = key.split('.')
        
        # Traverse the context dictionary
        current = context
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                # Key not found
                if default_value is not None:
                    return default_value
                else:
                    return f"{{{{{key}}}}}"  # Return original placeholder if no default
        
        # Convert to string
        return str(current)
    
    # Use regex to find all placeholders
    # Pattern: {{ ... }} where ... is any characters except closing braces
    pattern = r'{{\s*([^}]+)\s*}}'
    
    # Replace each placeholder with resolved value
    result = re.sub(pattern, lambda match: resolve_key(match.group(1)), template)
    
    return result
```

</details>
