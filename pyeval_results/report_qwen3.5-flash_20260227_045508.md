# PyEval Report: qwen3.5-flash

**Date:** 2026-02-27 04:55:08 UTC  
**Overall Pass Rate:** 100.0%  
**Weighted Score:** 100.0% (easy=1x, medium=2x, hard=3x)  
**Total:** 3/3

## Summary

| Category | Easy | Medium | Hard | Total |
|----------|------|--------|------|-------|
| OOP | - | 1/1 | 1/1 | 2/2 |
| Exceptions | - | - | 1/1 | 1/1 |
| **TOTAL** | **-** | **1/1** | **2/2** | **3/3** |

## Detailed Results

### [PASS] exc_003: Custom Exception Hierarchy and Schema Validation

- **Category:** exceptions
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
        super().__init__(f"Field '{field}': {message}")


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
        message = f"Expected type {expected_type.__name__}, got {actual_type.__name__}"
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
            message = f"Value {value} is not in range [{min_value}, {max_value}]"
        elif min_value is not None:
            message = f"Value {value} is less than minimum {min_value}"
        else:
            message = f"Value {value} is greater than maximum {max_value}"
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
        
        # Check if field exists
        if field not in data:
            if required:
                raise ValidationError(field, "Field is required but missing")
            continue
        
        value = data[field]
        
        # Type validation
        if 'type' in rules:
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                actual_type = type(value)
                raise TypeValidationError(field, expected_type, actual_type)
        
        # Range validation (only for numeric types)
        if 'min' in rules or 'max' in rules:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                min_val = rules.get('min')
                max_val = rules.get('max')
                
                if min_val is not None and value < min_val:
                    raise RangeValidationError(field, value, min_value=min_val, max_value=max_val)
                if max_val is not None and value > max_val:
                    raise RangeValidationError(field, value, min_value=min_val, max_value=max_val)
        
        # Pattern validation (only for strings)
        if 'pattern' in rules:
            if isinstance(value, str):
                pattern = rules['pattern']
                if not re.fullmatch(pattern, value):
                    raise PatternValidationError(field, pattern, value)
    
    return True
```

</details>

### [PASS] oop_003: BankAccount with Transaction History

- **Category:** oop
- **Difficulty:** medium

<details><summary>Generated Code</summary>

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
            raise ValueError("Initial balance cannot be negative.")
        self.owner = owner
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
        self._history.append(('transfer_out', amount))
        
        other_account._balance += amount
        other_account._history.append(('transfer_in', amount))

    def get_history(self) -> list:
        return list(self._history)
```

</details>

### [PASS] oop_005: ValidatedField Descriptor

- **Category:** oop
- **Difficulty:** hard

<details><summary>Generated Code</summary>

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
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        attr_name = f'_validated_{self.name}'
        try:
            return getattr(obj, attr_name)
        except AttributeError:
            raise AttributeError(f"'{objtype.__name__}' object has no attribute '{self.name}'")

    def __set__(self, obj, value):
        # Validation 1: Type check
        if not isinstance(value, self.field_type):
            raise TypeError(f"Expected {self.field_type.__name__}, got {type(value).__name__}")

        # Validation 2 & 3: Numeric range checks
        if self.field_type in (int, float):
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"Value {value} is less than minimum allowed value {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"Value {value} is greater than maximum allowed value {self.max_value}")

        # Validation 4: String length check
        if self.field_type is str and self.max_length is not None:
            if len(value) > self.max_length:
                raise ValueError(f"String length {len(value)} exceeds maximum allowed length {self.max_length}")

        # Store the validated value
        attr_name = f'_validated_{self.name}'
        setattr(obj, attr_name, value)


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
