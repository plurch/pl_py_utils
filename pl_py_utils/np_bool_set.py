import numpy as np

# TODO: add tests
# https://chatgpt.com/c/68cc5ed2-3fa4-832f-92de-c223aa638881

class NumpyBoolSet:
  """Efficient set that can store non-negative integers. Must know max value at construction time."""
  
  def __init__(self, max_value: int, initial_values: list[int] = []):
    """
    Initialize an empty set with integers in [0, max_value].
    """
    self.max_value = max_value
    self.mask = np.zeros(max_value + 1, dtype=bool)

    if len(initial_values) > 0:
      self.update(initial_values)

  def add(self, value: int):
    """Add an integer to the set."""
    if 0 <= value <= self.max_value:
      self.mask[value] = True
    else:
      raise ValueError(f"value {value} out of range [0, {self.max_value}]")

  def update(self, values: list[int]):
    """Add a list of integers to the set."""
    if len(values) > 0:
      for v in values:
        self.add(v)
    else:
      raise ValueError(f"Provide a non-empty list of integers")
    
  def remove(self, value: int):
    """Remove an integer from the set, raises KeyError if not present."""
    if not self.mask[value]:
      raise KeyError(value)
    self.mask[value] = False

  def discard(self, value: int):
    """Remove if present, ignore otherwise."""
    if 0 <= value <= self.max_value:
      self.mask[value] = False

  def __contains__(self, value: int) -> bool:
    """Membership test: value in set"""
    return 0 <= value <= self.max_value and self.mask[value]

  def __iter__(self):
    """Iterate over elements in the set."""
    return iter(np.flatnonzero(self.mask))

  def __len__(self) -> int:
    """Number of elements in the set."""
    return int(self.mask.sum())

  def to_list(self):
    """Return elements as a Python list."""
    return np.flatnonzero(self.mask).tolist()

  # --- set operations ---
  def union(self, other: "NumpyBoolSet") -> "NumpyBoolSet":
    if self.max_value != other.max_value:
      raise ValueError("Sets must have the same max_value")
    new_set = NumpyBoolSet(self.max_value)
    new_set.mask = np.bitwise_or(self.mask, other.mask)
    return new_set

  def intersection(self, other: "NumpyBoolSet") -> "NumpyBoolSet":
    if self.max_value != other.max_value:
      raise ValueError("Sets must have the same max_value")
    new_set = NumpyBoolSet(self.max_value)
    new_set.mask = np.bitwise_and(self.mask, other.mask)
    return new_set

  def difference(self, other: "NumpyBoolSet") -> "NumpyBoolSet":
    if self.max_value != other.max_value:
      raise ValueError("Sets must have the same max_value")
    new_set = NumpyBoolSet(self.max_value)
    new_set.mask = np.bitwise_and(self.mask, ~other.mask)
    return new_set
