import numpy as np
import numpy.typing as npt
from typing import Literal

def topk_indices_desc_new(a: npt.NDArray[np.floating], k: int) -> npt.NDArray[np.integer]:
  '''
  Similar to `topk_indices` below, but works for each row of input `a`
  TODO: add docs, unit tests. Integrate in app code. Replace below or keep?
  '''
  row_indices_range = np.arange(a.shape[0])[:, np.newaxis] # to select all rows
  i_all = np.argpartition(a, -k) # partition indices
  i = i_all[:, -k:] # get top k indices columns
  topk_values = a[row_indices_range, i] # get top k values for each row in initial array
  j = np.fliplr(np.argsort(topk_values)) # get indices sorted descending
  return i[row_indices_range, j] # map back to original indices

# can use numba @njit(nogil=True) in application code
# https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
def topk_indices(a: npt.NDArray[np.floating], k: int, ordering: Literal['desc', 'asc'] = 'desc') -> npt.NDArray[np.integer]:
  """
  Find the indices of the top `k` largest elements in the array `a`, sorted in specified order.

  Equivalent to `np.argsort(r)[::-1][:k]`, but doesn't have to sort entire array, only k values

  Parameters:
  -----------
  a : npt.NDArray[np.floating]
      The input array containing numerical values.
  k : int
      The number of top elements to select.
  ordering : Literal['desc', 'asc']
      The ordering either descending or ascending

  Returns:
  --------
  npt.NDArray[np.integer]
      An array of indices corresponding to the top `k` largest elements in `a`,
      sorted in descending order.

  Examples:
  ---------
  >>> topk_indices(np.array([1.2, 3.4, 0.8, 2.3]), 2)
  array([1, 3])

  >>> topk_indices(np.array([5.0, 4.5, 4.6]), 3)
  array([0, 2, 1])
  """
  if k < 0:
    raise ValueError(f"k must be non-negative, got {k}")

  n = len(a)
  if k > n:
      raise ValueError(f"k ({k}) cannot be larger than array size ({n})")

  if ordering == 'desc':
    i = np.argpartition(a, -k)[-k:] # find top k largest values O(n)
    j = np.argsort(a[i])[::-1] # sort only top k values O(k log k)
  elif ordering == 'asc':
    i = np.argpartition(a, k-1)[:k] # find top k smallest values O(n)
    j = np.argsort(a[i]) # sort only top k values O(k log k)
  else:
    raise ValueError("`ordering` must be 'asc' or 'desc'")

  return i[j] # return original indices sorted

def normalize_vec(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
  """
  Normalize a vector to have a unit length (L2 norm).

  This function calculates the L2 norm of the input vector `v` using NumPy's `linalg.norm` function,
  and then divides each element of the vector by the norm.

  Parameters:
  -----------
  v : npt.NDArray[np.floating]
      Input vector to be normalized. The vector can be 1D or higher.

  Returns:
  --------
  npt.NDArray[np.floating]
      A vector of the same shape as `v`, normalized to have a unit length.

  Raises:
  -------
  ZeroDivisionError
      If the input vector has an L2 norm of zero.

  Examples:
  ---------
  >>> normalize_vec(np.array([3.0, 4.0]))
  array([0.6, 0.8])

  >>> normalize_vec(np.array([0.0, 0.0]))
  ZeroDivisionError: The norm of the input vector is zero.

  """
  norm = np.linalg.norm(v)
  if norm == 0:
    raise ZeroDivisionError
  return v / norm

def json_serialize_np_float(obj):
  '''
  convert from numpy floating point to python float
  TODO: unit tests, better docs
  '''
  if isinstance(obj, np.floating):
    return float(obj)
  raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

def print_array_info(arr):
  '''
  Print memory usage info for numpy array
  '''
  nbytes = arr.nbytes  # total bytes
  size = arr.size      # total elements
  itemsize = arr.itemsize  # bytes per element
  
  # Convert to more readable units
  units = ['B', 'KB', 'MB', 'GB']
  unit_index = 0
  memory_size = nbytes
  while memory_size >= 1024 and unit_index < len(units) - 1:
      memory_size /= 1024
      unit_index += 1
      
  print(f"Shape: {arr.shape}")
  print(f"Total elements: {size:,}")
  print(f"Element size: {itemsize} bytes")
  print(f"Total memory: {memory_size:.2f} {units[unit_index]}")
  print(f"Data type: {arr.dtype}")

