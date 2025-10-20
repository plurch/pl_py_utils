import numpy as np
import numpy.typing as npt
from typing import Literal, Union, Final
from .resources import getSizePretty

# Gracefully try to import cupy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

def row_wise_top_k(
  a: npt.NDArray[np.floating],
  top_k: int,
  sort_order: Literal["asc", "desc"] = "asc",
  sort_within_rows: bool = True
) -> npt.NDArray[np.intp]:
  """
  Performs efficient row-wise top-k sorting for a 2D array using argpartition.

  Pass in a cupy array to use cupy/gpu.

  This function finds the indices of the top-k values for each row in the input
  array. It is highly efficient for large arrays when k is much smaller than
  the number of columns, as it avoids a full sort.

  Args:
    a: The 2D input array with shape (n_rows, n_cols). Either numpy or cupy array.
    top_k: The number of top elements to return for each row.
    sort_order: The order of sorting.
      "asc" for ascending (smallest values are "top").
      "desc" for descending (largest values are "top").
      Defaults to "asc".
    sort_within_rows: If True, the returned k indices for each row will be
      sorted according to their corresponding values. If False, the k indices
      will be returned in an arbitrary order. Defaults to True.

  Returns:
    A 2D integer array of shape (n_rows, top_k) containing the column
    indices of the top-k values for each row.

  Raises:
    ValueError: If top_k is larger than the number of columns in 'a' or if
                sort_order is not 'asc' or 'desc'.

  https://aistudio.google.com/app/prompts/1r1VEUqrb4Qm38OLvA8Cos90Il9aK2nAW
  """
  # Determine the array module (numpy or cupy) to use
  if CUPY_AVAILABLE:
      # print("USING CUPY - remove me!")
      xp = cp.get_array_module(a)
  else:
      # If cupy is not installed, we can only be working with numpy
      xp = np

  # --- Input Validation ---
  if a.ndim != 2:
    raise ValueError("Input array 'a' must be 2-dimensional.")
  n_rows, n_cols = a.shape
  if top_k > n_cols:
    raise ValueError(
      f"top_k ({top_k}) cannot be larger than the number of columns ({n_cols})."
    )
  if sort_order not in ["asc", "desc"]:
    raise ValueError("sort_order must be either 'asc' or 'desc'.")

  # --- Core Logic using argpartition ---
  # np.argpartition is efficient for finding the k-th smallest element's index
  # without performing a full sort. Elements before the k-th index are all
  # smaller, and elements after are all larger. Their relative order is not
  # guaranteed.

  # To find the top-k largest values (descending), we can find the top-k
  # smallest values of the negated array.
  partition_target = -a if sort_order == "desc" else a

  # Get the indices of the top_k elements for each row.
  # The first `top_k` columns of the result will contain the indices of the
  # `top_k` smallest elements, but in an unsorted order.
  # We use `top_k - 1` because argpartition is 0-indexed.
  # memory usage for argpartition:
  # "You can expect memory usage to be roughly 2-3x the size of the input array (not the output), though this can vary"
  # https://claude.ai/chat/c41e555a-af41-45c3-8f0e-9f40525b5160
  top_k_indices = xp.argpartition(partition_target, top_k - 1, axis=1)[:, :top_k]

  if not sort_within_rows:
    return top_k_indices

  # --- Optional Sorting within the Top-K Results ---
  # If sorting is required, we now perform a full sort but only on the
  # top_k elements we've already identified.

  # 1. Get the actual values corresponding to the top_k_indices.
  #    np.take_along_axis is an efficient way to do this.
  top_k_values = xp.take_along_axis(a, top_k_indices, axis=1)

  # 2. Get the sorting order for these top_k values.
  #    This gives us indices relative to the (0, k-1) range.
  sort_indices_within_k = xp.argsort(top_k_values, axis=1)

  # 3. For descending order, we reverse the sorted indices.
  if sort_order == "desc":
    sort_indices_within_k = xp.fliplr(sort_indices_within_k)

  # 4. Use the `sort_indices_within_k` to reorder the `top_k_indices`.
  #    This applies the small sort to our final index array.
  final_sorted_indices = xp.take_along_axis(
      top_k_indices, sort_indices_within_k, axis=1
  )

  return final_sorted_indices

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

def get_np_array_size_pretty(a: npt.NDArray) -> str:
  return getSizePretty(a.nbytes)

def get_csr_matrix_bytes(my_csr_matrix) -> int:
  '''
  get number of bytes used by a scipy CSR matrix
  '''
  return my_csr_matrix.data.nbytes + my_csr_matrix.indices.nbytes + my_csr_matrix.indptr.nbytes

def get_csr_matrix_size_pretty(my_csr_matrix) -> str:
  return getSizePretty(get_csr_matrix_bytes(my_csr_matrix))

