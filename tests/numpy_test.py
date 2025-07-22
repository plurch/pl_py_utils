import pytest
import numpy as np

from pl_py_utils.numpy import normalize_vec, topk_indices

class TestNormalizeVec:
  def test_normalize_simple_1D_vector(self):
    vec = np.array([3.0, 4.0])
    normalized_vec = normalize_vec(vec)
    np.testing.assert_allclose(normalized_vec, np.array([0.6, 0.8]), atol=1e-8)

  def test_normalize_zero_vector(self):
    vec = np.array([0.0, 0.0])
    with pytest.raises(ZeroDivisionError):
      normalize_vec(vec)

  def test_normalize_unit_length_vector(self):
    vec = np.array([1.0])
    normalized_vec = normalize_vec(vec)
    np.testing.assert_allclose(normalized_vec, np.array([1.0]), atol=1e-8)

class TestTopKIndicesDesc:
  def test_simple_1D_array(self):
    a = np.array([1.2, 3.4, 0.8, 2.3])
    indices = topk_indices(a, 2, 'desc')
    np.testing.assert_array_equal(indices, np.array([1, 3]))

  def test_k_equals_array_length(self):
    a = np.array([5.0, 4.5, 4.6])
    indices = topk_indices(a, 3, 'desc')
    np.testing.assert_array_equal(indices, np.array([0, 2, 1]))

  def test_single_element_array(self):
    a = np.array([1.0])
    indices = topk_indices(a, 1, 'desc')
    np.testing.assert_array_equal(indices, np.array([0]))

  def test_empty_array(self):
    a = np.array([])
    indices = topk_indices(a, 0, 'desc')
    np.testing.assert_array_equal(indices, np.array([]))

class TestTopKIndicesAsc:
  def test_simple_1D_array(self):
    a = np.array([1.2, 3.4, 0.8, 2.3])
    indices = topk_indices(a, 2, 'asc')
    np.testing.assert_array_equal(indices, np.array([2, 0]))

  def test_k_equals_array_length(self):
    a = np.array([5.0, 4.5, 4.6])
    indices = topk_indices(a, 3, 'asc')
    np.testing.assert_array_equal(indices, np.array([1, 2, 0]))

  def test_single_element_array(self):
    a = np.array([1.0])
    indices = topk_indices(a, 1, 'asc')
    np.testing.assert_array_equal(indices, np.array([0]))

  def test_empty_array(self):
    a = np.array([])
    indices = topk_indices(a, 0, 'asc')
    np.testing.assert_array_equal(indices, np.array([]))

class TestTopKIndicesErrors:
  def test_negative_k(self):
    a = np.array([1.2, 3.4, 0.8, 2.3])

    with pytest.raises(ValueError) as exc_info:
      indices = topk_indices(a, -2, 'asc')

    assert str(exc_info.value) == "k must be non-negative, got -2"

  def test_too_large_k(self):
    a = np.array([5.0, 4.5, 4.6])

    with pytest.raises(ValueError) as exc_info:
      indices = topk_indices(a, 4, 'asc')

    assert str(exc_info.value) == "k (4) cannot be larger than array size (3)"
