import pytest
from pl_py_utils.data_processing import getIdxMappings

class TestGetIdxMappings:
  """Test cases for getIdxMappings function."""

  def test_string_ids(self):
    """Test with string identifiers."""
    ids = ["apple", "banana", "cherry"]
    id_to_idx, idx_to_id = getIdxMappings(ids)

    assert id_to_idx == {"apple": 0, "banana": 1, "cherry": 2}
    assert idx_to_id == {0: "apple", 1: "banana", 2: "cherry"}

  def test_integer_ids(self):
    """Test with integer identifiers."""
    ids = [100, 200, 300]
    id_to_idx, idx_to_id = getIdxMappings(ids)

    assert id_to_idx == {100: 0, 200: 1, 300: 2}
    assert idx_to_id == {0: 100, 1: 200, 2: 300}

  def test_mixed_types(self):
    """Test with mixed integer and string identifiers."""
    ids = [1, "two", 3, "four"]
    id_to_idx, idx_to_id = getIdxMappings(ids)

    assert id_to_idx == {1: 0, "two": 1, 3: 2, "four": 3}
    assert idx_to_id == {0: 1, 1: "two", 2: 3, 3: "four"}

  def test_empty_list(self):
    """Test with empty list."""
    ids = []
    id_to_idx, idx_to_id = getIdxMappings(ids)

    assert id_to_idx == {}
    assert idx_to_id == {}

  def test_single_element(self):
    """Test with single element list."""
    ids = ["only"]
    id_to_idx, idx_to_id = getIdxMappings(ids)

    assert id_to_idx == {"only": 0}
    assert idx_to_id == {0: "only"}

  def test_duplicate_ids(self):
    """Test with duplicate IDs - later occurrences should overwrite earlier ones."""
    ids = ["a", "b", "a", "c", "b"]
    id_to_idx, idx_to_id = getIdxMappings(ids)

    # 'a' should map to 2 (last occurrence), 'b' to 4
    assert id_to_idx == {"a": 2, "b": 4, "c": 3}
    assert idx_to_id == {0: "a", 1: "b", 2: "a", 3: "c", 4: "b"}

  def test_none_values(self):
    """Test with None values in the list."""
    ids = ["a", None, "b", None]
    id_to_idx, idx_to_id = getIdxMappings(ids)

    assert id_to_idx == {"a": 0, None: 3, "b": 2}
    assert idx_to_id == {0: "a", 1: None, 2: "b", 3: None}

  def test_custom_objects(self):
    """Test with custom objects as identifiers."""

    class CustomId:
      def __init__(self, value):
        self.value = value

      def __eq__(self, other):
        return isinstance(other, CustomId) and self.value == other.value

      def __hash__(self):
        return hash(self.value)

    obj1, obj2, obj3 = CustomId(1), CustomId(2), CustomId(3)
    ids = [obj1, obj2, obj3]
    id_to_idx, idx_to_id = getIdxMappings(ids)

    assert id_to_idx == {obj1: 0, obj2: 1, obj3: 2}
    assert idx_to_id == {0: obj1, 1: obj2, 2: obj3}

  def test_large_list(self):
    """Test with a large list to ensure performance."""
    ids = list(range(10000))
    id_to_idx, idx_to_id = getIdxMappings(ids)

    assert len(id_to_idx) == 10000
    assert len(idx_to_id) == 10000
    assert id_to_idx[5000] == 5000
    assert idx_to_id[5000] == 5000

  @pytest.mark.parametrize(
    "ids_list,expected_id_to_idx",
    [
      (["x", "y", "z"], {"x": 0, "y": 1, "z": 2}),
      ([1.5, 2.5, 3.5], {1.5: 0, 2.5: 1, 3.5: 2}),
      ([True, False], {True: 0, False: 1}),
    ],
  )
  def test_various_types_parametrized(self, ids_list, expected_id_to_idx):
    """Parametrized test for various data types."""
    id_to_idx, idx_to_id = getIdxMappings(ids_list)

    assert id_to_idx == expected_id_to_idx
    # Verify bidirectional mapping
    for id_val, idx in id_to_idx.items():
      assert idx_to_id[idx] == id_val

  def test_return_types(self):
    """Test that return types are correct."""
    ids = ["a", "b", "c"]
    result = getIdxMappings(ids)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert isinstance(result[1], dict)
