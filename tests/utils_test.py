import numpy as np
from pl_py_utils.utils import max_sublist, chunker_list_striped, recursive_dict_merge

class TestMaxSubList:
  def test_list(self):
    res = max_sublist(list(range(27)), 6)
    expected = [
      [0, 1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10, 11],
      [12, 13, 14, 15, 16, 17],
      [18, 19, 20, 21, 22, 23],
      [24, 25, 26]
    ]
    assert res == expected

  def test_tuple(self):
    res = max_sublist(tuple(range(27)), 6)
    expected = [
      (0, 1, 2, 3, 4, 5),
      (6, 7, 8, 9, 10, 11),
      (12, 13, 14, 15, 16, 17),
      (18, 19, 20, 21, 22, 23),
      (24, 25, 26)
    ]
    assert res == expected

class TestChunkerListStriped:
  def test_array(self):
    res = chunker_list_striped(np.arange(27), 4)
    expected = [
      np.array([ 0,  4,  8, 12, 16, 20, 24]),
      np.array([ 1,  5,  9, 13, 17, 21, 25]),
      np.array([ 2,  6, 10, 14, 18, 22, 26]),
      np.array([ 3,  7, 11, 15, 19, 23])
    ]
    np.testing.assert_equal(res, expected)

class TestRecursiveDictMerge:
  def test_array(self):
    a = {"keyA": 1, "keyB": {"sub1": 10}}
    b = {"keyB": {"sub2": 20}}
    res = recursive_dict_merge(a, b)
    expected = {'keyA': 1, 'keyB': {'sub1': 10, 'sub2': 20}}
    assert res == expected
