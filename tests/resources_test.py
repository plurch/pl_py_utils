
import sys
import os

parent_dir_path = os.path.abspath(os.path.join('..'))
if parent_dir_path not in sys.path:
    sys.path.append(parent_dir_path)

from pl_py_utils.resources import getSizePretty

def test_getSizePretty():
  result = getSizePretty(totalSizeBytes=3747474)
  assert result == '3.574 mb'
