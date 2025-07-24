
from pl_py_utils.resources import getSizePretty

class TestGetSizePretty:
  def test_bytes(self):
    result = getSizePretty(totalSizeBytes=506)
    assert result == '506 b'

  def test_kilobytes(self):
      result = getSizePretty(totalSizeBytes=1702)
      assert result == '1.7 kb'

  def test_megabytes(self):
      result = getSizePretty(totalSizeBytes=3747474)
      assert result == '3.6 mb'

  def test_gigabytes(self):
      result = getSizePretty(totalSizeBytes=4783474736)
      assert result == '4.5 gb'
