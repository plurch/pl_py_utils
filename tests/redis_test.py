import pytest
import numpy as np
from pl_py_utils.redis import gen_redis_proto_bin

class TestRedis:
  def test_gen_redis_proto_bin(self):
    ba = gen_redis_proto_bin("HSET", str(1), "embedding", np.array([1.5, 2.4, 3.6], dtype=np.float32).tobytes())
    assert ba == bytearray(b'*4\r\n$4\r\nHSET\r\n$1\r\n1\r\n$9\r\nembedding\r\n$12\r\n\x00\x00\xc0?\x9a\x99\x19@fff@\r\n')

