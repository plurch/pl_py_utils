from typing import TypeAlias

RedisProtoArg: TypeAlias = str | bytes | bytearray

def gen_redis_proto_bin(*args: RedisProtoArg) -> bytearray:
  '''
  Generate Redis protocol into a bytearray. Can then write to a binary file.

  Any number of arguments (can be str, bytes, bytearray)
  
  Example writing to a file:

  with open("redis_commands.txt", "wb") as file:
    for ri in range(my_np_array.shape[0]):
      file.write(gen_redis_proto_bin("HSET", str(ri), "embedding", my_np_array[ri].tobytes()))

  Based on ruby example `gen_redis_proto` here:
  https://redis.io/docs/manual/patterns/bulk-loading/
  https://redis.io/docs/reference/protocol-spec/
  '''
  frame = bytearray()
  frame.extend(f'*{str(len(args))}\r\n'.encode('ascii'))
  for arg in args:
    arg_bytes = arg if isinstance(arg, (bytes, bytearray)) else arg.encode('ascii')
    frame.extend(f'${len(arg_bytes)}\r\n'.encode('ascii'))
    frame.extend(arg_bytes)
    frame.extend('\r\n'.encode('ascii'))
  return frame
