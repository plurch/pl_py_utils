import os
import math
import concurrent.futures
import pandas as pd
import subprocess
import psutil
from typing import Any, Callable, Sequence, NamedTuple
from io import StringIO
from timeit import default_timer as timer
from datetime import datetime

from .getsize import total_size

def getCurrentTimeStamp() -> str:
  # Ex. '2023-07-24T12_16_04'
  return datetime.now().isoformat(timespec='seconds').replace(':', '_')

def getSizeObject(obj: object) -> str:
  totalSizeBytes = total_size(obj)
  return getSizePretty(totalSizeBytes)

def getSizePretty(totalSizeBytes: int) -> str:
  kbSize = 1024
  mbSize = math.pow(kbSize, 2)
  gbSize = math.pow(kbSize, 3)

  if totalSizeBytes < kbSize:
    unit = 'b'
    resultSize = totalSizeBytes
  elif totalSizeBytes < mbSize:
    unit = 'kb'
    resultSize = totalSizeBytes / kbSize
  elif totalSizeBytes < gbSize:
    unit = 'mb'
    resultSize = totalSizeBytes / mbSize
  else:
    unit = 'gb'
    resultSize = totalSizeBytes / gbSize

  return f'{round(resultSize, 3)} {unit}'

def get_process_memory_usage():
  process = psutil.Process(os.getpid())
  return getSizePretty(process.memory_info().rss)

def timerPrint(msg: str):
  print(f'(timer: {round(timer())}) - {msg}', flush=True) # https://stackoverflow.com/a/36081434
  print(get_process_memory_usage())

def max_sublist(l: Sequence[Any], max_len: int) -> list[Sequence[Any]]:
  '''split a list into a list of lists with each sublist having a maximum length. preserves order'''
  return [l[i * max_len: (i+1) * max_len] for i in range(math.ceil(len(l) / max_len))]

def chunker_list_striped(seq: Any, num_chunks: int) -> list[Any]:
  '''split list into number of chunks. sublists are striped - they do not preserver overall list order'''
  # https://stackoverflow.com/a/43922107/
  return [seq[i::num_chunks] for i in range(num_chunks)]

def recursive_dict_merge(d1: dict[Any, Any], d2: dict[Any, Any]) -> dict[Any, Any]:
    '''update first dict with second recursively'''
    # https://stackoverflow.com/a/24088493/ (see rec_merge2)
    for k, v in d1.items():
        if k in d2:
            d2[k] = recursive_dict_merge(v, d2[k])
    d1.update(d2)
    return d1

class CPU_cores(NamedTuple):
  physical: int
  logical: int

def num_cpu_cores() -> CPU_cores:
  return CPU_cores(psutil.cpu_count(logical=False), psutil.cpu_count(logical=True))

def print_gpu_usage():
  command = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"
  command_output = subprocess.check_output(command.split()).decode('ascii')
  df = pd.read_csv(StringIO(command_output))
  print(df)

def print_system_info():
  cpu_cores = num_cpu_cores()
  print(f'CPU cores: {cpu_cores.physical} physical, {cpu_cores.logical} logical')
  mem_usage = psutil.virtual_memory()
  print(f'Memory total: {getSizePretty(mem_usage.total)}')
  print(f'Memory available: {getSizePretty(mem_usage.available)}')
  print(f'Current python process memory usage: {get_process_memory_usage()}')
  disk_usage = psutil.disk_usage("/")
  print(f'Disk (/) total: {getSizePretty(disk_usage.total)}')
  print(f'Disk (/) used: {getSizePretty(disk_usage.used)}')
  print(f'Disk (/) free: {getSizePretty(disk_usage.free)}')
  try:
    command_output = subprocess.check_output('nvidia-smi').decode('ascii')
  except:
    command_output = 'No GPU found!'
  print('nvidia-smi output:')
  print(command_output)

def map_parallel(
  fn: Callable,
  all_batches: Sequence[Any],
  use_threads=False,
  fn_args={}) -> list[Any]:
  """Applies a function to a list, equivalent to [fn(x, **args) for x in xs]."""
  if len(all_batches) == 1:
    return [fn(all_batches[0], **fn_args)]

  cpu_cores = num_cpu_cores()

  if use_threads:
    my_executor = concurrent.futures.ThreadPoolExecutor
    num_workers = cpu_cores.logical
  else:
    # use `ProcessPoolExecutor` if need python processes with GIL
    my_executor = concurrent.futures.ProcessPoolExecutor
    num_workers = cpu_cores.physical
  
  with my_executor(num_workers) as executor:
    # could refactor this to take a list of objects to be used with kwargs. Then can vary all func params
    futures = [executor.submit(fn, batch, **fn_args) for batch in all_batches]
    concurrent.futures.wait(futures)
    results = [future.result() for future in futures]

  return results

# Alternative to `map_parallel` above for testing
# return [fn(batch, **args) for batch in all_batches]
def map_serial(
  fn: Callable,
  all_batches: Sequence[Any],
  fn_args={}) -> list[Any]:
  # timerPrint(f'map_serial - starting')

  res = []
  for i, batch in enumerate(all_batches):
    res.append(fn(batch, **fn_args))
    # timerPrint(f'map_serial - completed iteration {i}')

  # timerPrint(f'map_serial - done')
  return res