import math
import concurrent.futures
from typing import Any, Callable, Sequence, NamedTuple
from timeit import default_timer as timer
from datetime import datetime

from .resources import get_process_memory_usage, num_cpu_cores

def getCurrentTimeStamp() -> str:
  # Ex. '2023-07-24T12_16_04'
  return datetime.now().isoformat(timespec='seconds').replace(':', '_')

def timerPrint(msg: str):
  """
  Print a message with an accompanying timestamp from a timer and the current process's memory usage.

  The function first retrieves the current timer value, formats it alongside the provided message,
  and then prints the message. Following this, it prints the memory usage of the current process.

  The flush=True argument in the print function ensures that the print output is immediately written 
  to the stream without buffering. This is useful in scenarios where immediate feedback is crucial, 
  such as long-running processes or real-time monitoring.

  Args:
      msg (str): The message to be printed alongside the timer's timestamp.
  """
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