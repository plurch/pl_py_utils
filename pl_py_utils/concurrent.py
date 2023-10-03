import concurrent.futures
from typing import Any, Callable, Sequence, NamedTuple

from .resources import num_cpu_cores

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

def map_serial(
  fn: Callable,
  all_batches: Sequence[Any],
  fn_args={}) -> list[Any]:
  """
  Alternative to `map_parallel` above for testing. 
  return [fn(batch, **args) for batch in all_batches]
  """
  # timerPrint(f'map_serial - starting')

  res = []
  for i, batch in enumerate(all_batches):
    res.append(fn(batch, **fn_args))
    # timerPrint(f'map_serial - completed iteration {i}')

  # timerPrint(f'map_serial - done')
  return res