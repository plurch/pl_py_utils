import math
import numpy.typing as npt
from typing import Any, Callable, Sequence, NamedTuple, TypeVar, Generic, Union, overload
from timeit import default_timer as timer
from datetime import datetime

from .resources import get_process_memory_usage, num_cpu_cores

T = TypeVar('T')
SequenceOrArray = Sequence[T] | npt.NDArray[Any]

def getCurrentTimeStamp() -> str:
  """
  Return timestamp as string. Ex. '2023-07-24T12_16_04'
  """
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
  print(get_process_memory_usage(), flush=True)

def max_sublist(l: Sequence[T], max_len: int) -> list[Sequence[T]]:
  """
  Split a list into a list of lists with each sublist having a maximum length. preserves order
  """
  return [l[i * max_len: (i+1) * max_len] for i in range(math.ceil(len(l) / max_len))]

def chunker_list_striped(seq: npt.NDArray, num_chunks: int) -> list[npt.NDArray]:
  """
  Split list into number of chunks. sublists are striped - they do not preserve overall list order
  """
  # https://stackoverflow.com/a/43922107/
  return [seq[i::num_chunks] for i in range(num_chunks)]

def recursive_dict_merge(d1: dict[Any, Any], d2: dict[Any, Any]) -> dict[Any, Any]:
  """
  Update first dict with second recursively
  """
  # https://stackoverflow.com/a/24088493/ (see rec_merge2)
  for k, v in d1.items():
    if k in d2:
      d2[k] = recursive_dict_merge(v, d2[k])
  d1.update(d2)
  return d1

def dict_filter_out(d: dict, filter_out: Sequence[Any]) -> dict:
  """
  Filters out specified keys from a dictionary.
  
  Parameters:
  - d (dict): The input dictionary to be filtered.
  - filter_out (Sequence[Any]): A sequence of keys that should be removed from the input dictionary.
  
  Returns:
  - dict: A new dictionary with the specified keys filtered out.
  
  Examples:
  >>> dict_filter_out({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
  {'b': 2}
  """
  return {k: v for k, v in d.items() if k not in filter_out}

def int_commas(n: int) -> str:
  return "{:,}".format(n)
