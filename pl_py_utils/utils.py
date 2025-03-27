import math
import os
import platform
import zoneinfo
import signal
import importlib.util
import numpy.typing as npt
from typing import Any, Sequence, TypeVar
from types import ModuleType
from timeit import default_timer as timer
from datetime import datetime
from importlib.metadata import version

from .resources import get_process_memory_usage, num_cpu_cores

T = TypeVar('T')
SequenceOrArray = Sequence[T] | npt.NDArray[Any]

class Timeout:
  # https://stackoverflow.com/a/22348885/359001
  def __init__(self, seconds=1, error_message='Timeout'):
    self.seconds = seconds
    self.error_message = error_message
  def handle_timeout(self, signum, frame):
    raise TimeoutError(self.error_message)
  def __enter__(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)
  def __exit__(self, type, value, traceback):
    signal.alarm(0)

def getCurrentTimeStamp() -> str:
  """
  Return timestamp as string. Ex. '2023-07-24T12_16_04'
  """
  return datetime.now().isoformat(timespec='seconds').replace(':', '_')

def get_iso_timestamp_with_zone() -> str:
  """
  Return current timestamp in ISO format. Useful for postgres timestamp with zone insertion.

  Ex. '2025-03-26T17:17:42.364646+00:00'
  """
  return datetime.now(zoneinfo.ZoneInfo("UTC")).isoformat()

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
  """
  Formats integer with comma separators.
  123456789 -> '123,456,789'
  """
  return "{:,}".format(n)

def print_module_versions(modules_list: Sequence[str]):
  """
  Prints the versions of specified modules.

  Parameters:
  - modules_list (Sequence[str]): A list or sequence of strings representing the names
    of the modules for which the version information is to be printed.

  Example usage:
  >>> print_module_versions(['numpy', 'pandas'])
  numpy: 1.19.2
  pandas: 1.1.3
  """
  print(f'python: {platform.python_version()}')

  for m in modules_list:
    print(f'{m}: {version(m)}')

def get_env_var(env_var: str) -> str:
  """Returns the value of `env_var` if it exists in env vars."""
  try:
    return os.environ[env_var]
  except KeyError:
    raise ValueError(f"Environment variable {env_var} is not set")

def print_time_elapsed(start_time: float):
  """Prints elapsed time in human readable format"""
  end_time = timer()
  elapsed = end_time - start_time
  
  if elapsed < 1:
    result_str = f"ran in {elapsed*1000:.2f} milliseconds"
  elif elapsed < 60:
    result_str = f"ran in {elapsed:.2f} seconds"
  elif elapsed < 3600:
    minutes, seconds = divmod(elapsed, 60)
    result_str = f"ran in {int(minutes)} minutes and {seconds:.2f} seconds"
  else:
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    result_str = f"ran in {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds"

  print(result_str, flush=True)

def import_from_path(module_name: str, module_path: str) -> ModuleType:
  """
    Dynamically imports a Python module from a file path.

    Args:
        module_name (str): The name to give to the imported module. This name will be used
                          to reference the module in the program.
        module_path (str): The absolute or relative file path to the Python module file (.py).
                          Should include the file extension.

    Returns:
        module (ModuleType): The imported module object that can be used to access its attributes and functions.

    Raises:
        ValueError: If the module specification cannot be created (spec is None).
        ImportError: If the module cannot be loaded or executed.
        FileNotFoundError: If the specified module_path does not exist.

    Examples:
        >>> # Import a module with a custom name
        >>> utils = import_from_path("utils", "/path/to/utilities.py")
        >>> utils.some_function()
        
        >>> # Import a local module
        >>> helper = import_from_path("helper", "./helper_functions.py")
        >>> result = helper.process_data()
    
    Notes:
        - The module_name parameter determines how you'll reference the module after import
        - The module_path must point to a valid .py file
        - Use absolute paths when possible to avoid path resolution issues
    """
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  if spec is None:
    raise ValueError(f"Failed to import - spec is None")
  mymodule = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mymodule) # type:ignore
  return mymodule
