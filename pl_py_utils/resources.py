import os
import math
import psutil
import subprocess
import pandas as pd
from typing import Any, Callable, Sequence, NamedTuple
from io import StringIO

from .getsize import total_size

def getSizeObject(obj: object) -> str:
  """
  Calculate and retrieve the memory size of a given object in a human-readable format.

  The function first determines the total size (in bytes) of the provided object and 
  then converts this value into a formatted, human-readable string representation.

  Args:
      obj (object): The Python object whose memory size is to be calculated.

  Returns:
      str: A string representation of the object's memory size. The string format
            includes the numerical size and a unit (b, kb, mb, or gb).
  """
  totalSizeBytes = total_size(obj)
  return getSizePretty(totalSizeBytes)

def getSizePretty(totalSizeBytes: int) -> str:
  """
  Convert a size in bytes to a human-readable format.

  This function takes in a size in bytes and returns it as a formatted string with 
  an appropriate unit. The units considered are bytes (b), kilobytes (kb), megabytes 
  (mb), and gigabytes (gb).

  Args:
      totalSizeBytes (int): The size in bytes to be converted.

  Returns:
      str: A human-readable string representation of the provided size in bytes.
            The string format includes the numerical size rounded to three decimal 
            places and its respective unit.
  """
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

class CPU_cores(NamedTuple):
  """
  Represents the number of CPU cores on a system.

  Attributes:
      physical (int): The number of physical cores on the CPU.
      logical (int): The number of logical cores on the CPU. This count
                      includes cores resulting from technologies like Intel's 
                      Hyper-Threading.
  """
  physical: int
  logical: int

def num_cpu_cores() -> CPU_cores:
  """
  Retrieve the number of physical and logical CPU cores on the system.

  Uses the `psutil` library to get the count of CPU cores. 

  Returns:
      CPU_cores: A named tuple containing the count of physical and logical 
                  CPU cores.
  """
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

def get_process_memory_usage():
  """
  Retrieve the memory usage of the current process.

  Uses the `psutil` library to get the memory usage (resident set size) of 
  the process with the ID equal to the current process's ID. The memory 
  usage is then formatted into a human-readable format using `getSizePretty`.

  Returns:
      str: The formatted string representing the memory usage of the 
            current process.
  """
  process = psutil.Process(os.getpid())
  return getSizePretty(process.memory_info().rss)