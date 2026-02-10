import os
import sys
import math
import psutil
import subprocess
import csv
from pprint import pprint
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

  return f'{round(resultSize, 1)} {unit}'

class CPU_cores(NamedTuple):
  """
  Represents the number of CPU cores on a system.

  Attributes:
      physical (int): The number of physical cores on the CPU.
      logical (int): The number of logical cores on the CPU. This count
                      includes cores resulting from technologies like Intel's 
                      Hyper-Threading.
  """
  physical: int | None
  logical: int | None

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
  result = subprocess.run(command.split(), capture_output=True, text=True)
  csv_output = result.stdout.strip()
  reader = csv.DictReader(csv_output.splitlines())
  data = next(reader, None)  # Get the single data row as a dictionary
  pprint(data)

def print_system_info():
  print('Current interpreter python version:')
  print(sys.version)
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
  mem_info = process.memory_info()
  # mem_info.vms for VMS
  return getSizePretty(mem_info.rss)

def parse_smaps(pid):
    """
    Parse /proc/[pid]/smaps and return memory breakdown in MB.
    
    The smaps file is a Linux pseudo-file that provides detailed memory 
    consumption statistics for each memory mapping in a process. Unlike 
    /proc/[pid]/status or /proc/[pid]/statm which give aggregate figures,
    smaps breaks down memory usage per-mapping with categories including:
    
    - Shared_Clean: Memory shared with other processes, unmodified since mapped
    - Shared_Dirty: Memory shared with other processes, modified (needs writeback)
    - Private_Clean: Memory exclusive to this process, unmodified (e.g., mapped code)
    - Private_Dirty: Memory exclusive to this process, modified (heap, stack, COW pages)
    
    This distinction is critical for understanding actual memory cost:
    - Shared memory is typically copy-on-write mappings (libraries, forked data)
      and doesn't count toward per-process memory pressure
    - Private_Dirty is the "true" memory cost unique to this process
    - Private_Clean can be reclaimed/reloaded from disk if needed
    
    Parameters
    ----------
    pid : int or str
        Process ID to inspect.
    
    Returns
    -------
    dict or None
        Dictionary with memory stats in MB:
        - 'shared_clean', 'shared_dirty', 'total_shared'
        - 'private_clean', 'private_dirty', 'total_private'
        Returns None if the process doesn't exist or is inaccessible.
    """
    try:
        with open(f'/proc/{pid}/smaps', 'r') as f:
            lines = f.readlines()
    except (FileNotFoundError, PermissionError):
        return None
    
    shared_clean = 0
    shared_dirty = 0
    private_clean = 0
    private_dirty = 0
    
    for line in lines:
        if line.startswith('Shared_Clean:'):
            shared_clean += int(line.split()[1])
        elif line.startswith('Shared_Dirty:'):
            shared_dirty += int(line.split()[1])
        elif line.startswith('Private_Clean:'):
            private_clean += int(line.split()[1])
        elif line.startswith('Private_Dirty:'):
            private_dirty += int(line.split()[1])
    
    return {
        'shared_clean': shared_clean / 1024,
        'shared_dirty': shared_dirty / 1024,
        'private_clean': private_clean / 1024,
        'private_dirty': private_dirty / 1024,
        'total_shared': (shared_clean + shared_dirty) / 1024,
        'total_private': (private_clean + private_dirty) / 1024
    }


def print_memory_stats(pid, label="Process"):
    """
    Print formatted memory statistics for a process.
    
    Retrieves and displays a human-readable breakdown of shared vs private
    memory usage, useful for diagnosing memory behavior in multiprocessing
    scenarios (e.g., verifying copy-on-write efficiency after fork()).
    
    Parameters
    ----------
    pid : int or str
        Process ID to inspect.
    label : str, optional
        Descriptive label for the output (default: "Process").
    
    Returns
    -------
    dict or None
        The memory stats dictionary from parse_smaps(), or None if unavailable.
    
    Example Output
    --------------
        Worker 0 (PID 12345):
          Shared memory:    512.00 MB
            - Clean:        510.00 MB
            - Dirty:          2.00 MB
          Private memory:    64.00 MB
            - Clean:         12.00 MB
            - Dirty:         52.00 MB
    """
    stats = parse_smaps(pid)
    if stats:
        print(f"\n{label} (PID {pid}):")
        print(f"  Shared memory:  {stats['total_shared']:8.2f} MB")
        print(f"    - Clean:      {stats['shared_clean']:8.2f} MB")
        print(f"    - Dirty:      {stats['shared_dirty']:8.2f} MB")
        print(f"  Private memory: {stats['total_private']:8.2f} MB")
        print(f"    - Clean:      {stats['private_clean']:8.2f} MB")
        print(f"    - Dirty:      {stats['private_dirty']:8.2f} MB")
        return stats
    return None
