import time
import statistics
import signal
import zoneinfo
from datetime import datetime
from timeit import default_timer as timer

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

class TimeoutContext:
  """
  A context manager that enforces a timeout on code execution using Unix signals.

  This class implements a timeout mechanism using SIGALRM, allowing you to limit
  the execution time of a code block. If the timeout is exceeded, a TimeoutError
  is raised.

  see: https://stackoverflow.com/a/22348885/359001

  Note: This implementation only works on Unix-like systems (Linux, macOS, etc.)
  as it relies on SIGALRM, which is not available on Windows.

  Attributes:
      seconds (int): The timeout duration in seconds
      error_message (str): Custom error message for the TimeoutError

  Example:
      >>> with TimeoutContext(seconds=5):
      ...     long_running_operation()  # Will raise TimeoutError after 5 seconds
      
      >>> with TimeoutContext(seconds=10, error_message='Operation took too long'):
      ...     another_operation()

  Raises:
      TimeoutError: When the code block exceeds the specified timeout duration

  Limitations:
      - Only works on Unix-like systems (not Windows)
      - Cannot be nested (SIGALRM doesn't support nesting)
      - Only supports integer seconds (not fractional seconds)
  """
  def __init__(self, seconds=1, error_message='Timeout'):
    if not hasattr(signal, 'SIGALRM'):
        raise NotImplementedError("TimeoutContext is not supported on this platform (requires Unix signals)")
    
    self.seconds = seconds
    self.error_message = error_message
  
  def handle_timeout(self, signum, frame):
    raise TimeoutError(self.error_message)
  
  def __enter__(self) -> 'TimeoutContext':
    self.old_handler = signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)
    return self

  def __exit__(self, type, value, traceback):
    signal.alarm(0)
    signal.signal(signal.SIGALRM, self.old_handler) # restore previous signal handler

class TimerContext:
    """
    Context timer for muliple durations

    # Example usage:
    timer = TimerContext()

    for _ in range(3):
      with timer:
        # simulate some work
        time.sleep(0.2)

    timer.print_stats()
    """
    def __init__(self, description="my-timer", decimal_places=2, print_each=False):
        self.description = description
        self.decimal_places = decimal_places
        self.print_each = print_each
        self.durations = []
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.print_each:
           print(f'Starting: {self.description}')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.durations.append(duration)
        if self.print_each:
           print(f'Completed: {self.description}. Duration: {duration:.{self.decimal_places}f}')
    
    def print_stats(self):
        count_val = len(self.durations)
        stdev_val = f"{statistics.stdev(self.durations):.{self.decimal_places}f} seconds" if count_val > 1 else "N/A"
        print("-" * 60)
        print(f'Stats for: {self.description}')
        if count_val > 0:
          print(f"Mean:      {statistics.mean(self.durations):.{self.decimal_places}f} seconds")
          print(f"Median:    {statistics.median(self.durations):.{self.decimal_places}f} seconds")
          print(f"Min:       {min(self.durations):.{self.decimal_places}f} seconds")
          print(f"Max:       {max(self.durations):.{self.decimal_places}f} seconds")
          print(f"Total:     {sum(self.durations):.{self.decimal_places}f} seconds")
          print(f"StdDev:    {stdev_val}")
        print(f"Count:     {count_val}")
        print("-" * 60)