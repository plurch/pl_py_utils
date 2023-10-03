import ctypes

# Access the _PyTime_AsSecondsDouble and _PyTime_GetSystemClock functions from pythonapi
get_system_clock = ctypes.pythonapi._PyTime_GetSystemClock
as_seconds_double = ctypes.pythonapi._PyTime_AsSecondsDouble

# Set the argument types and return types of the functions
get_system_clock.argtypes = []
get_system_clock.restype = ctypes.c_int64

as_seconds_double.argtypes = [ctypes.c_int64]
as_seconds_double.restype = ctypes.c_double

def get_current_time() -> float:
  """
  Retrieve the current system time as a floating-point number of seconds.

  # call njit in application code:
  # get_current_time_njit = (njit(nogil=True))(get_current_time)
  # get_current_time_njit = cast(FunctionType, get_current_time_njit)
  # https://github.com/numba/numba/issues/4003#issuecomment-1578146534

  Returns:
  --------
  float
      The current system time represented as a floating-point number of seconds.

  Notes:
  ------
  - `get_system_clock` is assumed to return a system-specific clock object.
  - `as_seconds_double` is assumed to convert the clock object to seconds in float format.

  Examples:
  ---------
  >>> time = get_current_time()
  >>> print(f"Current time: {time} seconds")
  Current time: 1633320485.358723 seconds

  """
  system_clock = get_system_clock()
  return as_seconds_double(system_clock)
