import signal
import sys
from types import FrameType
from typing import Optional, Type

class ShutdownManager:
    """
    A context manager to gracefully handle SIGTERM and SIGINT (Ctrl+C).

    It encapsulates the shutdown state, avoiding the need for global variables. guarantee that your custom signal handling is only active while your code is running.
    Avoid overwriting signal handlers for entire application when your code is imported as a module.

    # testing: `kill -SIGTERM <process id>` or CTRL-C in terminal (for SIGINT)
    # see test script: `test_signal.py` (old version - needs update)

    Usage:
        with ShutdownManager() as manager:
            while not manager.is_shutdown_requested():
                # do work...
                if some_condition:
                    break
    """
    def __init__(self):
        self._shutdown_requested = False
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        self.original_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def __enter__(self):
        """Register the signal handlers and return the manager instance."""
        print("Graceful shutdown manager activated. Listening for SIGTERM/SIGINT.")
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        """Restore the original signal handlers upon exiting the context."""
        print("Restoring original signal handlers.")
        signal.signal(signal.SIGINT, self.original_sigint_handler)
        signal.signal(signal.SIGTERM, self.original_sigterm_handler)

    def _handle_signal(self, signum: int, frame: Optional[FrameType]) -> None:
        """The internal signal handler that sets the shutdown flag."""
        # Check if already requested to avoid spamming logs on repeated signals
        if not self._shutdown_requested:
            signame = signal.Signals(signum).name
            print(f"\nShutdown signal received ({signame}). Finishing current task...")
            self._shutdown_requested = True

    def is_shutdown_requested(self) -> bool:
        """Check if a shutdown has been requested."""
        return self._shutdown_requested

    # You can also make it behave like a boolean for more concise checks
    def __bool__(self) -> bool:
        return self._shutdown_requested

    def check_and_exit(self, exit_status = 1):
        """Check if a shutdown has been requested and then exit if so."""
        if self._shutdown_requested:
            print("Shutdown was requested - exiting now.")
            sys.exit(exit_status)

# Example usage if the module is run directly
if __name__ == "__main__":
    import time
    print("Running example usage of ShutdownManager.")
    print("Press Ctrl+C to test the graceful shutdown.")

    with ShutdownManager() as manager:
        counter = 0
        while not manager.is_shutdown_requested():
            print(f"Working... (loop {counter})")
            time.sleep(1)
            counter += 1
            if counter > 10:
                print("Loop finished naturally.")
                break
    
    print("Script has exited the 'with' block.")
