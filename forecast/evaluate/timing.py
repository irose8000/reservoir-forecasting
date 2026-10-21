"""Measure how long a block of code takes (for the cost comparison)."""
import time


class Timer:
    """Context manager that records time in seconds.

    Usage:
        with Timer() as t:
            model.fit(train, val)
        print(t.seconds)
    """

    def __init__(self):
        self.seconds = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        # Runs even if the timed block raises, so a failed fit still reports its time.
        self.seconds = time.perf_counter() - self._start
        return False 