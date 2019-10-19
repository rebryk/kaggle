import time


class Timer:
    """Class to measure execution time."""

    def __init__(self):
        self.count = 0
        self.total_duration = 0
        self._start = None

    def __enter__(self):
        self.start()

    def __exit__(self, *args, **kwargs):
        self.stop()

    def reset(self):
        self.count = 0
        self.total_duration = 0

    def start(self):
        self._start = time.time()

    def stop(self):
        self.total_duration += time.time() - self._start
        self.count = self.count + 1

    @property
    def average(self) -> float:
        """Return average execution time."""

        return self.total_duration / self.count if self.count else 0
