from abc import abstractmethod, ABCMeta
from collections import deque


class MovingAverage(metaclass=ABCMeta):
    @abstractmethod
    def update(self, value: float):
        pass

    @abstractmethod
    def value(self) -> float:
        pass


class SimpleMovingAverage(MovingAverage):
    def __init__(self, window_size: int):
        self.window_size = window_size
        self._total = 0.0
        self._values = deque()

    def update(self, value: float):
        self._total += value
        self._values.append(value)

        if len(self._values) == self.window_size + 1:
            self._total -= self._values.popleft()

    def value(self) -> float:
        return self._total / len(self._values)


class ExponentialMovingAverage(MovingAverage):
    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self._loss = 0
        self._count = 0

    def update(self, value: float):
        self._loss = self._loss * self.alpha + value * (1 - self.alpha)
        self._count += 1

    def value(self) -> float:
        return self._loss / (1 - self.alpha ** self._count)
