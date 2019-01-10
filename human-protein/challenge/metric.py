import ignite.metrics
import numpy as np
from sklearn.metrics import f1_score


class Accuracy(ignite.metrics.Accuracy):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        super().__init__()

    def update(self, output):
        y_pred, y = output
        y_pred = (y_pred > self.threshold).float()
        super().update((y_pred, y))


class F1Score(ignite.metrics.Metric):
    def __init__(self, threshold: float = 0.0, average='macro'):
        self.threshold = threshold
        self.average = average
        self._y_pred = []
        self._y_true = []
        super().__init__()

    def update(self, output):
        y_pred, y_true = output
        y_pred = (y_pred > self.threshold).int()
        self._y_pred.append(y_pred.cpu().numpy())
        self._y_true.append(y_true.cpu().numpy())

    def reset(self):
        self._y_pred = []
        self._y_true = []

    def compute(self):
        return f1_score(np.concatenate(self._y_true), np.concatenate(self._y_pred), average=self.average)
