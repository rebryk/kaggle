from typing import List

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWithRestarts(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_len: int,
                 lr_div: int = 0,
                 last_epoch: int = -1):
        self.cycle_len = cycle_len
        self.lr_div = lr_div

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cycle_iter = self.last_epoch % self.cycle_len
        k = (1 + math.cos(math.pi * cycle_iter / (self.cycle_len - 1))) / 2
        return [base_lr * (1 + k * (self.lr_div - 1)) / self.lr_div for base_lr in self.base_lrs]

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(cycle_len: {self.cycle_len}, lr_div: {self.lr_div})'


class CircularLR(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 cycle_len: int,
                 lr_div: float,
                 cut_div: int,
                 last_epoch: int = -1):
        self.cycle_len = cycle_len
        self.lr_div = lr_div
        self.cut_div = cut_div
        self._cut_iter = cycle_len // cut_div
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        cycle_iter = self.last_epoch % self.cycle_len

        if cycle_iter <= self._cut_iter:
            k = cycle_iter / self._cut_iter
        else:
            k = (self.cycle_len - cycle_iter) / (self.cycle_len - self._cut_iter)

        return [base_lr * (1 + k * (self.lr_div - 1)) / self.lr_div for base_lr in self.base_lrs]

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(cycle_len: {self.cycle_len}, lr_div: {self.lr_div}, cut_div: {self.cut_div})'
