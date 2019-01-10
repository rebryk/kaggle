from typing import Tuple

import torch


class AdaptiveConcatPool2d(torch.nn.Module):
    def __init__(self, sz: Tuple[int, int] = (1, 1)):
        super().__init__()
        self.average_pool = torch.nn.AdaptiveAvgPool2d(sz)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.max_pool(x), self.average_pool(x)], 1)
