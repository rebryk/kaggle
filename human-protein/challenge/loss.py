import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if target.size() != input.size():
            raise ValueError(f'Target size ({target.size()}) must be the same as input size ({input.size()})')

        input = input.float()
        target = target.float()

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y):
        return self.loss(y_pred.double(), y.double())
