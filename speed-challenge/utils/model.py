import itertools
from typing import Dict, List, Tuple

import numpy as np
import pretrainedmodels
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


def freeze(model: nn.Module):
    """Freeze all model parameters."""

    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: nn.Module):
    """Unfreeze all model parameters."""

    for param in model.parameters():
        param.requires_grad = True


def get_group_params(groups: List[List[nn.Module]], lrs: np.ndarray) -> List[Dict]:
    """Create dicts defining parameter groups for an optimizer."""

    group_params = []

    for group, lr in zip(groups, lrs):
        params = {'params': list(itertools.chain(*[layer.parameters() for layer in group])), 'lr': lr}
        group_params.append(params)

    return group_params


def get_grad_norm(model: nn.Module, norm_type: int = 2) -> float:
    """Calculate gradients norm."""

    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def load_model(model_name: str, num_classes: int, pretrained: str):
    return pretrainedmodels.__dict__[model_name](num_classes=num_classes, pretrained=pretrained)


def get_model_size(model: torch.nn.Module) -> Tuple[int, int]:
    """Return number of model parameters."""

    num_total_params = sum([p.numel() for p in model.parameters()])
    num_train_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num_total_params, num_train_params


def save_checkpoint(path: str,
                    epoch: int,
                    model: nn.Module,
                    optimizer: Optimizer,
                    scheduler: _LRScheduler):
    """Save the checkpoint."""

    state = {
        'epoch': epoch,
        'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optimizer = None,
                    scheduler: _LRScheduler = None) -> int:
    """Load the checkpoint."""

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint['epoch']
