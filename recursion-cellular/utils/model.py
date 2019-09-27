import itertools
from typing import Dict, List

import numpy as np
import pretrainedmodels
import torch


def freeze(model: torch.nn.Module):
    """Freeze all model parameters."""

    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: torch.nn.Module):
    """Unfreeze all model parameters."""

    for param in model.parameters():
        param.requires_grad = True


def get_group_params(groups: List[List[torch.nn.Module]], lrs: np.ndarray) -> List[Dict]:
    """Create dicts defining parameter groups for an optimizer."""

    group_params = []

    for group, lr in zip(groups, lrs):
        params = {'params': list(itertools.chain(*[layer.parameters() for layer in group])), 'lr': lr}
        group_params.append(params)

    return group_params


def get_norm(model: torch.nn.Module, norm_type: int = 2) -> float:
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def load_model(model_name: str, num_classes: int, pretrained: str):
    return pretrainedmodels.__dict__[model_name](num_classes=num_classes, pretrained=pretrained)


def save_checkpoint(epoch: int,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    path: str):
    state = {
        'epoch': epoch,
        'state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
