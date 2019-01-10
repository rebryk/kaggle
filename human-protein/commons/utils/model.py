import itertools
import random
from typing import Dict, List

import numpy as np
import torch


def fix_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def freeze(model: torch.nn.Module):
    """Freeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model: torch.nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True


def save_checkpoint(state: dict, filename: str):
    torch.save(state, filename)


def load_checkpoint(filename: str) -> Dict:
    return torch.load(filename)


def get_group_params(groups: List[List[torch.nn.Module]], lrs: np.ndarray) -> List[Dict]:
    """Create dicts defining parameter groups for an optimizer."""
    group_params = []

    for group, lr in zip(groups, lrs):
        params = {'params': list(itertools.chain(*[layer.parameters() for layer in group])), 'lr': lr}
        group_params.append(params)

    return group_params
