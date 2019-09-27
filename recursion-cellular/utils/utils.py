import importlib.util
import random
import uuid
from typing import List, Any

import numpy as np
import torch


def get_uuid() -> str:
    """
    Generate UUID.

    :return: generated UUID
    """

    return str(uuid.uuid4())


def fix_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(path: str) -> Any:
    spec = importlib.util.spec_from_file_location('config', path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def get_all_device_ids() -> List[int]:
    return [it for it in range(torch.cuda.device_count())]


def get_default_device(device_ids: List[int]) -> torch.device:
    if not torch.cuda.is_available() or not device_ids:
        return torch.device('cpu')

    return torch.device(f'cuda:{device_ids[0]}')
