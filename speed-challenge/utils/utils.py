import importlib
import importlib.util
import os
import random
import socket
import sys
import uuid
from types import ModuleType
from typing import List, Any

import numpy as np
import psutil
import torch
import torch.distributed as dist

from utils import get_logger


def get_uuid() -> str:
    """Generate UUID."""

    return str(uuid.uuid4())


def set_seed(seed: int = 0):
    """Set the random seed."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_memory_usage() -> float:
    """Return  memory usage in MBs."""

    return psutil.virtual_memory()._asdict()['used'] / 1_000_000


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


def init_gpu_params(params: ModuleType):
    """Handle single and multi-GPU / multi-node."""

    logger = get_logger(__name__, stream=sys.stdout if params.local_rank <= 0 else None)

    if params.n_gpu <= 0:
        params.local_rank = 0
        params.master_port = -1
        params.is_master = True
        params.multi_gpu = False
        return

    assert torch.cuda.is_available()

    logger.info('Initializing GPUs')
    if params.n_gpu > 1:
        assert params.local_rank != -1

        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['N_GPU_NODE'])
        params.global_rank = int(os.environ['RANK'])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.multi_gpu = True

        assert params.n_nodes == int(os.environ['N_NODES'])
        assert params.node_id == int(os.environ['NODE_RANK'])
    else:
        # Local job (single GPU)
        assert params.local_rank == -1

        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1
        params.multi_gpu = False

    # Sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # Define whether this is the master process / if we are in multi-node distributed mode
    params.is_master = params.global_rank == 0
    params.multi_node = params.n_nodes > 1

    # Summary
    prefix = f'--- Global rank: {params.global_rank} - '
    logger.info(f'{prefix}Number of nodes: {params.n_nodes}')
    logger.info(f'{prefix}Node ID        : {params.node_id}')
    logger.info(f'{prefix}Local rank     : {params.local_rank}')
    logger.info(f'{prefix}World size     : {params.world_size}')
    logger.info(f'{prefix}GPUs per node  : {params.n_gpu_per_node}')
    logger.info(f'{prefix}Master         : {params.is_master}')
    logger.info(f'{prefix}Multi-node     : {params.multi_node}')
    logger.info(f'{prefix}Multi-GPU      : {params.multi_gpu}')
    logger.info(f'{prefix}Hostname       : {socket.gethostname()}')

    # Set GPU device
    torch.cuda.set_device(params.local_rank)

    # Initialize multi-GPU
    if params.multi_gpu:
        logger.info('Initializing PyTorch distributed')
        dist.init_process_group(
            init_method='env://',
            backend='nccl',
            rank=params.global_rank,
            world_size=params.world_size
        )
