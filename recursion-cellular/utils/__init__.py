import sys
from pathlib import Path
from typing import Union, Any

from utils.logger import Logger
from utils.model import load_checkpoint
from utils.model import save_checkpoint
from utils.path import create
from utils.path import to_path
from utils.utils import fix_seed
from utils.utils import get_all_device_ids
from utils.utils import get_default_device
from utils.utils import load_config

logger: Logger = None


def get_logger(*args, **kwargs) -> Logger:
    if logger is None:
        init_logger(*args, **kwargs)

    return logger


def init_logger(name: str = 'logger',
                path: Union[str, Path] = 'logs',
                pathx: Union[str, Path] = 'logs',
                stream: Any = sys.stdout):
    """
    Initialize logger.

    :param name: log file name
    :param path: log directory path
    :param pathx: tensorboardX log directory path
    :param stream: log stream
    """

    global logger

    if path is not None:
        create(path)
        path = to_path(path) / f'{name}.log'

    if pathx is not None:
        create(pathx)

    logger = Logger(name, path=path, pathx=pathx, stream=stream)
    logger.info(f'Log file path: {str(path)}')


__all__ = ['get_logger', 'init_logger', 'Logger', 'create', 'to_path', 'fix_seed', 'get_all_device_ids',
           'get_default_device', 'load_config', 'load_checkpoint', 'save_checkpoint']
