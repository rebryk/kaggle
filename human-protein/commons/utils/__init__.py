from pathlib import Path
from typing import Union

from .common import get_uuid, remove_resource_limits
from .logger import Logger
from .path import create, _to_path

logger: Logger = None


def get_logger() -> Logger:
    return logger


def init_logger(name: str, path: Union[str, Path] = None, pathx: Union[str, Path] = None):
    """
    Initialize logger.

    :param name: log file name
    :param path: log directory path
    :param pathx: tensorboardX log directory path
    """
    global logger

    if path is not None:
        create(path)
        path = _to_path(path) / f'{name}.log'

    if pathx is not None:
        create(pathx)

    logger = Logger(path=path, pathx=pathx)
    logger.info(f'Log file name: {name}')


__all__ = ['get_logger', 'init_logger', 'Logger', 'remove_resource_limits']
