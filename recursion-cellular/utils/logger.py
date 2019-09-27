import logging
import sys
from pathlib import Path
from typing import Union, Any

from torch.utils.tensorboard import SummaryWriter


class Logger(logging.Logger):
    """This class allows you to log information to the console, file and tensorboardX."""

    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%m/%d/%Y %I:%M:%S %p'

    def __init__(self,
                 name: str = 'logger',
                 level: int = logging.INFO,
                 path: Union[str, Path] = None,
                 pathx: Union[str, Path] = None,
                 stream: Any = None):
        super().__init__(name, level)
        self.pathx = pathx
        self.writers = dict()

        formatter = logging.Formatter(Logger.LOG_FORMAT, Logger.DATE_FORMAT)

        if path is not None:
            handler = logging.FileHandler(path)
            handler.setLevel(self.level)
            handler.setFormatter(formatter)
            self.addHandler(handler)

        if stream is not None:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
            handler.setFormatter(formatter)
            self.addHandler(handler)

    def scalar_summary(self, logger_tag: str, tag: str, value: float, step: int):
        if self.pathx is None:
            pass

        if logger_tag not in self.writers:
            self.writers[logger_tag] = SummaryWriter(f'{self.pathx}/{logger_tag}')

        self.writers[logger_tag].add_scalar(tag, value, step)
