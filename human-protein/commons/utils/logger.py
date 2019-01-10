import logging
import sys
from pathlib import Path
from typing import Union

from tensorboardX import SummaryWriter


class Logger(logging.Logger):
    """This class allows you to log information to the console, file and tensorboardX."""

    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%m/%d/%Y %I:%M:%S %p'

    def __init__(self,
                 name: str = 'logger',
                 level: int = logging.INFO,
                 path: Union[str, Path] = None,
                 pathx: Union[str, Path] = None):
        super().__init__(name, level)
        self.path = path
        self.pathx = pathx
        self.writers = dict()
        self._init_logger()

    def scalar_summary(self, logger_tag: str, tag: str, value: float, step: int):
        if self.pathx is None:
            pass

        if logger_tag not in self.writers:
            self.writers[logger_tag] = SummaryWriter(f'{self.pathx}/{logger_tag}')

        self.writers[logger_tag].add_scalar(tag, value, step)

    def _init_logger(self):
        """Create logger handlers."""

        formatter = logging.Formatter(Logger.LOG_FORMAT, Logger.DATE_FORMAT)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        self.addHandler(stream_handler)

        if self.path is not None:
            file_handler = logging.FileHandler(self.path)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
