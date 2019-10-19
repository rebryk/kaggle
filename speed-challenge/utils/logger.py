import logging
from typing import Any

LOG_PATH = None
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s'
DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
FORMATTER = logging.Formatter(LOG_FORMAT, DATE_FORMAT)


def get_logger(name: str, level: int = logging.INFO, stream: Any = None) -> logging.Logger:
    """Create logger with handlers."""

    logger = logging.Logger(name)

    if stream is not None:
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(FORMATTER)
        logger.addHandler(handler)

    if LOG_PATH is not None:
        handler = logging.FileHandler(LOG_PATH)
        handler.setLevel(level)
        handler.setFormatter(FORMATTER)
        logger.addHandler(handler)

    return logger


def set_log_path(path: str):
    global LOG_PATH

    LOG_PATH = path
