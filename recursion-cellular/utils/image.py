from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_mask(path: Union[str, Path]) -> np.array:
    """
    Read grayscale mask from the disk.

    :param path: path to a local file on disk
    :return: image as a NumPy array
    """

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise RuntimeError(f'Unable to load mask {str(path)}')

    return mask.astype(np.uint8)


def load_image(path: Union[str, Path]) -> np.array:
    """
    Read three channel image into the RGB format.

    :param path: path to a local file on disk
    :return: image as a NumPy array
    """

    img = cv2.imread(str(path))

    if img is None:
        raise RuntimeError(f'Unable to load image {str(path)}')

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
