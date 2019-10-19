import numbers
from typing import Tuple, List

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.image import load_image


def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor


def optical_flow_dense(image_current: np.array, image_next: np.array) -> np.array:
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros_like(image_current)

    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Obtain dense optical flow parameters
    flow = cv2.calcOpticalFlowFarneback(
        prev=gray_current,
        next=gray_next,
        flow=None,
        pyr_scale=0.5,
        levels=1,
        winsize=15,
        iterations=2,
        poly_n=5,
        poly_sigma=1.3,
        flags=0
    )

    # Convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # HUE corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)

    # Value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to float32
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow


class ImageDataset(Dataset):
    def __init__(self,
                 root: str,
                 df: pd.DataFrame,
                 enable_augmentation: bool = False,
                 is_test: bool = False):
        self.root = root
        self.df = df
        self.is_test = is_test

        self.enable_augmentation = enable_augmentation
        self.augmentation = albu.Compose(
            [albu.RandomBrightnessContrast(), albu.HorizontalFlip()],
            additional_targets={'image_next': 'image'}
        )

    def __len__(self) -> int:
        # We can not use the last image as the first one in the tuple
        return len(self.df) - 1

    @staticmethod
    def _process_image(image: np.array) -> np.array:
        # Remove sky and hood, left and right stripes
        return image[100:380, 45:-45]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Take average speed
        if self.is_test:
            label = self.df.iloc[index, 0]
        else:
            # Note: dataset is not shuffled, so we can do this
            label = (self.df.iloc[index, 1] + self.df.iloc[index + 1, 1]) / 2.0

        image_index = self.df.iloc[index, 0]
        image = self._process_image(load_image(f'{self.root}/{image_index}.jpg'))
        image_next = self._process_image(load_image(f'{self.root}/{image_index + 1}.jpg'))

        # Apply augmentation
        if self.enable_augmentation:
            transformed = self.augmentation(image=image, image_next=image_next)
            image, image_next = transformed['image'], transformed['image_next']

        image = optical_flow_dense(image, image_next)
        image_tensor = img_to_tensor(image)

        # There is no need to convert labels to tensors, when you create a test dataset,
        # where the label type is not a number
        if isinstance(label, numbers.Number):
            label = torch.tensor(label)

        return image_tensor, label


def load_labels(path: str) -> List:
    with open(path, 'r') as f:
        return [float(it.strip()) for it in f.readlines()]


def get_train_valid_loaders(root: str,
                            batch_size: int,
                            train_ratio: float,
                            multi_gpu: bool,
                            is_test: bool = False):
    y_train = load_labels(f'{root}/train.txt')

    df = pd.DataFrame({
        'index': range(len(y_train)),
        'speed': y_train
    })

    num_train = int(df.shape[0] * train_ratio)
    df_train, df_valid = df.iloc[:num_train], df.iloc[num_train:]

    dataset_train = ImageDataset(f'{root}/train', df_train, is_test=is_test)
    dataset_valid = ImageDataset(f'{root}/train', df_valid, is_test=is_test)

    train_sampler = DistributedSampler(dataset_train, shuffle=True) if multi_gpu else None
    valid_sampler = DistributedSampler(dataset_valid, shuffle=False) if multi_gpu else None

    return {
        'train': DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=None if train_sampler else True,
            sampler=train_sampler
        ),
        'valid': DataLoader(
            dataset=dataset_valid,
            batch_size=batch_size,
            shuffle=None if valid_sampler else False,
            sampler=valid_sampler
        )
    }


def get_test_loader(root: str,
                    batch_size: int,
                    multi_gpu: bool):
    df_test = pd.DataFrame({'index': range(10798)})
    dataset_valid = ImageDataset(f'{root}/test', df_test, is_test=True)
    test_sampler = DistributedSampler(dataset_valid, shuffle=False) if multi_gpu else None

    return DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=None if test_sampler else False,
        sampler=test_sampler
    )
