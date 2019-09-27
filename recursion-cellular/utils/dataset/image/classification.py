import numbers
from pathlib import Path
from typing import Tuple, List, Optional

import albumentations
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset

from utils.image import load_image, load_mask


class ImageClassificationDataset(Dataset):
    def __init__(self,
                 root: Path,
                 df: pd.DataFrame,
                 transform: Optional[albumentations.Compose] = None,
                 use_cache: bool = True):
        self.root = root
        self.df = df
        self.transform = transform
        self.use_cache = use_cache
        self.cache = [None] * len(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        files = self.df.iloc[index, 0]
        label = self.df.iloc[index, 1]

        if self.cache[index] is None:
            if isinstance(files, str):
                image = load_image(self.root / files)
            elif isinstance(files, List):
                channels = [load_mask(self.root / it) for it in files]
                image = np.stack(channels, axis=-1)
            else:
                raise RuntimeError(f'Failed to parse image path: {files}')

            if self.use_cache:
                self.cache[index] = image
        else:
            image = self.cache[index]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        image_tensor = img_to_tensor(image)

        # There is no need to convert labels to tensors, when you create a test dataset,
        # where the label type is not a number
        if isinstance(label, numbers.Number):
            label = torch.tensor(label)

        return image_tensor, label
