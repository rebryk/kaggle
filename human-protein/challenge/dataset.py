from typing import Tuple, List, Any

import albumentations as album
import numpy as np
import pandas as pd
from PIL import Image
from attrdict import AttrDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler, Sampler
from torchvision import transforms


class HumanProteinDataset(Dataset):
    NUM_CLASSES: int = 28

    def __init__(self, df: pd.DataFrame, path: str, train_mode: bool = True, transforms=None):
        self.df = df
        self.path = path
        self.train_mode = train_mode
        self.transform = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index) -> Tuple:
        image_id = self.df.iloc[index].Id
        x = self._read_image(image_id)
        y = self.parse_target(self.df.iloc[index].Target) if self.train_mode else image_id

        if self.transform is not None:
            augmented = self.transform(image=np.array(x))
            x = transforms.ToTensor()(augmented['image'])

        return x, y

    @staticmethod
    def parse_target(target: str) -> np.ndarray:
        y = np.zeros(HumanProteinDataset.NUM_CLASSES, dtype=np.int)
        indices = [int(it) for it in target.split()]
        y[indices] = 1
        return y

    def _read_image(self, image_id: str) -> Image:
        bands = []

        for ch in ['red', 'green', 'blue']:
            bands.append(Image.open(f'{self.path}/{image_id}_{ch}.png').convert('L'))

        return Image.merge('RGB', bands=bands)


def _get_transforms(image_size: int) -> Tuple[album.Compose, album.Compose, album.Compose]:
    transforms_train = album.Compose([
        album.Resize(image_size, image_size, interpolation=Image.BICUBIC),
        album.Rotate(interpolation=Image.BICUBIC),
        album.RandomRotate90(),
        album.HorizontalFlip(),
        album.RandomBrightnessContrast(),
        album.Normalize([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
    ])

    transforms_test = album.Compose([
        album.Resize(image_size, image_size, interpolation=Image.BICUBIC),
        album.Normalize([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
    ])

    transforms_test_aug = album.Compose([
        album.Resize(image_size, image_size, interpolation=Image.BICUBIC),
        album.Rotate(interpolation=Image.BICUBIC),
        album.RandomRotate90(),
        album.HorizontalFlip(),
        album.Normalize([0.08069, 0.05258, 0.05487], [0.13704, 0.10145, 0.15313])
    ])

    return transforms_train, transforms_test, transforms_test_aug


def _k_fold(df: pd.DataFrame, n_splits: int, random_state: Any = 42):
    X = np.array(df.Id)
    y = np.array([HumanProteinDataset.parse_target(target) for target in df.Target])
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, random_state=random_state)
    return mskf.split(X, y)


def _get_sampler(df: pd.DataFrame, alpha: float = 0.5) -> Sampler:
    y = np.array([HumanProteinDataset.parse_target(target) for target in df.Target])
    class_weights = np.round(np.log(alpha * y.sum() / y.sum(axis=0)), 2)
    class_weights[class_weights < 1.0] = 1.0

    weights = np.zeros(len(df))
    for i, target in enumerate(y):
        weights[i] = class_weights[target == 1].max()

    return WeightedRandomSampler(weights, len(df))


def get_loaders(path: str,
                image_size: int,
                n_splits: int = 1,
                test_size: float = 0.1,
                batch_size: int = 128,
                num_workers: int = 4,
                external: bool = False,
                use_sampler: bool = False) -> Tuple[AttrDict, List[AttrDict]]:
    df = pd.read_csv(f'{path}/train.csv')
    df_external = pd.read_csv(f'{path}/external.csv')

    X = np.array(df.Id)
    y = np.array([HumanProteinDataset.parse_target(target) for target in df.Target])

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train, valid = list(msss.split(X, y))[0]

    df_train, df_valid = df.iloc[train], df.iloc[valid]
    df_test = pd.read_csv(f'{path}/sample_submission.csv')

    if external:
        df_train = pd.concat([df_train, df_external], axis=0)

    transforms_train, transforms_test, transforms_test_aug = _get_transforms(image_size)

    dataset_train = HumanProteinDataset(df_train, f'{path}/train', transforms=transforms_train)
    dataset_train_test = HumanProteinDataset(df_train, f'{path}/train', transforms=transforms_test)
    dataset_train_aug = HumanProteinDataset(df_train, f'{path}/train', transforms=transforms_test_aug)
    dataset_valid = HumanProteinDataset(df_valid, f'{path}/train', transforms=transforms_test)
    dataset_valid_aug = HumanProteinDataset(df_valid, f'{path}/train', transforms=transforms_test_aug)
    dataset_test = HumanProteinDataset(df_test, f'{path}/test', train_mode=False, transforms=transforms_test)
    dataset_test_aug = HumanProteinDataset(df_test, f'{path}/test', train_mode=False, transforms=transforms_test_aug)

    default_loaders = AttrDict()
    default_loaders.train = DataLoader(dataset_train, batch_size, num_workers=num_workers)
    default_loaders.train_test = DataLoader(dataset_train_test, batch_size, num_workers=num_workers)
    default_loaders.train_aug = DataLoader(dataset_train_aug, batch_size, num_workers=num_workers)
    default_loaders.valid = DataLoader(dataset_valid, batch_size, pin_memory=True, num_workers=num_workers)
    default_loaders.valid_aug = DataLoader(dataset_valid_aug, batch_size, pin_memory=True, num_workers=num_workers)
    default_loaders.test = DataLoader(dataset_test, batch_size, pin_memory=True, num_workers=num_workers)
    default_loaders.test_aug = DataLoader(dataset_test_aug, batch_size, pin_memory=True, num_workers=num_workers)

    if n_splits == 1:
        sampler = _get_sampler(df_train) if use_sampler else None

        loaders = AttrDict()
        loaders.train = DataLoader(dataset_train, batch_size, not use_sampler, sampler, num_workers=num_workers)
        loaders.valid = default_loaders.valid
        loaders.valid_aug = default_loaders.valid_aug

        return default_loaders, [loaders]

    folds = []

    for train, valid in _k_fold(df_train, n_splits):
        fold_train, fold_valid = df_train.iloc[train], df_train.iloc[valid]
        dataset_train = HumanProteinDataset(fold_train, f'{path}/train', transforms=transforms_train)
        dataset_valid = HumanProteinDataset(fold_valid, f'{path}/train', transforms=transforms_test)
        dataset_valid_aug = HumanProteinDataset(fold_valid, f'{path}/train', transforms=transforms_test_aug)

        sampler = _get_sampler(fold_train) if use_sampler else None

        loaders = AttrDict()
        loaders.train = DataLoader(dataset_train, batch_size, not use_sampler, sampler, num_workers=num_workers)
        loaders.valid = DataLoader(dataset_valid, batch_size, pin_memory=True, num_workers=num_workers)
        loaders.valid_aug = DataLoader(dataset_valid_aug, batch_size, pin_memory=True, num_workers=num_workers)

        folds.append(loaders)

    return default_loaders, folds
