from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from albumentations import Compose
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utils.dataset import ImageClassificationDataset

NUM_CLASSES = 1108

EXP_TRAIN = [
    ['HEPG2-01', 'HEPG2-02', 'HEPG2-03', 'HEPG2-04', 'HEPG2-05', 'HEPG2-06', 'HEPG2-07'],
    ['HUVEC-01', 'HUVEC-02', 'HUVEC-03', 'HUVEC-04', 'HUVEC-05', 'HUVEC-06', 'HUVEC-07',
     'HUVEC-08', 'HUVEC-09', 'HUVEC-10', 'HUVEC-11', 'HUVEC-12', 'HUVEC-13', 'HUVEC-14', 'HUVEC-15',
     'HUVEC-16'],
    ['RPE-01', 'RPE-02', 'RPE-03', 'RPE-04', 'RPE-05', 'RPE-06', 'RPE-07'],
    ['U2OS-01', 'U2OS-02', 'U2OS-03']
]

EXP_TEST = [
    ['HEPG2-08', 'HEPG2-09', 'HEPG2-10', 'HEPG2-11'],
    ['HUVEC-17', 'HUVEC-18', 'HUVEC-19', 'HUVEC-20', 'HUVEC-21', 'HUVEC-22', 'HUVEC-23', 'HUVEC-24'],
    ['RPE-08', 'RPE-09', 'RPE-10', 'RPE-11'],
    ['U2OS-04', 'U2OS-05']
]

VALID_SIZE = [1, 1, 1, 1]


def _add_sites(df: pd.DataFrame) -> pd.DataFrame:
    df1 = df.copy()
    df1['site'] = 1

    df2 = df1.copy()
    df2['site'] = 2

    return pd.concat([df1, df2])


def _to_files(row) -> List[List[str]]:
    return [f'{row.experiment}/Plate{row.plate}/{row.well}_s{row.site}_w{w}.png' for w in range(1, 7)]


def _format_train_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            'files': df.apply(_to_files, axis=1),
            'label': df['sirna']
        }
    )


def _format_test_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        data={
            'files': df.apply(_to_files, axis=1),
            'label': df.id_code
        }
    )


def _split_train_valid(df: pd.DataFrame, experiment_id: int) -> Tuple:
    experiments_ids = range(len(EXP_TRAIN)) if experiment_id is None else [experiment_id]

    experiments_train = []
    experiments_valid = []

    for experiment_id in experiments_ids:
        ids = EXP_TRAIN[experiment_id]
        cnt = len(ids) - VALID_SIZE[experiment_id]
        experiments_train += ids[:cnt]
        experiments_valid += ids[cnt:]

    df_train = df[np.isin(df.experiment, experiments_train)]
    df_valid = df[np.isin(df.experiment, experiments_valid)]

    return df_train, df_valid


def get_original_train_valid_dataset(path: Path, experiment_id: int) -> Tuple:
    df_train = pd.read_csv(path / 'train.csv')

    df_train = _add_sites(df_train)
    df_train, df_valid = _split_train_valid(df_train, experiment_id)

    return df_train, df_valid


def get_train_valid_dataset(path: Path, experiment_id: int) -> Tuple:
    df_train, df_valid = get_original_train_valid_dataset(path, experiment_id)
    return _format_train_dataframe(df_train), _format_train_dataframe(df_valid)


def get_train_valid_loaders(path: Path,
                            batch_size: int,
                            num_workers: int = 0,
                            train_transform: Compose = None,
                            valid_transform: Compose = None,
                            shuffle: bool = True,
                            experiment_id: int = None,
                            drop_last: bool = False,
                            collate_fn=default_collate) -> Dict:
    df_train, df_valid = get_train_valid_dataset(path, experiment_id)
    dataset_train = ImageClassificationDataset(path / 'train', df_train, train_transform)
    dataset_valid = ImageClassificationDataset(path / 'train', df_valid, valid_transform)

    loaders = {
        'train': DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last
        ),
        'valid': DataLoader(
            dataset=dataset_valid,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    }

    return loaders


def get_original_test_dataset(path: Path, experiment_id: int) -> pd.DataFrame:
    df_test = pd.read_csv(path / 'test.csv')
    df_test = _add_sites(df_test)

    if experiment_id is not None:
        mask = np.isin(df_test.experiment, EXP_TEST[experiment_id])
        df_test = df_test[mask]

    return df_test


def get_test_loader(path: Path,
                    batch_size: int,
                    num_workers: int = 0,
                    test_transform: Compose = None,
                    experiment_id: int = None) -> DataLoader:
    df_test = get_original_test_dataset(path, experiment_id)
    df_test = _format_test_dataframe(df_test)

    dataset_test = ImageClassificationDataset(path / 'test', df_test, test_transform)

    loader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    return loader
