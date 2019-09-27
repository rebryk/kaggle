import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch.nn as nn

import utils
from challenge.dataset import get_original_train_valid_dataset, get_original_test_dataset
from challenge.dataset import get_test_loader, get_train_valid_loaders, EXP_TRAIN
from core.test import Tester
from utils.module import Identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to configuration file')
    parser.add_argument('--device_ids', nargs='+', type=int, default=utils.get_all_device_ids(), help='GPU device ids')
    parser.add_argument('--checkpoint', nargs='+', type=str, required=True, help='path to the checkpoints')
    parser.add_argument('--n_aug', type=int, default=1, help='number of test augmentations')
    return parser.parse_args()


def fold(result: np.array, output: Tuple) -> Tuple:
    ids, y_pred = output
    y_pred = y_pred.cpu().detach().numpy()

    if result is None:
        return ids, y_pred
    else:
        return np.concatenate([result[0], ids]), np.concatenate([result[1], y_pred])


if __name__ == '__main__':
    args = parse_args()
    config = utils.load_config(args.config)

    utils.fix_seed(0)
    logger = utils.get_logger(name='test', path=config.path)

    root = Path(config.root)

    device = utils.get_default_device(device_ids=args.device_ids)

    tester = None

    for exp_id in range(len(EXP_TRAIN)):
        logger.info(f'Experiment id: {exp_id}')

        if exp_id == 0 or len(EXP_TRAIN) == len(args.checkpoint):
            model = config.model()

            logger.info(f'Loading checkpoint: {args.checkpoint[exp_id]}')
            utils.load_checkpoint(model, args.checkpoint[exp_id])

            model.last_linear = model.last_linear[0] if isinstance(model.last_linear, nn.Sequential) else Identity()

            if args.device_ids and len(args.device_ids) > 1:
                model = nn.DataParallel(model, device_ids=args.device_ids).cuda()

            tester = Tester(model=model, fold=fold, device=device)

        data_path = f'{config.path}/data/{exp_id}'
        utils.path.create(data_path)

        # Load original data
        df_train, df_valid = get_original_train_valid_dataset(path=root, experiment_id=exp_id)
        df_test = get_original_test_dataset(path=root, experiment_id=exp_id)
        df = {
            'train': df_train,
            'valid': df_valid,
            'test': df_test
        }

        loaders = get_train_valid_loaders(
            path=root,
            batch_size=config.inference_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            train_transform=config.train_aug,
            valid_transform=config.valid_aug,
            experiment_id=exp_id,
        )

        loaders['test'] = get_test_loader(
            path=root,
            batch_size=config.inference_batch_size,
            num_workers=config.num_workers,
            test_transform=config.test_aug,
            experiment_id=exp_id
        )

        for stage in ['valid', 'train', 'test']:
            logger.info(f'Running inference on {stage} data...')
            embeddings_ = []

            for i in range(args.n_aug):
                labels, embeddings = tester(loaders[stage])
                embeddings_.append(embeddings)

            embeddings = np.stack(embeddings_).mean(axis=0)
            plates = df[stage].plate.values
            experiments = df[stage].experiment.values

            np.save(f'{data_path}/labels_{stage}.npy', labels)
            np.save(f'{data_path}/embeddings_{stage}.npy', embeddings)
            np.save(f'{data_path}/plates_{stage}.npy', plates)
            np.save(f'{data_path}/experiments_{stage}.npy', experiments)
