import argparse
import os
import shutil
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss

import utils
from challenge.dataset import NUM_CLASSES
from challenge.dataset import get_train_valid_loaders
from core.train import SupervisedTrainer

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to configuration file')
    parser.add_argument('--device_ids', nargs='+', type=int, default=utils.get_all_device_ids(), help='GPU device ids')
    parser.add_argument('--clean', action='store_true', help='clean the experiment folder before training')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to the checkpoint')
    return parser.parse_args()


def output_transform(output):
    y_pred, y = output
    y = y.argmax(dim=1)
    return y_pred, y


def get_onehot_collate_fn(num_classes: int) -> Callable:
    def _onehot_collate_fn(batch: Tuple) -> Tuple:
        x = torch.stack([it for it, _ in batch])
        y = torch.stack([it for _, it in batch])
        y = F.one_hot(y, num_classes=num_classes).type(torch.FloatTensor)
        return x, y

    return _onehot_collate_fn


if __name__ == '__main__':
    args = parse_args()
    config = utils.load_config(args.config)

    if args.clean:
        shutil.rmtree(config.path, ignore_errors=True)

    utils.fix_seed(0)
    logger = utils.get_logger(name='train', path=config.path, pathx=config.path)
    logger.info(f'Process PID: {os.getpid()}')
    logger.info(f'Using {len(args.device_ids)} GPUs')

    root = Path(config.root)
    loaders = get_train_valid_loaders(
        path=root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_transform=config.train_aug,
        valid_transform=config.valid_aug,
        collate_fn=get_onehot_collate_fn(NUM_CLASSES)
    )

    model = config.get_model()
    criterion = config.criterion()

    device = utils.get_default_device(device_ids=args.device_ids)

    num_finished_epochs = 0
    for num_epochs in config.stages:
        if num_epochs > 0:
            optimizer = config.optimizer(model)

            model = model.to(device)
            criterion = criterion.to(device)

            opt_level = getattr(config, 'opt_level', 'O0')
            mixed_precision = False if opt_level == 'O0' else True

            if mixed_precision:
                model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

            if args.device_ids and len(args.device_ids) > 1:
                model = nn.DataParallel(model, device_ids=args.device_ids)

            trainer = SupervisedTrainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loaders=loaders,
                num_epochs=num_epochs,
                start_epoch=num_finished_epochs,
                metrics={
                    'loss': Loss(criterion),
                    'acc': Accuracy(output_transform=output_transform)
                },
                device=device,
                backup_path=config.backup_path,
                mixed_precision=mixed_precision,
                stage='main'
            )

            trainer.run()

        num_finished_epochs += num_epochs
        utils.model.unfreeze(model)
