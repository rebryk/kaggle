import argparse
import os
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Loss

import utils
from challenge.dataset import EXP_TRAIN
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
    parser.add_argument('--checkpoint', type=str, default=None, required=True, help='path to the checkpoint')
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

    utils.fix_seed(0)
    logger = utils.get_logger(name='train', path=config.path, pathx=config.path)
    logger.info(f'Process PID: {os.getpid()}')
    logger.info(f'Using {len(args.device_ids)} GPUs')

    root = Path(config.root)

    device = utils.get_default_device(device_ids=args.device_ids)

    for experiment_id in range(len(EXP_TRAIN)):
        loaders = get_train_valid_loaders(
            path=root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            train_transform=config.train_aug,
            valid_transform=config.valid_aug,
            collate_fn=get_onehot_collate_fn(NUM_CLASSES),
            experiment_id=experiment_id,
            drop_last=True
        )

        model = config.get_model()
        utils.model.unfreeze(model)

        optimizer = config.optimizer(model)

        model = model.to(device)
        criterion = config.criterion()

        opt_level = getattr(config, 'opt_level', 'O0')
        mixed_precision = False if opt_level == 'O0' else True

        if mixed_precision:
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        logger.info(f'Loading checkpoint: {args.checkpoint}')
        utils.load_checkpoint(model, args.checkpoint)

        if args.device_ids and len(args.device_ids) > 1:
            model = nn.DataParallel(model, device_ids=args.device_ids)

        trainer = SupervisedTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=config.num_finetuning_epoch,
            metrics={
                'loss': Loss(criterion),
                'acc': Accuracy(output_transform=output_transform)
            },
            device=device,
            backup_path=config.backup_path,
            mixed_precision=mixed_precision,
            stage=f'finetuning_{experiment_id}'
        )

        trainer.run()

        del model
