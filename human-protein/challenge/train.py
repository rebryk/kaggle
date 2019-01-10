from typing import Dict, List

import numpy as np
import torch
from attrdict import AttrDict
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from challenge.loss import BCEWithLogitsLoss
from commons.schedulers import CircularLR
from commons.trainer import Trainer
from commons.utils import get_logger
from commons.utils.model import unfreeze, fix_seed, get_group_params


def train_model(name: str,
                model: torch.nn.Module,
                data_loaders: AttrDict,
                metrics: Dict,
                device: torch.cuda.device,
                lr: float,
                num_epochs: List[int],
                cycles_len: List[int],
                lr_divs: List[int],
                mixed_precision: bool = False,
                backup_path: str = None):
    fix_seed()
    logger = get_logger()

    num_batches = len(data_loaders.train)
    criterion = BCEWithLogitsLoss()

    param_groups = [
        [model.conv1, model.bn1, model.layer1, model.layer2],
        [model.layer3, model.layer4],
        [model.last_linear]
    ]

    lrs = np.array([lr / 10, lr / 3, lr])

    logger.info(f'Learning rate: {lr:.5f}')
    logger.info(f'Learning rates: {lrs}')

    finished_epochs = 0
    for it, (epochs, lr_div, cycle_len) in enumerate(zip(num_epochs, lr_divs, cycles_len)):
        logger.info('Creating new trainer...')
        logger.info(f'Epochs: {epochs}')
        logger.info('Optimizer: Adam')

        if lr_div:
            optimizer = Adam(get_group_params(param_groups, lrs / lr_div), lr=lr)
            logger.info(f'Learning rate divider: {lr_div}')
        else:
            optimizer = Adam(model.parameters(), lr=lr)

        if cycle_len:
            scheduler = CircularLR(optimizer, cycle_len=cycle_len * num_batches, lr_div=10, cut_div=30)
            logger.info(f'Scheduler: {scheduler}')
        else:
            scheduler = None

        trainer = Trainer(name=name,
                          model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          metrics=metrics,
                          data_loaders=data_loaders,
                          max_norm=1,
                          scheduler=scheduler,
                          is_iteration_scheduler=True,
                          device=device,
                          mixed_precision=mixed_precision,
                          finished_epochs=finished_epochs)
        trainer.train(num_epochs=epochs)
        finished_epochs += epochs
        trainer.save_checkpoint(f'{backup_path}/checkpoint_{name}_{finished_epochs:02}.pth')
        unfreeze(model)


def find_lr(*,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            metrics: Dict,
            data_loaders: AttrDict,
            num_epochs: int,
            device: torch.cuda.device,
            min_lr: float = 1e-4,
            max_lr: float = 1):
    min_arg = np.log(min_lr)
    max_arg = np.log(max_lr)
    gamma = np.exp((max_arg - min_arg) / (num_epochs * len(data_loaders.train) - 1))
    scheduler = ExponentialLR(optimizer, gamma)

    trainer = Trainer(model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      metrics=metrics,
                      data_loaders=data_loaders,
                      max_norm=1,
                      scheduler=scheduler,
                      is_iteration_scheduler=True,
                      device=device,
                      name='find_lr')
    trainer.train(num_epochs=num_epochs)
