import os

import torch
from ignite.metrics import Loss

from challenge.dataset import get_loaders
from challenge.loss import BCEWithLogitsLoss
from challenge.metric import Accuracy, F1Score
from challenge.train import train_model
from challenge.utils import load_model
from commons.utils import init_logger, get_logger, remove_resource_limits
from commons.utils.model import unfreeze
from config import config

if __name__ == '__main__':
    remove_resource_limits()

    init_logger(f'{config.exp}_sz{config.image_size}_x{config.batch_size}', config.log_path, config.tensorboard_path)
    logger = get_logger()
    logger.info(f'PID: {os.getpid()}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    default_loaders, loaders = get_loaders(path=config.data_path,
                                           image_size=config.image_size,
                                           n_splits=config.k_fold,
                                           test_size=config.test_size,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           external=config.external_data,
                                           use_sampler=config.use_sampler)

    metrics = {
        'loss': Loss(BCEWithLogitsLoss()),
        'acc': Accuracy(),
        'F1': F1Score()
    }

    logger.info(f'Model: {config.model}')
    logger.info(f'External data: {config.external_data}')

    logger.info(f'K-fold: {config.k_fold}')
    logger.info(f'Mixed precision: {config.mixed_precision}')
    logger.info(f'Image size: {config.image_size}')
    logger.info(f'Batch size: {config.batch_size}')

    for fold in range(config.k_fold):
        if config.k_fold > 1:
            logger.info(f'Fold: {fold + 1}')
            suffix = f'_f{fold + 1}'
        else:
            suffix = ''

        if config.checkpoint:
            checkpoint = config.checkpoint.format(fold + 1) if config.k_fold > 1 else config.checkpoint
            logger.info(f'Use checkpoint: checkpoint_{checkpoint}.pth')
            model = load_model(config.model, f'{config.model_path}/checkpoint_{checkpoint}.pth', config.mixed_precision)
            unfreeze(model)
        else:
            model = load_model(config.model)

        train_model(name=f'{config.exp}_sz{config.image_size}_x{config.batch_size}{suffix}',
                    model=model,
                    data_loaders=loaders[fold],
                    metrics=metrics,
                    device=device,
                    lr=config.lr,
                    num_epochs=config.num_epochs,
                    cycles_len=config.cycles_len,
                    lr_divs=config.lr_divs,
                    mixed_precision=config.mixed_precision,
                    backup_path=config.model_path)
