import os
from typing import List

import numpy as np
import torch
from apex.fp16_utils import network_to_half
from attrdict import AttrDict
from ignite.metrics import Loss

import challenge.utils as utils
from challenge.dataset import get_loaders, HumanProteinDataset
from challenge.loss import FocalLoss
from challenge.metric import Accuracy, F1Score
from challenge.model import get_model
from commons.utils import init_logger, get_logger, remove_resource_limits
from commons.utils.model import fix_seed, load_checkpoint
from config import config


def evaluate(models: List[torch.nn.Module], loaders: AttrDict):
    fix_seed()

    logger.info(f'Train n_aug: {config.n_aug_train}')
    logger.info(f'Test n_aug: {config.n_aug_test}')

    logger.info('Evaluating train...')
    y_pred_train, y_true_train = utils.eval_ensemble(models,
                                                     loaders.train_test,
                                                     loaders.train_aug,
                                                     n_aug=config.n_aug_train,
                                                     device=device)
    y_pred_train = np.mean(y_pred_train, axis=-1) if config.n_aug_train > 0 or config.k_fold > 1 else y_pred_train
    y_pred_train = utils.sigmoid(y_pred_train)

    logger.info('Evaluating valid...')
    y_pred_valid, y_true_valid = utils.eval_ensemble(models,
                                                     loaders.valid,
                                                     loaders.valid_aug,
                                                     n_aug=config.n_aug_test,
                                                     device=device)
    y_pred_valid = np.mean(y_pred_valid, axis=-1) if config.n_aug_test > 0 or config.k_fold > 1 else y_pred_valid
    y_pred_valid = utils.sigmoid(y_pred_valid)

    logger.info('Evaluating test...')
    y_pred_test, y_true_test = utils.eval_ensemble(models,
                                                   loaders.test,
                                                   loaders.test_aug,
                                                   n_aug=config.n_aug_test,
                                                   device=device)
    y_pred_test = np.mean(y_pred_test, axis=-1) if config.n_aug_test > 0 or config.k_fold > 1 else y_pred_test
    y_pred_test = utils.sigmoid(y_pred_test)

    for threshold in [0.5, 0.3, 0.2]:
        logger.info(f'Threshold: {threshold:.2f}')

        thresholds = np.zeros(HumanProteinDataset.NUM_CLASSES) + threshold
        utils.eval_thresholds(y_true_train, y_pred_train, thresholds, 'train')
        utils.eval_thresholds(y_true_valid, y_pred_valid, thresholds, 'valid')

        path = f'{config.submission_path}/{config.exp}_{int(10 * threshold):02}.csv'
        utils.save_submission(y_pred_test > threshold, y_true_test, path)


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
        'loss': Loss(FocalLoss()),
        'acc': Accuracy(),
        'F1': F1Score()
    }

    logger.info(f'External data: {config.external_data}')

    models = []

    for fold in range(config.k_fold):
        checkpoint = config.checkpoint.format(fold + 1) if config.k_fold > 1 else config.checkpoint
        logger.info(f'Use checkpoint: checkpoint_{checkpoint}.pth')

        model = get_model(config.model, HumanProteinDataset.NUM_CLASSES)
        checkpoint = load_checkpoint(f'{config.model_path}/checkpoint_{checkpoint}.pth')

        if config.mixed_precision:
            model = network_to_half(model)

        model.load_state_dict(checkpoint['state_dict'])
        models.append(model)

    evaluate(models, default_loaders)
