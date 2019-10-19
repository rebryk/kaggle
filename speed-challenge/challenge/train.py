import argparse
import sys

import utils
from challenge.dataset import get_train_valid_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to configuration file')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs in the node')
    parser.add_argument('--local_rank', type=int, default=-1, help='Distributed training - Local rank')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = utils.load_config(args.config)

    utils.set_seed(0)
    utils.create(f'{config.path}/logs')
    log_filename = f'train.log' if args.local_rank == -1 else f'train_{args.local_rank}.log'
    utils.set_log_path(f'{config.path}/logs/{log_filename}')
    logger = utils.get_logger(__name__, stream=sys.stdout if args.local_rank <= 0 else None)

    # Initialize GPUs
    config.n_gpu = args.n_gpu
    config.local_rank = args.local_rank
    utils.init_gpu_params(config)

    loaders = get_train_valid_loaders(
        root=config.root,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        multi_gpu=config.multi_gpu
    )

    model = config.get_model()
    optimizer = config.get_optimizer(model)
    criterion = config.get_criterion()

    if config.is_master:
        num_total_params, num_train_params = utils.get_model_size(model)
        logger.info(f'Number of trainable parameters: {num_train_params}')
        logger.info(f'Number of parameters: {num_total_params}')

    trainer = utils.SupervisedTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=config.num_epochs,
        n_gpu=config.n_gpu,
        is_master=config.is_master,
        local_rank=config.local_rank,
        max_grad_norm=config.max_grad_norm,
        mixed_precision=config.mixed_precision,
        backup_path=config.path,
        log_interval=config.log_interval
    )

    trainer.train()
