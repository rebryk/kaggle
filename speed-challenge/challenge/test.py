import argparse
import sys

import pandas as pd

import utils
from challenge.dataset import get_train_valid_loaders, get_test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--output', type=str, required=True, help='path to the output folder')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs in the node')
    parser.add_argument('--local_rank', type=int, default=-1, help='Distributed training - Local rank')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = utils.load_config(args.config)

    utils.set_seed(0)
    utils.create(args.output)
    log_filename = f'test.log' if args.local_rank == -1 else f'test_{args.local_rank}.log'
    utils.set_log_path(f'{config.path}/logs/{log_filename}')
    logger = utils.get_logger(__name__, stream=sys.stdout if args.local_rank <= 0 else None)

    # Initialize GPUs
    config.n_gpu = args.n_gpu
    config.local_rank = args.local_rank
    utils.init_gpu_params(config)

    model = config.get_model()
    tester = utils.Tester(
        model=model,
        checkpoint=args.checkpoint,
        n_gpu=config.n_gpu,
        is_master=config.is_master,
        local_rank=config.local_rank
    )

    loaders = get_train_valid_loaders(
        root=config.root,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        multi_gpu=config.multi_gpu,
        is_test=True
    )
    loaders['test'] = get_test_loader(
        root=config.root,
        batch_size=config.batch_size_test,
        multi_gpu=config.multi_gpu
    )

    for loader in loaders:
        ids, y_preds = tester(loaders[loader])

        if config.is_master:
            df = pd.DataFrame({
                'index': ids.astype(int),
                'y_preds': y_preds.astype(float)
            })

            logger.info(f'Saving results for {loader}')
            df.to_csv(f'{args.output}/pred_{loader}.csv', index=False)
