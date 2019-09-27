import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import utils
from challenge.dataset import EXP_TRAIN
from utils.neighbors import k_neighbors_classify, k_neighbors_classify_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='+', type=str, required=True, help='path to configuration file')
    parser.add_argument('-n', '--n_neighbors', nargs='+', type=int, required=False, default=[20, 20, 20, 20],
                        help='number of neighbors')
    parser.add_argument('--use_valid', action='store_true', help='whether to use valid for test predictions')
    return parser.parse_args()


def get_group_scores(embeddings_train: np.ndarray,
                     labels_train: np.ndarray,
                     groups_train: np.ndarray,
                     embeddings_test: np.ndarray,
                     n_neighbors: int) -> np.ndarray:
    scores = np.zeros(4, dtype=float)

    for group in range(4):
        mask = groups_train == group
        _, scores_ = k_neighbors_classify(
            X_train=embeddings_train[mask],
            y_train=labels_train[mask],
            X_test=embeddings_test,
            n_neighbors=n_neighbors
        )
        scores[group] = scores_.mean()

    return scores


def get_train_group_mapping(root: Path) -> dict:
    # Mapping from the first sirna in group to group number
    sirna_to_group = {0: 0, 1: 1, 2: 2, 4: 3}

    df_train = pd.read_csv(root / 'train.csv')
    groups = df_train.groupby(["experiment", "plate"]).apply(lambda it: sirna_to_group[it.sirna.min()])

    return dict(groups.items())


def get_group_redictions(embeddings_train: np.ndarray,
                         labels_train: np.ndarray,
                         embeddings_test: np.ndarray,
                         n_neighbors: int) -> np.ndarray:
    scores, labels = k_neighbors_classify_scores(
        X_train=embeddings_train,
        y_train=labels_train,
        X_test=embeddings_test,
        n_neighbors=n_neighbors
    )

    _, col_ind = linear_sum_assignment(-scores)
    preds = labels[col_ind]

    return preds


def get_predictions(embeddings_train: np.ndarray,
                    labels_train: np.ndarray,
                    groups_train: np.ndarray,
                    embeddings_test: np.ndarray,
                    experiments_test: np.ndarray,
                    plates_test: np.ndarray,
                    n_neighbors: int) -> np.ndarray:
    preds = np.zeros(len(experiments_test), dtype=int)
    plates = np.array([1, 2, 3, 4])

    for experiment in np.unique(experiments_test):
        plate_group_score = np.zeros((4, 4), dtype=float)

        for i, plate in enumerate(plates):
            mask_test = (experiments_test == experiment) & (plates_test == plate)

            plate_group_score[i] = get_group_scores(
                embeddings_train=embeddings_train,
                labels_train=labels_train,
                groups_train=groups_train,
                embeddings_test=embeddings_test[mask_test],
                n_neighbors=n_neighbors
            )

        # Match groups with plates
        rows, groups = linear_sum_assignment(-plate_group_score)

        for plate, group in zip(plates, groups):
            mask_test = (experiments_test == experiment) & (plates_test == plate)
            mask_train = (groups_train == group)

            preds[mask_test] = get_group_redictions(
                embeddings_train=embeddings_train[mask_train],
                labels_train=labels_train[mask_train],
                embeddings_test=embeddings_test[mask_test],
                n_neighbors=n_neighbors
            )

    return preds


if __name__ == '__main__':
    args = parse_args()
    config = utils.load_config(args.config[0])

    utils.fix_seed(0)
    logger = utils.get_logger(name='test', path=config.path)

    df_result = pd.DataFrame(columns=['id_code', 'sirna'])

    root = Path(config.root)
    train_group_mapping = get_train_group_mapping(root)

    acc_scores = []

    if len(args.n_neighbors) == 1:
        args.n_neighbors = args.n_neighbors * 4

    configs = [utils.load_config(it) for it in args.config]

    for exp_id in range(len(EXP_TRAIN)):
        data = dict()

        for stage in ['train', 'valid', 'test']:
            embeddings = []
            labels = None
            plates = None
            experiments = None

            for config_ in configs:
                data_path = f'{config_.path}/data/{exp_id}'
                labels_ = np.load(f'{data_path}/labels_{stage}.npy', allow_pickle=True)
                embeddings_ = np.load(f'{data_path}/embeddings_{stage}.npy', allow_pickle=True)
                plates_ = np.load(f'{data_path}/plates_{stage}.npy', allow_pickle=True)
                experiments_ = np.load(f'{data_path}/experiments_{stage}.npy', allow_pickle=True)

                # Average embeddings for sites
                n = len(labels_) // 2
                labels_ = labels_[:n]
                plates_ = plates_[:n]
                experiments_ = experiments_[:n]
                embeddings_ = (embeddings_[:n] + embeddings_[n:]) / 2

                # Collect embeddings
                embeddings.append(embeddings_)
                labels = labels_
                plates = plates_
                experiments = experiments_

            # Average embeddings for experiments
            embeddings = np.mean(embeddings, axis=0)

            data[stage] = {
                'labels': labels,
                'embeddings': embeddings,
                'plates': plates,
                'experiments': experiments,
            }

            if stage != 'test':
                data[stage]['groups'] = np.array([train_group_mapping[it] for it in zip(experiments, plates)])

        embeddings_train = data['train']['embeddings']
        labels_train = data['train']['labels']
        groups_train = data['train']['groups']

        logger.info(f'Making predictions for valid...')
        pred = get_predictions(
            embeddings_train=embeddings_train,
            labels_train=labels_train,
            groups_train=groups_train,
            embeddings_test=data['valid']['embeddings'],
            experiments_test=data['valid']['experiments'],
            plates_test=data['valid']['plates'],
            n_neighbors=args.n_neighbors[exp_id]
        )

        acc_score = np.mean(pred == data['valid']['labels'])
        acc_scores.append(acc_score)
        logger.info(f'Valid accuracy score for experiment {exp_id}: {acc_score:0.3f}')

        # Use validation data
        if args.use_valid:
            embeddings_train = np.concatenate([embeddings_train, data['valid']['embeddings']])
            labels_train = np.concatenate([labels_train, data['valid']['labels']])
            groups_train = np.concatenate([groups_train, data['valid']['groups']])

        logger.info(f'Making predictions for test...')
        df_test = pd.DataFrame({
            'id_code': data['test']['labels'],
            'sirna': get_predictions(
                embeddings_train=embeddings_train,
                labels_train=labels_train,
                groups_train=groups_train,
                embeddings_test=data['test']['embeddings'],
                experiments_test=data['test']['experiments'],
                plates_test=data['test']['plates'],
                n_neighbors=args.n_neighbors[exp_id]
            )
        })

        df_result = df_result.append(df_test)

    logger.info(f'Valid accuracy score: {np.mean(acc_scores):0.3f}')
    df_result.to_csv(f'{config.path}/pred.csv', index=False)
