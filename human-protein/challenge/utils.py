from typing import List, Union, Optional

import numpy as np
import pandas as pd
import scipy.optimize as opt
import torch
from apex.fp16_utils import network_to_half
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from challenge.dataset import HumanProteinDataset
from challenge.model import get_model
from commons.math import sigmoid
from commons.test import tta, predict_with_targets
from commons.utils import get_logger
from commons.utils.model import load_checkpoint


def save_submission(y_pred: np.ndarray, ids: np.ndarray, path: str):
    labels = [' '.join([str(label) for label in np.nonzero(it)[0]]) for it in y_pred]
    df = pd.DataFrame({'Id': ids, 'Predicted': labels})
    df.to_csv(path, index=False)


def load_model(model_name: str, path: Optional[str] = None, mixed_precision: bool = False) -> torch.nn.Module:
    model = get_model(model_name, HumanProteinDataset.NUM_CLASSES)

    if mixed_precision:
        model = network_to_half(model)

    if path is not None:
        checkpoint = load_checkpoint(path)
        model.load_state_dict(checkpoint['state_dict'])

    return model


def eval(model: torch.nn.Module,
         loader: DataLoader,
         loader_aug: DataLoader = None,
         n_aug: int = 0,
         device: torch.cuda.device = None):
    if n_aug > 0:
        loader_aug = loader_aug or loader
        y_pred, y_true = tta(model, loader, loader_aug, n_aug=n_aug, device=device)
    else:
        y_pred, y_true = predict_with_targets(model, loader, device=device)

    return y_pred, y_true


def eval_ensemble(models: List[torch.nn.Module],
                  loader: DataLoader,
                  loader_aug: DataLoader = None,
                  n_aug: int = 0,
                  device: torch.cuda.device = None):
    y_pred = []
    y_true = None

    for i in range(len(models)):
        y_pred_, y_true = eval(models[i], loader, loader_aug, n_aug, device)
        y_pred.append(y_pred_)

    return np.concatenate(y_pred, axis=-1), y_true


def find_threshold(y_true: np.ndarray, y_pred: np.ndarray, step: float = 0.1) -> float:
    thresholds = np.linspace(0, 1, int(1.0 / step + 1))
    scores = np.array([f1_score(y_true, y_pred > it) for it in thresholds])
    return thresholds[np.argmax(scores)]


def find_thresholds(y_true: np.ndarray, y_pred: np.ndarray, step: float) -> List[float]:
    return [find_threshold(y_true[:, it], y_pred[:, it], step) for it in range(28)]


def apply_thresholds(y_pred: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(y_pred)

    for i, th in enumerate(thresholds):
        pred[:, i] = y_pred[:, i] > th

    return pred


def eval_thresholds(y_true: np.ndarray, y_pred: np.ndarray, thresholds: np.ndarray, tag: str):
    f1_init = f1_score(y_true, y_pred > 0.5, average='macro')
    f1 = f1_score(y_true, apply_thresholds(y_pred, thresholds), average='macro')
    f1_delta = f1 - f1_init
    f1_inc = 100.0 * f1_delta / f1_init

    logger = get_logger()
    logger.info(f'Init {tag} F1-score: {f1_init:0.4}')
    logger.info(f'New {tag} F1-score: {f1:0.4}')
    logger.info(f'Delta {tag} F1-score: {f1_delta:0.4} ({f1_inc:0.2}%)')


def soft_f1(y_pred: np.ndarray,
            y_true: np.ndarray,
            threshold: Union[float, np.ndarray] = 0.0,
            alpha: float = 25.0):
    y_pred = sigmoid(alpha * (y_pred - threshold))
    y_true = y_true.astype(np.float)
    score = 2.0 * (y_pred * y_true).sum(axis=0) / ((y_pred + y_true).sum(axis=0) + 1e-6)
    return score


def fit_valid_thresholds(y_pred: np.ndarray, y_true: np.ndarray):
    def loss(x: np.ndarray):
        return np.concatenate((soft_f1(y_pred, y_true, x) - 1.0, wd * x), axis=None)

    params = np.zeros(28)
    wd = 1e-5
    x, success = opt.leastsq(loss, params)
    return x
