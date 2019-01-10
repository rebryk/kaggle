from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def predict_with_targets(model: nn.Module,
                         loader: DataLoader,
                         device: torch.cuda.device = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    if device:
        model.to(device)

    y_pred = []
    y_true = []

    with torch.no_grad():
        with tqdm(initial=0, leave=False, total=len(loader)) as progress_bar:
            for x, y in loader:
                if device:
                    x = x.to(device=device, non_blocking=False)

                y_pred.append(model(x).cpu().numpy())
                y_true.append(y)

                progress_bar.update(1)

    return np.concatenate(y_pred, axis=0), np.concatenate(y_true, axis=0)


def tta(model: nn.Module,
        loader: DataLoader,
        loader_aug: DataLoader,
        n_aug: int = 16,
        device: torch.cuda.device = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict with Test Time Augmentation (TTA).

    :param model: model to use
    :param loader: test data loader
    :param loader_aug: test augmented data loader
    :param n_aug: a number of augmentation images to use per original image
    :param device: device to use
    :return: scores and true targets
    """
    y_pred, y_true = predict_with_targets(model, loader, device=device)
    pred = [y_pred]
    pred_aug = [predict_with_targets(model, loader_aug, device)[0] for _ in tqdm(range(n_aug), leave=False)]
    return np.stack(pred + pred_aug, axis=-1), y_true
