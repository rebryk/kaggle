import pretrainedmodels
import torch
import torch.nn as nn

from challenge.module import AdaptiveConcatPool2d
from commons.utils.model import freeze


def load_model(model_name: str, num_classes: int, pretrained: str):
    return pretrainedmodels.__dict__[model_name](num_classes=num_classes, pretrained=pretrained)


def get_model(model_name: str,
              num_classes: int,
              num_channels: int = 3,
              dropout: float = 0.5,
              frozen: bool = True):
    model = load_model(model_name, num_classes=1000, pretrained='imagenet')

    if frozen:
        freeze(model)

    if num_channels == 4:
        w = model.conv1.weight
        model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.conv1.weight = nn.Parameter(torch.cat((w, w[:, 1:2, :, :]), dim=1))

    model.avgpool = nn.Sequential(AdaptiveConcatPool2d())

    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(4096),
        nn.Dropout(dropout),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )

    return model
