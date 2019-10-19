"""
Experiment 001

Pre-trained ResNet50 with Optical Flow Farneback.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from pretrainedmodels.models.torchvision_models import load_pretrained
from pretrainedmodels.models.torchvision_models import pretrained_settings

exp_name = Path(__file__).stem

root = '/tmp/speed-challenge/data'
path = f'experiments/{exp_name}'

batch_size = 32
batch_size_test = 128
num_epochs = 20
num_workers = 8
train_ratio = 0.9
log_interval = 10

lr = 0.001
max_grad_norm = 1
mixed_precision = False


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.size(0))


def get_model(pretrained='imagenet', num_classes=1000) -> nn.Module:
    model = models.resnet50(pretrained=False, num_classes=num_classes)

    if pretrained is not None:
        settings = pretrained_settings['resnet50'][pretrained]
        model = load_pretrained(model, num_classes, settings)

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 1),
        Squeeze()
    )

    return model


def get_criterion():
    return nn.MSELoss()


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)
