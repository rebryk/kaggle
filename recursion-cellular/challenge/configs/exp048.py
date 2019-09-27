from pathlib import Path

import albumentations as album
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pretrainedmodels.models.torchvision_models import load_pretrained
from pretrainedmodels.models.torchvision_models import pretrained_settings
from torch.functional import F

from utils.model import get_group_params

exp_name = Path(__file__).stem

root = '/data/RCIC'
path = f'experiments/{exp_name}'
backup_path = f'{path}/checkpoints'


class DenseNet(nn.Module):
    def __init__(self, model):
        super(DenseNet, self).__init__()
        self.features = model.features
        self.last_linear = model.classifier

        self.input_space = getattr(model, 'input_space', 'RGB')
        self.input_size = getattr(model, 'input_size', [3, 224, 224])
        self.input_range = getattr(model, 'input_range', [0, 1])
        self.mean = getattr(model, 'mean', [0.485, 0.456, 0.406])
        self.std = getattr(model, 'std', [0.229, 0.224, 0.225])

    def logits(self, x):
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def densenet201(num_classes=1000, pretrained='imagenet'):
    model = models.densenet201(pretrained=False, num_classes=num_classes)

    if pretrained is not None:
        settings = pretrained_settings['densenet201'][pretrained]
        model = load_pretrained(model, num_classes, settings)

    w = model.features.conv0.weight
    model.features.conv0 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.features.conv0.weight = nn.Parameter(torch.cat((w, w), dim=1))

    model.classifier = nn.Sequential(
        nn.Linear(1920, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 1108)
    )

    return DenseNet(model)


def get_model():
    return densenet201(num_classes=1000, pretrained='imagenet')


def optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    param_groups = [
        [model.features.conv0, model.features.norm0],
        [model.features.denseblock1, model.features.transition1,
         model.features.denseblock2, model.features.transition2],
        [model.features.denseblock3, model.features.transition3,
         model.features.denseblock4, model.features.norm5],
        [model.last_linear]
    ]

    lr = 0.0010
    lrs = np.array([lr / 3, lr / 2, lr / 2, lr])

    return torch.optim.Adam(get_group_params(param_groups, lrs), lr=lr)


num_epochs = 100
batch_size = 64
inference_batch_size = 256
num_workers = 32
model = get_model
criterion = nn.BCEWithLogitsLoss

augmentation = album.Compose([
    album.RandomScale(),
    album.Rotate(),
    album.HorizontalFlip(),
    album.VerticalFlip(),
    album.Resize(height=512, width=512, interpolation=cv2.INTER_CUBIC),
    album.RandomBrightnessContrast(),
    album.RandomGamma(),
    album.Normalize(
        mean=(0.026, 0.058, 0.041, 0.026, 0.058, 0.041),
        std=(0.056, 0.056, 0.041, 0.057, 0.055, 0.042)
    )
])

train_aug = augmentation
valid_aug = augmentation
test_aug = augmentation

num_finetuning_epoch = 20
stages = [0, 100]

opt_level = 'O1'
