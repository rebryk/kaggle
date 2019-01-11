# [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)
![Public LB](https://img.shields.io/badge/public%20LB-0.595-orange.svg)
![Private LB](https://img.shields.io/badge/private%20LB-0.523-orange.svg)
![Place](https://img.shields.io/badge/place-92-blue.svg)
![Silver medal](https://img.shields.io/badge/medal-silver-c0c0c0.svg)
<!--- ![Bronze medal](https://img.shields.io/badge/medal-bronze-cd7f32.svg) -->
<!--- ![Gold medal](https://img.shields.io/badge/medal-gold-ffd700.svg) -->

## Overview
#### Data
- Channels: RGB
- Oversampling
- External data: http://v18.proteinatlas.org

#### Augmentation
- Resize, Rotate, RandomRotate90, HorizontalFlip, RandomBrightnessContrast, Normalize

#### Model design
- Backbone: Resnet50 pretrained on ImageNet
- Head: 2 linear layers with batch normalization and dropout

#### Loss
- Binary Cross Entropy Loss

#### Training
- 5-fold CV
- Optimizer: Adam
- Different learning rates for different layers
- Head fine-tuning with frozen backbone (`1` epoch)
- Scheduler: Cyclical Learning Rates

Stage 1:
- Image size: `256`
- Batch size: `128`
- Epochs: `16`

Stage 2:
- Image size: `512`
- Batch size: `32`
- Epochs: `6`

#### Prediciton
- TTA: `8`
- TTA augmentation: Resize, Rotate, RandomRotate90, HorizontalFlip, Normalize
- The mean of the predictions
- Threshold: `0.2`

#### Result
- Training takes `~35` hours on Tesla v100
- Public LB: `0.595`
- Private LB: `0.523`

## Observations
- Mixed precision works poorly
- External data helps a lot
- BCE Loss with oversampling is much better than [Focal Loss](https://arxiv.org/abs/1708.02002)
- `Resnet50` outperforms `Resnet18` and `Resnet34`
- 5 folds improve score by `0.024`
- TTA helps too

## Installation
First, clone the repository
```
git clone https://github.com/rebryk/kaggle.git
cd kaggle/human-protein
```

Second, install requirements
```
pip install -r requirements.txt
```

Third, install [apex](https://github.com/NVIDIA/apex.git)
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

And last but not least, update config files in the `configs` folder to match your preferences!

### External data
Use `scripts/external_data.py` and `scripts/convert_data.py` to download and convert external data.

### Training
```
# Stage 1
cp configs/train256.py config.py
python train.py

# Stage 2
cp configs/train512.py config.py
python train.py
```

### Prediction
```
cp configs/test.py config.py
python test.py
```

Submissions are saved in the `submissions` folder.
