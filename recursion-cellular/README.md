# [Recursion Cellular Image Classification](https://www.kaggle.com/c/recursion-cellular-image-classification)
![Public LB](https://img.shields.io/badge/public%20LB-0.701-orange.svg)
![Private LB](https://img.shields.io/badge/private%20LB-0.959-orange.svg)
![Place](https://img.shields.io/badge/place-41-blue.svg)
![Silver medal](https://img.shields.io/badge/medal-silver-c0c0c0.svg)
<!--- ![Bronze medal](https://img.shields.io/badge/medal-bronze-cd7f32.svg) -->
<!--- ![Gold medal](https://img.shields.io/badge/medal-gold-ffd700.svg) -->

## Overview
#### Data
- 6 channels

#### Augmentation
- RandomScale, Rotate, HorizontalFlip, VerticalFlip, Resize, RandomBrightnessContrast, RandomGamma, Normalize

#### Model design
- Backbone: DenseNet201 pretrained on ImageNet
- Head: 2 linear layers with batch normalization

#### Loss
- Cross Entropy Loss

#### Training
- Optimizer: Adam
- Different learning rates for different layers
- Image size: `512`
- Batch size: `64`
- Epochs: `75`
- Finetuning for each cell type
- Mixed precision

#### Prediciton
- TTA: `10`
- Use embeddings instead of final probability scores 
- Run k-Nearest Neighbors for each cell type separately
- [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) is used to match cell types with plates, wells with siRNAs

#### Result
- Public LB: `0.701`
- Private LB: `0.959`

## Observations
- I didn't manage to leverage ArcFace
- Hungarian algorithm boosted score a lot
- TTA helps too

## Installation
First, clone the repository
```
git clone https://github.com/rebryk/kaggle.git
cd kaggle/recursion-cellular
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

### Training
```
python -m challenge/train.py -c challenge/configs/exp048.py

# Finetuning
python -m challenge/finetune.py -c challenge/configs/exp048.py --checkpoint experiments/exp048/checkpoints/main/checkpoint_75.pth
```

### Prediction
```
# Get embeddings
python -m challenge/embeddings.py -c challenge/configs/exp048.py --n_aug 10 --checkpoint <checkpoints>

# Get test predictions
python -m challenge/test.py -c challenge/configs/exp048.py --use_valid --n_neighbors 30 100 30 15
```
