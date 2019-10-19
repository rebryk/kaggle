# Speed Challenge 2017
![Train MSE](https://img.shields.io/badge/Train%20MSE-0.369-orange.svg)
![Valid MSE](https://img.shields.io/badge/Valid%20MSE-2.176-orange.svg)

<!--- ![Silver medal](https://img.shields.io/badge/medal-silver-c0c0c0.svg) -->
<!--- ![Bronze medal](https://img.shields.io/badge/medal-bronze-cd7f32.svg) -->
<!--- ![Gold medal](https://img.shields.io/badge/medal-gold-ffd700.svg) -->

## Overview
#### Data
Images are grouped into pairs of two successive frames, and the Farneback method is used to generate an optical flow.
The speed, that should be predicted, is equal to average speed of two frames.<br>
First 90% of the images represent training set, the remaining 10% - validation set. 
Images are not shuffled before split, because if two successive pairs are in training and validation set, respectively, the validation loss is low (inputs and targets are highly correlated), but it doesn't mean that model is accurate. 

Images are cropped to:
* Remove black stripe at the top of the image
* Remove the hood of the car
* Remove stripes on the left and right sides of the image

#### Model
There are many studies on how to predict distance and so on. But I decided to go with a simple and straightforward solution.
Of course, it is just ResNet-50 pre-trained on ImageNet!

#### Predictions
The model predictions cannot be used as final predictions, because:
* There are less groups than images: `#groups = #images - 1`
* It's not allowed to use future information

So all predictions are shifted to the right by one frame. The speed of the first frame is equal to the speed of the second frame.
It's not legal (because we actually need the second frame to be able to predict this speed), but one point doesn't affect loss value much.
 
The average of 10 previous frames (0.5 seconds) improves results.

|                   | Train MSE | Valid MSE |
|-------------------|:---------:|:---------:|
| Without averaging |   0.514   |   2.439   |
| With averaging    |   0.369   |   2.176   |

Test video contains some situations that are rare for training set.  For example, a long wait at a crossroads. Thus, erros is high.

<img width="638" alt="error" src="https://user-images.githubusercontent.com/4231665/67137092-a8e61800-f1e4-11e9-8a29-79e52c9b82a0.png">
 
#### Plots📈

![train](https://user-images.githubusercontent.com/4231665/67136921-9cf95680-f1e2-11e9-9ea5-5b16a308e2a0.png)
![valid](https://user-images.githubusercontent.com/4231665/67136933-b39fad80-f1e2-11e9-98d9-224bb5b085f9.png)
![test](https://user-images.githubusercontent.com/4231665/67136947-c9ad6e00-f1e2-11e9-8897-3e669860f67b.png)

#### Videos📹
You can open [this link](https://drive.google.com/drive/folders/1pUmcfEQLxDhYJQcymkJ8oBVE8Sb9EOZ2?usp=sharing) to find generated videos.

<img width="637" alt="video" src="https://user-images.githubusercontent.com/4231665/67136985-4a6c6a00-f1e3-11e9-8643-36f66e556cdc.png">

#### Future work
It would be nice to see how the model works if cars were removed from the images. To do this one can use some pre-trained models.
One can also use the model to detect lanes to remove all other objects from the scene.

## Installation
```
# Clone the repository
git clone https://github.com/rebryk/_kaggle.git
cd _kaggle/speed-challenge

# Install requirements
pip install -r requirements.txt

# Install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

Do not forget to update the data path in the configuration file in the `configs` folder!

### Checkpoint
If you don't want to train the model, you can find a checkpoint [here](https://drive.google.com/drive/folders/1pUmcfEQLxDhYJQcymkJ8oBVE8Sb9EOZ2?usp=sharing).

### Training
Run the following command to start single node, single GPU training:
```
python -m challenge/train.py -c challenge/configs/exp001.py
```

In case if you have a few GPUs use another command:
```
export N_GPU_NODE=<N_GPUS>;
export NODE_RANK=0;
export N_NODES=1;

python -m torch.distributed.launch 
    --nproc_per_node=<N_GPUS> 
    --nnodes=1 
    --node_rank 0
    --master_addr 0.0.0.0 
    --master_port 1234 
    challenge/train.py
    --n_gpu=<N_GPUS>
    --config challenge/configs/exp001.py
```

### Test
The following command will produce `pred_train.csv`, `pred_valid.csv` and `pred_test.csv`:
```
python -m challenge/test.py 
    -c challenge/configs/exp001.py 
    --checkpoint experiments/exp001/checkpoints/checkpoint_14.pth 
    --output experiments/exp001
```

You can run multi-GPU inference using another command:
```
export N_GPU_NODE=<N_GPUS>;
export NODE_RANK=0;
export N_NODES=1;

python -m torch.distributed.launch 
    --nproc_per_node=<N_GPUS>
    --nnodes=1 
    --node_rank=0 
    --master_addr 0.0.0.0 
    --master_port 1234 
    challenge/test.py 
    --n_gpu=<N_GPUS>
    -c challenge/configs/exp001.py 
    --checkpoint experiments/exp001/checkpoints/checkpoint_14.pth 
    --output experiments/exp001
```

### Predictions
Generated above files **are not predictions** for the original task! <br>
If you open `challenge/predictions.ipynb` Jupyter Notebook you can:
* make predictions for the original problem☺️
* draw plots📈
* generate nice videos📹
