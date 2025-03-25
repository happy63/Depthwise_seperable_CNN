# Depthwise Separable Convolution on CIFAR-10

## Overview
This project implements a Convolutional Neural Network (CNN) using **Depthwise Separable Convolutions** to classify images from the **CIFAR-10 dataset**. Depthwise Separable Convolutions help reduce computational complexity while maintaining accuracy.

## Dataset
**CIFAR-10** consists of 60,000 32x32 color images in 10 different classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each class has 6,000 images, with 50,000 images for training and 10,000 for testing.

## Model Architecture
The model uses **Depthwise Separable Convolutions**, which decompose a standard convolution into:
1. **Depthwise Convolution**: Applies a single filter per input channel.
2. **Pointwise Convolution**: Uses 1x1 convolutions to combine the outputs from the depthwise step.

### Layers:
- Input Layer: 32x32x3
- Depthwise Separable Convolution Layers
- Batch Normalization & ReLU activation
- Max Pooling Layers
- Fully Connected (Dense) Layers
- Softmax Output Layer

## Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV (optional for visualization)

## Installation
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Training the Model
Run the following script to train the model:
```bash
python train.py
```

### Training Parameters:
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 50
- Batch Size: 64

## Evaluation
To test the model on the CIFAR-10 test set:
```bash
python evaluate.py
```

## Results
- Expected accuracy: ~85%
- Lower computational cost compared to standard CNNs

## References
- [Depthwise Separable Convolution Paper](https://arxiv.org/abs/1704.04861)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## License
This project is open-source under the MIT License.

