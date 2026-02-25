# CNN and Transfer Learning: CIFAR-10 Classification

A comparison of two approaches to image classification on the CIFAR-10 dataset:

1. **Custom CNN trained from scratch** — a 3-block convolutional network with batch normalization and dropout
2. **Transfer learning with ResNet-18** — a pre-trained ImageNet model with a frozen backbone and a fine-tuned classification head

## Results

| Model | Test Accuracy | Trainable Parameters | Training Time |
|---|---|---|---|
| Custom CNN | 75.94% | 815,018 | ~3.6 min |
| ResNet-18 (frozen) | 80.20% | 5,130 / 11.2M | ~7.5 min |

## Model Details

### Custom CNN
- 3 convolutional blocks (32 → 64 → 128 channels)
- Batch normalization and spatial dropout after each block
- Adam optimizer, StepLR scheduler, 15 epochs

### ResNet-18 Transfer Learning
- Pre-trained on ImageNet (weights frozen)
- Final FC layer replaced: 512 → 10 classes
- Only the new FC layer is trained (5,130 parameters)
- Input images resized to 224×224 for ResNet compatibility
- Adam optimizer, 5 epochs

## Setup

```bash
pip install torch torchvision matplotlib numpy
```

The CIFAR-10 dataset downloads automatically on first run.

## Usage

Open and run `cnn_transfer_learning.ipynb` in Jupyter. The notebook is self-contained and will download the dataset, train both models, and produce comparison plots.

Tested on Apple Silicon (MPS) — also compatible with CUDA and CPU.
