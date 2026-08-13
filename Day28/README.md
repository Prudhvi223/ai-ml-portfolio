# Day 28 — MNIST Digit Classifier

## Project

Built a handwritten digit classifier using PyTorch and the MNIST dataset.

## Dataset

MNIST contains handwritten images of digits from 0 to 9.

- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10

## Model

A feedforward neural network was used.

Architecture:

28 × 28
↓
Flatten
↓
128 neurons + ReLU
↓
64 neurons + ReLU
↓
10 output classes

## Training

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Epochs: 5
- Batch size: 64

## Evaluation

The trained model was evaluated on the MNIST test set using classification accuracy.

## Training Curves

Training loss and accuracy were plotted during training.

## How to Run

Install dependencies:

```bash
pip install torch torchvision matplotlib