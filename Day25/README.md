# Day 25 — Feedforward Neural Network

## Objective

Build and train a Feedforward Neural Network using PyTorch.

## Dataset

Breast Cancer Wisconsin dataset from scikit-learn.

- Samples: 569
- Features: 30
- Task: Binary classification

## Neural Network Architecture

30 Input Features
↓
Linear Layer (30 → 16)
↓
ReLU
↓
Linear Layer (16 → 8)
↓
ReLU
↓
Linear Layer (8 → 1)
↓
Binary Classification Output

## Training

- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 100

## Key Concepts

- Feedforward Neural Networks
- Forward propagation
- Backpropagation
- Autograd
- Binary classification
- Loss functions
- Optimizers
- Model evaluation