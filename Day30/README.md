# Day 30 — CNN from Scratch

## Objective

Build and train a small Convolutional Neural Network using PyTorch
on the CIFAR-10 image classification dataset.

## Dataset

CIFAR-10 contains 60,000 color images belonging to 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The dataset contains:

- 50,000 training images
- 10,000 test images
- 32 × 32 RGB images
- 10 classes

## CNN Architecture

The model contains:

```text
Input: 3 × 32 × 32

Conv2d: 3 → 32
ReLU
MaxPool

Conv2d: 32 → 64
ReLU
MaxPool

Flatten

Linear: 4096 → 128
ReLU

Linear: 128 → 10