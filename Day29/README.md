# Day 29 — CNN Fundamentals

## Objective

Learn the fundamental components of Convolutional Neural Networks using PyTorch.

## Topics Covered

- Convolution
- Filters / kernels
- Feature maps
- ReLU activation
- Max pooling
- CNN architecture
- Feature map visualization

## Dataset

MNIST handwritten digit images were used for demonstration.

Each MNIST image has:

- 1 grayscale channel
- 28 × 28 pixels
- 10 possible classes

## Convolution

A convolution filter scans across an image and responds to local patterns.

The output produced by a filter is called a feature map.

## Filters

Multiple filters can be applied to the same image.

For example:

```text
1 image
   ↓
8 filters
   ↓
8 feature maps