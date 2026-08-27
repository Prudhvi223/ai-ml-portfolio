# Day 33 - Project 2 Kickoff

## Project
Tomato Disease Classification

## Objective
Prepare a real-world image dataset for a tomato disease
classification project.

## Classes
- Tomato Early Blight
- Tomato Healthy
- Tomato Late Blight

## Dataset
The dataset contains training and validation images
from the PlantVillage dataset.

## Dataset Statistics

### Training
- Early Blight: 800
- Healthy: 1273
- Late Blight: 1527
- Total: 3600

### Validation
- Early Blight: 200
- Healthy: 318
- Late Blight: 382
- Total: 900

## Preprocessing
- Resize images to 224 x 224
- Convert images to PyTorch tensors
- RGB images

## PyTorch Pipeline
The dataset is loaded using `torchvision.datasets.ImageFolder`
and batches are created using PyTorch `DataLoader`.

## Current Status
Dataset preparation and loading pipeline completed.

## Next Step
Build and train a CNN model for tomato disease classification.