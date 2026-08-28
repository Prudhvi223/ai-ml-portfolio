# Project 2 — Tomato Leaf Disease Classifier

## Overview

A deep learning image classification application that identifies
tomato leaf diseases using a pretrained ResNet18 model.

## Classes

The model classifies images into three categories:

- Tomato Early Blight
- Tomato Healthy
- Tomato Late Blight

## Model

ResNet18 with transfer learning.

The pretrained ImageNet model was adapted by replacing the final
fully connected layer with a 3-class classification layer.

## Dataset

The dataset was prepared during Day 33 and contains separate
training and validation directories.

## Training

- Model: ResNet18
- Framework: PyTorch
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Input size: 224 × 224
- Epochs: 5

## Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Explainability

Grad-CAM was used to visualize the image regions that influenced
the model's predictions.

## Deployment

A Streamlit application was created to allow users to upload
tomato leaf images and receive predictions with confidence scores.

## Project Structure

```text
Day35/
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
    ├── home.png
    ├── prediction.png
    └── gradcam.png