# Day 34 — Project 2: Tomato Disease Classification

## Objective

Train a pretrained ResNet18 model to classify tomato leaf images
into three disease categories.

## Dataset

The dataset was prepared on Day 33.

Classes:

- Tomato Early Blight
- Tomato Healthy
- Tomato Late Blight

## Model

ResNet18 pretrained on ImageNet.

The final fully connected layer was replaced to support 3 classes.

## Training

- Optimizer: Adam
- Learning Rate: 0.0001
- Loss: CrossEntropyLoss
- Batch Size: 32
- Epochs: 5
- Input Size: 224 × 224

## Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Explainability

Grad-CAM was used to visualize the image regions that contributed
to the model's prediction.

## Key Learning

This project demonstrates transfer learning, image classification,
model evaluation, and explainable AI using Grad-CAM.