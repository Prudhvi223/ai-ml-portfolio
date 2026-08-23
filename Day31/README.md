# Day 31 — Transfer Learning

## Objective

Fine-tune a pretrained ResNet18 model on the CIFAR-10 dataset.

## Dataset

CIFAR-10:
- 50,000 training images
- 10,000 testing images
- 10 classes

## Model

Used pretrained ResNet18 from torchvision.

The pretrained layers were frozen and the final fully connected layer was replaced:

512 → 10

## Training

- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 5
- Loss: CrossEntropyLoss
- Batch size: 64

## Transfer Learning Process

1. Load pretrained ResNet18
2. Freeze pretrained layers
3. Replace final classifier
4. Train the new classifier
5. Evaluate on CIFAR-10 test set
6. Save the trained model

## Result

Test Accuracy: XX.XX%