# Day 26 — Optimizers & Regularization

## Objective

Study different optimizers and regularization techniques in PyTorch.

## Topics

- SGD
- Adam
- Dropout
- Batch Normalization
- Overfitting
- Training vs Validation Loss

## Dataset

Breast Cancer Wisconsin dataset from scikit-learn.

## Experiments

### Experiment 1 — SGD

Trained a feedforward neural network using Stochastic Gradient Descent.

### Experiment 2 — Adam

Trained the same type of network using Adam and compared its convergence with SGD.

### Experiment 3 — Dropout

Added Dropout layers to reduce overfitting.

### Experiment 4 — Batch Normalization

Added Batch Normalization and Dropout to create a regularized neural network.

## Key Findings

- SGD and Adam use gradients to update model parameters.
- Adam uses adaptive updates and often converges faster.
- Dropout helps reduce over-dependence on individual neurons.
- Batch Normalization can make training more stable.
- Training and validation loss can be used to diagnose overfitting.