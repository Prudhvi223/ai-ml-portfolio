# Day 27 — Training Mechanics

## Topics Covered

- PyTorch training loop
- Forward pass
- Loss calculation
- Backpropagation
- Optimizer step
- Learning-rate scheduling
- Model checkpointing
- Saving and loading PyTorch models

## Training Loop

The training loop follows these steps:

1. Forward pass
2. Calculate loss
3. Clear gradients
4. Backpropagation
5. Update model parameters
6. Repeat for multiple epochs

## Learning Rate Scheduling

A StepLR scheduler was used to reduce the learning rate during training.

## Checkpointing

The trained model was saved using:

`torch.save()`

The model was loaded using:

`load_state_dict()`

## Result

The model successfully learned the relationship:

`y = 2x`

and the saved model was successfully loaded and used for prediction.