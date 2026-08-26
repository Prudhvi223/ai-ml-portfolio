# Day 32 — Data Augmentation

## Objective

Study data augmentation for computer vision and measure
its impact on CIFAR-10 classification accuracy.

## Techniques Used

- RandomCrop
- RandomHorizontalFlip
- RandomRotation
- ToTensor
- Normalization

## Experiments

### Baseline

The CNN was trained on CIFAR-10 without random augmentation.

Accuracy:
XX%

### With Data Augmentation

The same CNN was trained using:

- RandomCrop
- RandomHorizontalFlip
- RandomRotation

Accuracy:
YY%

## Results

| Experiment | Accuracy |
|------------|----------|
| Without Augmentation | XX% |
| With Augmentation | YY% |

## Conclusion

Data augmentation introduces additional variation into
the training data and can improve the model's ability
to generalize to unseen images.

Training augmentation was applied only to the training
dataset, while the test dataset was kept unchanged for
fair evaluation.