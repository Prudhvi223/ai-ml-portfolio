# Day 38 — RNNs, LSTMs and GRUs

## Objective

Understand sequence modeling and how recurrent neural networks process sequential data.

## Topics Covered

- Sequence modeling
- Recurrent Neural Networks
- Hidden states
- Vanishing gradients
- Long-term dependencies
- LSTM
- GRU

## Hands-on Work

Implemented RNN, LSTM and GRU models using PyTorch and compared their output shapes and memory mechanisms.

## Key Learnings

- RNNs process data sequentially using hidden states.
- Vanilla RNNs struggle with long-term dependencies because of the vanishing gradient problem.
- LSTMs use cell states and gates to maintain important information for longer sequences.
- GRUs provide a simpler gated architecture with similar benefits.

## Conclusion

RNNs introduced memory into neural networks for sequence modeling, while LSTMs and GRUs improved the ability to learn long-term dependencies.