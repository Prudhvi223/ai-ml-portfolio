# Day 39 — Attention Mechanism

## Objective

Understand the attention mechanism and why attention became an important alternative to recurrent sequence processing.

## Topics Covered

- Attention mechanism
- Self-attention
- Query, Key and Value
- Attention scores
- Softmax attention weights
- Scaled dot-product attention
- Limitations of recurrence

## Hands-on Work

Implemented a simplified self-attention calculation both manually and using PyTorch.

The implementation calculates:

1. Query-Key similarity scores
2. Scaled attention scores
3. Softmax attention weights
4. Weighted combination of Value vectors

## Key Learning

Attention allows a model to directly determine which parts of a sequence are important instead of relying entirely on sequential hidden-state propagation.

## Formula

Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V

## Conclusion

Attention provides a mechanism for modeling relationships between different positions in a sequence and forms the foundation of modern Transformer architectures.