
# Day 45 — NLP Evaluation

## Overview

On Day 45, I learned how to evaluate Natural Language Processing (NLP) models using different evaluation metrics.

Different NLP tasks require different evaluation methods. In this session, I learned F1 Score for classification, BLEU Score for text generation, and ROUGE Score for summarization.

## Topics Covered

- F1 Score for classification
- Precision and Recall
- BLEU Score for text generation
- N-gram overlap
- ROUGE-1, ROUGE-2, and ROUGE-L
- Comparing NLP evaluation metrics

## 1. F1 Score

F1 Score is used to evaluate classification models.

It combines Precision and Recall using the harmonic mean.

Formula:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

I implemented F1 Score using scikit-learn and evaluated actual and predicted labels.

## 2. BLEU Score

BLEU stands for Bilingual Evaluation Understudy.

It is commonly used to evaluate machine translation and text generation.

BLEU measures n-gram overlap between generated text and reference text.

I implemented BLEU Score using the NLTK library.

## 3. ROUGE Score

ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation.

It is commonly used for evaluating text summarization.

The main metrics I learned were:

- ROUGE-1: Word-level overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest Common Subsequence

I implemented ROUGE evaluation using the rouge-score library.

## Tools and Technologies

- Python
- Jupyter Notebook
- Scikit-learn
- NLTK
- Rouge-score
- Pandas

## What I Learned

- How to evaluate NLP classification models using F1 Score.
- The difference between precision and recall.
- How BLEU measures n-gram overlap.
- How ROUGE evaluates generated summaries.
- Why different NLP tasks require different evaluation metrics.
- The limitations of overlap-based evaluation metrics.

## Conclusion

Day 45 helped me understand how to evaluate NLP models using classification and text-generation evaluation metrics.

These evaluation methods are useful for analyzing model performance in NLP and Generative AI applications.
