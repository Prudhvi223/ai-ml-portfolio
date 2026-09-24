
# Day 44 — Other NLP Tasks

## Overview

On Day 44, I explored additional Natural Language Processing (NLP) tasks using pretrained transformer models from Hugging Face.

## Topics Covered

- Named Entity Recognition (NER)
- Text Summarization
- Hugging Face `pipeline()`
- Using pretrained transformer models
- Understanding model predictions and confidence scores

## 1. Named Entity Recognition (NER)

Named Entity Recognition identifies important entities in text and assigns categories to them.

### Common Entity Types

- `PER` — Person
- `ORG` — Organization
- `LOC` — Location
- `MISC` — Miscellaneous

### Implementation

I used a Hugging Face NER pipeline to identify entities in sample text.

```python
from transformers import pipeline

ner = pipeline(
    "ner",
    aggregation_strategy="simple"
)

text = """
Elon Musk founded SpaceX.
SpaceX is headquartered in California.
"""

entities = ner(text)

for entity in entities:
    print(entity)
```

## 2. Text Summarization

Text summarization generates a shorter version of a longer text while preserving important information.

I used a pretrained transformer summarization model.

### Implementation

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

text = """
Artificial intelligence is transforming many industries.
Machine learning allows computers to learn patterns from data.
Deep learning uses neural networks to solve complex problems.
Natural language processing helps computers understand human language.
These technologies are used in healthcare, education, finance,
and recommendation systems.
"""

summary = summarizer(
    text,
    max_length=50,
    min_length=15,
    do_sample=False
)

print(summary[0]["summary_text"])
```

## What I Learned

- NER extracts meaningful entities from text.
- Summarization reduces long text into a shorter version.
- Hugging Face pipelines simplify the use of pretrained NLP models.
- Model outputs should be checked because predictions can contain errors.
- Different pretrained models may produce different results.

## Tools and Technologies

- Python
- Hugging Face Transformers
- PyTorch
- Natural Language Processing

## Conclusion

Day 44 introduced practical NLP tasks beyond sentiment analysis and text classification. I explored Named Entity Recognition and text summarization using pretrained transformer pipelines.
