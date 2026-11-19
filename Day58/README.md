# Day 58 – Embeddings and Semantic Search

## Overview
Explored text embeddings, cosine similarity, and semantic search using Ollama.

## Concepts Learned
- Sentence Transformers and embedding models
- Text-to-vector conversion
- Embedding dimensions
- Cosine similarity
- Keyword search vs semantic search
- Semantic search using 10 sentences

## Technologies
- Python
- Ollama
- nomic-embed-text

## Project Structure
```text
Day58/
├── semantic_search.py
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
python -m pip install -r Day58/requirements.txt
```

Download the embedding model:

```bash
ollama pull nomic-embed-text
```

Run:

```bash
python Day58/semantic_search.py
```

## Outcome
Successfully generated embeddings for 10 sentences, calculated cosine similarity, and retrieved the top 3 semantically similar sentences for a user query.