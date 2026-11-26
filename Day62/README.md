# Day 62 – Improving RAG Retrieval

## Overview
Improved the RAG retrieval pipeline using hybrid search, reranking, and basic answer evaluation.

## Technologies
- Python
- ChromaDB
- Ollama
- BM25
- Reciprocal Rank Fusion (RRF)

## Features
1. Keyword-based retrieval using BM25.
2. Semantic retrieval using vector embeddings.
3. Hybrid search combining both retrieval methods using RRF.
4. Reranking of retrieved chunks using Ollama.
5. Answer generation using the local Llama 3.2 model.
6. Basic faithfulness and relevancy evaluation.

## How to Run

Install dependencies:

```bash
python -m pip install -r Day62/requirements.txt
```

Run:

```bash
python Day62/improve_retrieval.py
```

## Concepts Learned
- BM25 keyword search
- Vector similarity search
- Hybrid retrieval
- Reciprocal Rank Fusion
- Reranking
- Faithfulness and answer relevancy

## Outcome
Successfully implemented an improved RAG retrieval pipeline combining keyword and semantic search, reranking retrieved chunks, and evaluating generated answers.