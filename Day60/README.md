# Day 60 – Chunking Strategies and LlamaIndex

## Overview
Explored different chunking strategies and implemented document loading, indexing, embeddings, and retrieval using LlamaIndex.

## Concepts Learned
- Fixed-size chunking
- Overlapping chunking
- Sentence-based chunking
- Semantic chunking
- LlamaIndex framework
- Data connectors
- Document loading
- SentenceSplitter
- Nodes and indexing
- Embedding integration with Ollama
- Information retrieval

## Technologies
- Python
- LlamaIndex
- Ollama
- nomic-embed-text

## Project Structure
```text
Day60/
├── data/
│   └── sample.txt
├── chunking.py
├── llamaindex_demo.py
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
python -m pip install -r Day60/requirements.txt
```

Download the embedding model:

```bash
ollama pull nomic-embed-text
```

Run chunking:

```bash
python Day60/chunking.py
```

Run LlamaIndex:

```bash
python Day60/llamaindex_demo.py
```

## Outcome
Successfully implemented fixed-size and overlapping chunking, loaded documents using LlamaIndex, generated embeddings using Ollama, created a searchable vector index, and retrieved relevant information using a retriever.