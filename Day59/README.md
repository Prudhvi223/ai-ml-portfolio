# Day 59 – Vector Databases

## Overview
Explored vector databases and implemented a local vector database using ChromaDB and Ollama.

## Concepts Learned
- Vector databases and their purpose
- ChromaDB collections
- Storing documents and embeddings
- Querying similar documents
- Metadata filtering
- Updating existing records
- Deleting records

## Technologies
- Python
- ChromaDB
- Ollama
- nomic-embed-text

## Project Structure
```text
Day59/
├── chroma_db/           # Local database (excluded from Git)
├── vector_database.py
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
python -m pip install -r Day59/requirements.txt
```

Download the embedding model:

```bash
ollama pull nomic-embed-text
```

Run the project:

```bash
python Day59/vector_database.py
```

## Operations Implemented
1. Insert documents and embeddings using `upsert()`.
2. Search documents using `query()`.
3. Filter records using metadata with `where`.
4. Update existing records using `update()`.
5. Delete records using `delete()`.
6. Count stored records using `count()`.

## Outcome
Successfully created a local ChromaDB vector database and implemented insertion, similarity search, metadata filtering, updating, and deletion of records.