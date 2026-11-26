# Day 61 – PDF Question Answering System using RAG

## Overview
Built a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents.

## Technologies Used
- Python
- LlamaIndex
- ChromaDB
- Ollama
- nomic-embed-text
- llama3.2:1b
- Streamlit
- PyPDF

## Features
- PDF document ingestion
- Text extraction and chunking
- Embedding generation
- Persistent vector storage using ChromaDB
- Semantic similarity retrieval
- Answer generation using a local LLM
- Interactive Streamlit chatbot
- Retrieved source inspection

## Project Structure
```text
Day61/
├── data/
│   └── sample.pdf
├── chroma_db/
├── ingest.py
├── query.py
├── app.py
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
python -m pip install -r Day61/requirements.txt
```

Download models if not already installed:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

Start Ollama, then ingest PDFs:

```bash
python Day61/ingest.py
```

Run terminal question answering:

```bash
python Day61/query.py
```

Run the Streamlit application:

```bash
streamlit run Day61/app.py
```

## Outcome
Successfully developed a local RAG-based PDF question answering application using LlamaIndex, ChromaDB, Ollama and Streamlit.