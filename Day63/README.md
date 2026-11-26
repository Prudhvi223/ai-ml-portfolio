# Day 63 – Project 4: RAG Application

## Overview
Completed a PDF Question Answering application using Retrieval-Augmented Generation (RAG).

## Technologies
- Python
- LangChain
- LlamaIndex (previous ingestion pipeline)
- ChromaDB
- Ollama
- Streamlit

## Features
- PDF document ingestion
- Text chunking and embeddings
- Persistent vector database
- Semantic retrieval
- LangChain workflow orchestration
- Local LLM answer generation
- Interactive Streamlit chat interface
- Retrieved source inspection

## How to Run

Install dependencies:

```bash
python -m pip install -r Day63/requirements.txt
```

Ensure Ollama is running and the required models are installed.

Run:

```bash
streamlit run Day63/app.py
```

## Outcome
Successfully integrated LangChain with ChromaDB and Ollama to build an interactive RAG-based PDF question answering application.