# Day 63 – Advanced AI PDF RAG Assistant

## Project Overview

This project is an AI-powered PDF Question Answering Assistant built using Retrieval-Augmented Generation (RAG).

It allows users to upload PDF documents and ask questions about their content. The application retrieves relevant information from the documents and uses a Large Language Model (LLM) to generate clear and context-aware answers.

The project combines the retrieval improvements from Day 62 with LangChain orchestration and a Streamlit chat interface.

## Technologies Used

- **Python** – Main programming language
- **Streamlit** – Interactive web interface
- **LangChain** – LLM orchestration and prompt management
- **ChromaDB** – Vector database for storing document embeddings
- **Ollama** – Runs local language and embedding models
- **BM25** – Keyword-based document retrieval
- **RRF (Reciprocal Rank Fusion)** – Combines keyword and vector search rankings
- **Reranking** – Selects the most relevant retrieved chunk

## Project Architecture

PDF Documents → Text Extraction → Chunking → Embeddings → ChromaDB → Hybrid Search → RRF → Reranking → LLM → Generated Answer → Streamlit UI

## Key Features

1. **PDF Question Answering:** Ask questions about indexed PDF documents.
2. **Vector Search:** Retrieves semantically similar document chunks.
3. **BM25 Search:** Finds relevant documents using keyword matching.
4. **Hybrid Search:** Combines BM25 and vector search.
5. **Reciprocal Rank Fusion:** Combines the rankings from both retrieval methods.
6. **Reranking:** Selects the most relevant chunk for answer generation.
7. **LLM Answer Generation:** Uses Ollama and LangChain to generate answers.
8. **Answer Evaluation:** Provides basic faithfulness and answer relevance evaluation.
9. **Chat Interface:** Supports interactive conversations through Streamlit.

## Project Structure

```text
Day63/
│
├── app.py
├── requirements.txt
└── README.md
```

The application uses the existing indexed PDF knowledge base created during Day 61 and the retrieval concepts implemented during Day 62.

## Installation

Install the required dependencies:

```bash
pip install -r Day63/requirements.txt
```

Install and run Ollama, then download the required models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Make sure the Day 61 ChromaDB knowledge base has already been created.

## Run the Application

From the root directory of the repository, execute:

```bash
streamlit run Day63/app.py
```

The application will open in your browser.

## Example Questions

- What is Artificial Intelligence?
- What is the difference between AI and ML?
- What is Deep Learning?
- Explain embeddings.
- What is Retrieval-Augmented Generation?

## Learning Outcomes

Through this project, I learned how to:

- Build an end-to-end RAG application.
- Combine keyword-based and semantic retrieval.
- Implement Reciprocal Rank Fusion.
- Understand and implement document reranking.
- Integrate LangChain with local Ollama models.
- Build an interactive AI application using Streamlit.
- Evaluate generated answers using basic faithfulness and relevance checks.

## Final Outcome

Successfully developed an Advanced AI PDF Assistant that combines PDF-based RAG, hybrid retrieval, reranking, LangChain, ChromaDB, Ollama, and Streamlit into an interactive question-answering application.