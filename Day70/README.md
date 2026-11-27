# AI Research Assistant

An AI-powered research assistant that answers questions using uploaded PDF documents, web search, or both. It uses an agentic workflow to select the appropriate research sources and generate evidence-grounded reports.

## Features

- **Agentic workflow:** Uses LangGraph to coordinate research tasks.
- **PDF question answering:** Upload PDFs and retrieve relevant information.
- **Hybrid search:** Combines BM25 keyword retrieval and vector similarity search.
- **Reciprocal Rank Fusion:** Combines rankings from keyword and vector retrieval.
- **Web research:** Retrieves external information using DuckDuckGo search.
- **Automatic source selection:** Chooses PDF search, web search, or both.
- **Fallback retrieval:** Uses web search when relevant PDF evidence is unavailable.
- **Research reports:** Generates structured reports with source references.

## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- ChromaDB
- Ollama
- Groq API
- Hugging Face Sentence Transformers
- BM25 and Reciprocal Rank Fusion
- DuckDuckGo Search

## Architecture

1. User submits a research question.
2. The planner determines which sources are required.
3. PDF documents are processed into smaller chunks.
4. Embeddings are generated and stored in ChromaDB.
5. Hybrid retrieval combines BM25 and vector search.
6. Web search retrieves external information when required.
7. LangGraph coordinates the workflow.
8. The language model generates the final research report.

## Local Setup

Clone the repository and install the dependencies:

```bash
pip install -r Day70/requirements.txt
```

Install and start Ollama, then download the required models:

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

Run the application from the repository root:

```bash
streamlit run Day70/app.py
```

## Cloud Deployment

The application supports cloud deployment using:

- Groq API for language generation.
- Hugging Face embeddings.
- Streamlit Community Cloud.

Configure these values in Streamlit Cloud Secrets:

```toml
DEPLOYMENT_MODE = "cloud"
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.1-8b-instant"
```

Never commit API keys or secret files to GitHub.

## Project Structure

```text
Day70/
├── agents/
│   └── planner.py
├── app.py
├── config.py
├── graph.py
├── ingestion.py
├── llm_factory.py
├── rag.py
├── report_generator.py
├── requirements.txt
├── state.py
├── tools.py
└── web.py
```

## Future Improvements

- Improve retrieval relevance evaluation.
- Add report export options.
- Add more advanced source verification.
- Deploy and monitor the application on Streamlit Community Cloud.

## Author

AI/ML Portfolio — Day 70 Capstone Project