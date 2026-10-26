# Day 53 — LLM APIs

## Overview
Learned how Python applications communicate with Large Language Models (LLMs), generate structured outputs, and use tools through function calling.

For hands-on practice, I used **Ollama with Llama 3.2 1B** as a free local alternative to paid cloud APIs.

## Concepts Learned
- What APIs are and how request-response communication works
- Sending prompts to an LLM from Python
- Using Ollama's local chat API
- Generating structured JSON output
- Using Pydantic and JSON Schema
- Understanding tool/function calling
- Connecting Python functions to an LLM
- Building a reusable local LLM assistant

## Technologies Used
- Python
- Ollama
- Llama 3.2 1B
- Pydantic
- Jupyter Notebook

## Project Structure

```text
Day53/
├── LLM_APIs.ipynb
└── README.md
```

## Setup

1. Install Ollama from https://ollama.com/download
2. Download the model:

   ```bash
   ollama pull llama3.2:1b
   ```

3. Install the Python packages in your notebook:

   ```python
   %pip install ollama pydantic
   ```

4. Start the model:

   ```bash
   ollama run llama3.2:1b
   ```

5. Run the notebook cells to test chat requests, structured outputs, and tool calling.

## Key Takeaway

LLM applications can use APIs to communicate with models, request structured data, and invoke Python functions. Ollama provides a way to experiment with these concepts locally without paying for cloud API usage.

## Status
Completed the Day 53 hands-on exercises.
