
# DAY 57 - RAG PIPELINE

import ollama
import chromadb
from pathlib import Path


# Step 1: Document ingestion

def load_document(file_path):

    with open(file_path, "r", encoding="utf-8") as file:

        text = file.read()

    return text


# Load our sample document

document_path = "Day57/data/sample.txt"

document_text = load_document(document_path)


print("Document loaded successfully!")
print("Total characters:", len(document_text))
print("\nDocument content:\n")
print(document_text)


# Step 2: Chunking

def create_chunks(text, chunk_size=300, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Create chunks from our document

chunks = create_chunks(document_text)

print("\nTotal chunks:", len(chunks))

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i + 1}:")
    print(chunk)


# Create a persistent ChromaDB database
client = chromadb.PersistentClient(path="Day57/chroma_db")

# Create a collection
collection = client.get_or_create_collection(
    name="my_documents"
)

# Convert each chunk into an embedding
embeddings = []

for chunk in chunks:
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )
    embeddings.append(response["embeddings"][0])

# Store chunks and embeddings in ChromaDB

# Store or update chunks in ChromaDB
collection.upsert(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("\nStored chunks in ChromaDB:", collection.count())

# Ask a question
query = "What is Retrieval Augmented Generation?"

# Convert question into an embedding
query_response = ollama.embed(
    model="nomic-embed-text",
    input=query
)

query_embedding = query_response["embeddings"][0]

# Search for the most relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nUser Question:", query)
print("\nRetrieved Chunks:")

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i+1}:")
    print(document)


# Combine retrieved chunks into context
context = "\n\n".join(results["documents"][0])

# Create prompt for the LLM
prompt = f"""
Answer the question using ONLY the context below.

If the answer is not available in the context,
say "I don't know based on the provided context."

Context:
{context}

Question:
{query}

Answer:
"""

# Generate answer using Ollama
response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nGenerated Answer:")
print(response["message"]["content"])
