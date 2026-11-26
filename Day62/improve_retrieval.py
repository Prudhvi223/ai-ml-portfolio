
import re
import chromadb
import ollama

from rank_bm25 import BM25Okapi


# 1. Connect to Day 61 ChromaDB
client = chromadb.PersistentClient(
    path="Day61/chroma_db"
)

collection = client.get_collection(
    name="pdf_knowledge"
)

# Get stored documents
stored = collection.get(
    include=["documents"]
)

documents = stored["documents"]
ids = stored["ids"]

print("Documents available:", len(documents))


# 2. Prepare BM25 keyword search
def tokenize(text):
    return re.findall(r"\w+", text.lower())


tokenized_docs = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)


# 3. User query
query = input("\nAsk a question: ")

# BM25 results
bm25_scores = bm25.get_scores(tokenize(query))
bm25_ranked = sorted(
    range(len(documents)),
    key=lambda i: bm25_scores[i],
    reverse=True
)

# Vector search
query_embedding = ollama.embed(
    model="nomic-embed-text",
    input=query
)["embeddings"][0]

vector_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(3, len(documents))
)

vector_ids = vector_results["ids"][0]


# 4. Hybrid search using Reciprocal Rank Fusion
rrf_scores = {}

# Add BM25 results using actual ChromaDB IDs
for rank, doc_index in enumerate(bm25_ranked, start=1):
    doc_id = ids[doc_index]

    rrf_scores[doc_id] = (
        rrf_scores.get(doc_id, 0) + 1 / (60 + rank)
    )

# Add vector search results
for rank, doc_id in enumerate(vector_ids, start=1):
    rrf_scores[doc_id] = (
        rrf_scores.get(doc_id, 0) + 1 / (60 + rank)
    )

# Sort by combined RRF score
hybrid_ranked = sorted(
    rrf_scores,
    key=rrf_scores.get,
    reverse=True
)

# Retrieve corresponding documents
id_to_document = dict(zip(ids, documents))

candidates = [
    id_to_document[doc_id]
    for doc_id in hybrid_ranked[:3]
]

print("\nHybrid Search Results:")

for i, doc in enumerate(candidates):
    print(f"\nChunk {i+1}: {doc}")


# 5. Rerank candidates using Ollama
rerank_prompt = f"""
Question: {query}

Choose the most relevant chunk from the numbered candidates.
Return only its number.

Candidates:
"""

for i, doc in enumerate(candidates, start=1):
    rerank_prompt += f"\n{i}. {doc}\n"

rerank_response = ollama.chat(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": rerank_prompt}]
)

selected = rerank_response["message"]["content"].strip()

match = re.search(r"\b([1-3])\b", selected)

if match and int(match.group(1)) <= len(candidates):
    best_chunk = candidates[int(match.group(1)) - 1]
else:
    best_chunk = candidates[0]

print("\nReranked Best Chunk:")
print(best_chunk)


# 6. Generate answer using selected context
answer_response = ollama.chat(
    model="llama3.2:1b",
    messages=[{
        "role": "user",
        "content": f"""
Answer the question using ONLY the context.
If the answer is missing, say you cannot find it in the PDF.

Context:
{best_chunk}

Question:
{query}
"""
    }]
)

answer = answer_response["message"]["content"]

print("\nGenerated Answer:")
print(answer)


# 7. Basic model-assisted evaluation
evaluation = ollama.chat(
    model="llama3.2:1b",
    messages=[{
        "role": "user",
        "content": f"""
Evaluate this answer.

Question: {query}
Context: {best_chunk}
Answer: {answer}

Check:
1. Faithfulness: Is the answer supported by the context?
2. Relevancy: Does it answer the question?

Give each a score from 1 to 5 and a short reason.
"""
    }]
)

print("\nBasic Evaluation:")
print(evaluation["message"]["content"])
