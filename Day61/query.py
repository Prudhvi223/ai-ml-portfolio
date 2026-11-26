
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    StorageContext
)

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# 1. Connect to existing ChromaDB
client = chromadb.PersistentClient(
    path="Day61/chroma_db"
)

collection = client.get_collection(
    name="pdf_knowledge"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# 2. Configure embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)


# 3. Load existing index
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
    embed_model=embed_model
)


# 4. Configure LLM
llm = Ollama(
    model="llama3.2:1b",
    request_timeout=120.0
)


# 5. Create query engine
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3
)


# 6. Ask a question
question = input("\nAsk a question about your PDF: ")

response = query_engine.query(question)

print("\nGenerated Answer:")
print(response)
