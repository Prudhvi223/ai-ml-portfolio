
import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# 1. Load PDF documents
documents = SimpleDirectoryReader(
    input_dir="Day61/data"
).load_data()

print("Documents loaded:", len(documents))


# 2. Configure chunking
Settings.node_parser = SentenceSplitter(
    chunk_size=300,
    chunk_overlap=50
)


# 3. Configure embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)


# 4. Create ChromaDB database
client = chromadb.PersistentClient(
    path="Day61/chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_knowledge"
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# 5. Create index and store embeddings
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("PDF successfully indexed!")
print("Stored chunks:", collection.count())
