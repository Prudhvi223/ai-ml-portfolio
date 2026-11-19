
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding


# 1. Load documents using a data connector
documents = SimpleDirectoryReader(
    input_dir="Day60/data"
).load_data()

print("Documents loaded:", len(documents))


# 2. Configure chunking
splitter = SentenceSplitter(
    chunk_size=100,
    chunk_overlap=20
)


# 3. Configure Ollama embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

Settings.embed_model = embed_model


# 4. Create an index
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[splitter]
)

print("Index created successfully!")


# See how LlamaIndex splits the document
nodes = splitter.get_nodes_from_documents(documents)

print("\nTotal chunks created:", len(nodes))

for i, node in enumerate(nodes):
    print(f"\nChunk {i+1}:")
    print(node.text)


# 5. Retrieve relevant information
retriever = index.as_retriever(similarity_top_k=2)

results = retriever.retrieve(
    "What are neural networks?"
)

print("\nRetrieved information:")

for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(result.text)
