
import chromadb
import ollama

# 1. Create persistent ChromaDB client
client = chromadb.PersistentClient(
    path="Day59/chroma_db"
)

# 2. Create a collection
collection = client.get_or_create_collection(
    name="day59_documents"
)

# 3. Documents
documents = [
    "Python is a programming language.",
    "Machine learning learns from data.",
    "Deep learning uses neural networks.",
    "Cricket is a popular sport.",
    "Football is played between two teams."
]

# 4. Generate embeddings
response = ollama.embed(
    model="nomic-embed-text",
    input=documents
)

embeddings = response["embeddings"]

# 5. Insert into ChromaDB
collection.upsert(
    ids=["1", "2", "3", "4", "5"],
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"category": "Programming"},
        {"category": "AI"},
        {"category": "AI"},
        {"category": "Sports"},
        {"category": "Sports"}
    ]
)

print("Total stored records:", collection.count())
print("Insertion completed!")


# 6. Search query
query = "How do computers learn from data?"

# 7. Convert query into embedding
query_response = ollama.embed(
    model="nomic-embed-text",
    input=query
)

query_embedding = query_response["embeddings"][0]

# 8. Search ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# 9. Display retrieved documents
print("\nSearch Query:", query)
print("\nRetrieved Documents:")

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i+1}:")
    print(document)
    print("Metadata:", results["metadatas"][0][i])


# 10. Search only AI category documents

filtered_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"category": "AI"}
)

print("\nFiltered Results (AI only):")

for document, metadata in zip(
    filtered_results["documents"][0],
    filtered_results["metadatas"][0]
):
    print("\nDocument:", document)
    print("Metadata:", metadata)


# 11. Update an existing document

updated_document = "Python is widely used for AI and machine learning."

updated_response = ollama.embed(
    model="nomic-embed-text",
    input=updated_document
)

updated_embedding = updated_response["embeddings"][0]

collection.update(
    ids=["1"],
    documents=[updated_document],
    embeddings=[updated_embedding],
    metadatas=[{"category": "AI"}]
)

print("\nDocument updated successfully!")

# 12. Delete a document

collection.delete(ids=["5"])

print("Document deleted successfully!")
print("Remaining records:", collection.count())
