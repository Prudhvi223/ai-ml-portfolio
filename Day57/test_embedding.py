
import ollama


text = "Machine Learning is a subset of Artificial Intelligence."


response = ollama.embed(
    model="nomic-embed-text",
    input=text
)


embedding = response["embeddings"][0]


print("Original text:")
print(text)

print("\nEmbedding:")
print(embedding)

print("\nNumber of dimensions:")
print(len(embedding))
