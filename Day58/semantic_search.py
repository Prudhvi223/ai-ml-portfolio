
import ollama
import math

# 10 example sentences
sentences = [
    "Machine learning allows computers to learn from data.",
    "Deep learning uses neural networks.",
    "Python is a popular programming language.",
    "Artificial intelligence makes machines intelligent.",
    "Football is played between two teams.",
    "Cricket is a popular sport in India.",
    "Neural networks are inspired by the human brain.",
    "Data science involves analyzing large datasets.",
    "A car uses an engine to move.",
    "Natural language processing helps computers understand text."
]

# Generate embeddings for all 10 sentences
response = ollama.embed(
    model="nomic-embed-text",
    input=sentences
)

embeddings = response["embeddings"]

print("Total sentences:", len(sentences))
print("Embedding dimensions:", len(embeddings[0]))

# Cosine similarity function


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    return dot / (mag_a * mag_b)


# User query
query = "How do computers learn from information?"

# Convert query into embedding
query_response = ollama.embed(
    model="nomic-embed-text",
    input=query
)

query_embedding = query_response["embeddings"][0]

# Compare query with every sentence
scores = []

for i, embedding in enumerate(embeddings):
    score = cosine_similarity(query_embedding, embedding)
    scores.append((score, sentences[i]))

# Sort by highest similarity
scores.sort(reverse=True)

print("\nUser Query:", query)
print("\nMost Similar Sentences:")

for score, sentence in scores[:3]:
    print(f"\nSimilarity: {score:.4f}")
    print("Sentence:", sentence)
