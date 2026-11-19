
# Sample text
with open("Day60/data/sample.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 1. Fixed-size chunking
def fixed_chunking(text, chunk_size=100):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


# 2. Overlapping chunking
def overlapping_chunking(text, chunk_size=100, overlap=20):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# Test fixed-size chunking
fixed_chunks = fixed_chunking(text)

print("FIXED-SIZE CHUNKING")
print("Total chunks:", len(fixed_chunks))

for i, chunk in enumerate(fixed_chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)


# Test overlapping chunking
overlap_chunks = overlapping_chunking(text)

print("\n\nOVERLAPPING CHUNKING")
print("Total chunks:", len(overlap_chunks))

for i, chunk in enumerate(overlap_chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
