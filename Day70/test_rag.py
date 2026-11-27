from rag import RAGRetriever

retriever = RAGRetriever()

question = "What is Artificial Intelligence?"

results = retriever.retrieve(question)

print("\nRETRIEVED DOCUMENTS\n")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("Source:", result["source"])
    print("Page:", result["page"])
    print("RRF Score:", result["score"])
    print("Content:", result["text"][:300])
