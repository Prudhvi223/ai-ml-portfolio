from tools import web_search

query = "What is Artificial Intelligence?"

results = web_search(
    query,
    max_results=3
)

print("\nWEB SEARCH RESULTS\n")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")

    print("Title:", result["title"])

    print("Description:", result["body"])

    print("URL:", result["url"])
